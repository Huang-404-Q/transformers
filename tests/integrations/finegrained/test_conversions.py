# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The `finegrained` conversion ops a load and a save run whatever the format: the layout ops (gate|up
interleave, scale container, swizzle) and the ops each expert tensor takes from its module, and a core chain
composed with on-the-fly quantization. Each format's checkpoint layouts are tested in its own folder. Pure
torch: no kernel is loaded and no GPU is needed."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.finegrained.layers as fg
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.core_model_loading import (
    Chunk,
    PermuteForRope,
    WeightConverter,
    convert_and_load_state_dict_in_model,
)
from transformers.integrations.moe import use_experts_implementation
from transformers.modeling_utils import LoadStateDictConfig
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import (
    Fp8Config,
)

from .utils import (
    E,
    H,
    I,
    _Cfg,
    _format_config,
    _load_and_save,
    _moe_config,
    _MoeModel,
    _quantizer_for,
    fake_finegrained_kernel,
    patch_finegrained_kernel,
)


@use_experts_implementation(is_transposed=True)
class _TransposedExperts(torch.nn.Module):
    """GPT-OSS's layout: the stacked experts stored `(E, in, out)`."""

    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.empty(E, H, 2 * I))
        self.down_proj = torch.nn.Parameter(torch.empty(E, I, H))


@require_torch
class FineGrainedLayoutOpsTest(unittest.TestCase):
    """The module HOLDS its block scales swizzled (a 5-D Parameter) when its chain runs the tcgen05
    scaled-MMA under a triton dispatch on SM100 — keyed off the declared weight format, not the
    scale dtype (V4-style block-FP8 ships UE8M0 scales) — and affine otherwise; the loader's
    `FineGrainedSwizzleScales` op fills that layout and its reverse restores the checkpoint grid."""

    E, HIDDEN, INTER = 4, 256, 128  # gate_up rows 2*INTER = 256 and down rows HIDDEN = 256: whole 128-row blocks

    def _experts(self, weight_format, activation_format=None, impl="grouped_mm", sm100=True):
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = self.HIDDEN, self.INTER, self.E
        cfg._experts_implementation = impl
        with (
            mock.patch.object(fg, "is_sm100", return_value=sm100),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            experts = fg.FineGrainedExperts(
                cfg,
                quantization_config=_format_config(
                    weight_format,
                    activation_format=activation_format,
                    # the group formats fix their own block
                    **({"weight_block_size": (128, 128)} if weight_format == "fp8" else {}),
                ),
            )
        model = torch.nn.Module()
        model.config = cfg
        model.experts = experts
        return model, experts

    def test_deepgemm_backends_hold_gate_up_stacked(self):
        """DeepGEMM's grouped GEMM reads a block grid row-major, which interleaved rows do not follow."""
        for impl in ("deepgemm", "deepgemm_megamoe"):
            _, experts = self._experts("fp8", impl=impl)
            self.assertFalse(experts.holds_interleaved_gate_up, impl)

    def _op_kernel(self):
        kernel, _ = fake_finegrained_kernel()
        kernel.swizzle_mx_scales = mock.Mock(
            side_effect=lambda s: s.reshape(s.shape[0], s.shape[1] // 128, -1, 2, 256)
        )
        kernel.unswizzle_mx_scales = mock.Mock(
            side_effect=lambda s, rows, cols, num_experts=None: s.reshape(num_experts, rows, cols)
        )
        return kernel

    def test_reverse_op_restores_the_affine_grid_from_the_artifact_alone(self):
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

        kernel = self._op_kernel()
        reverse = FineGrainedSwizzleScales(hf_quantizer=None).reverse_op
        artifact = torch.zeros(4, 2, 2, 2, 256, dtype=torch.float8_e8m0fnu)
        affine = torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu)
        with patch_finegrained_kernel(kernel):
            out = reverse.convert({"experts.gate_up_proj_scale_inv": artifact}, model=None, full_layer_name="x")
            self.assertEqual(out["experts.gate_up_proj_scale_inv"].shape, (4, 256, 8))
            out = reverse.convert({"experts.gate_up_proj_scale_inv": affine}, model=None, full_layer_name="x")
            self.assertIs(out["experts.gate_up_proj_scale_inv"], affine)

    def test_on_the_fly_quantization_still_interleaves_expert_biases(self):
        from transformers.integrations.finegrained.conversions import FineGrainedDeserialize

        model, experts = self._experts("fp8", sm100=False)
        quantizer = _quantizer_for(experts.quantization_config)
        quantizer.pre_quantized = False
        bias = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256)
        key = "experts.gate_up_proj_bias"
        actual = FineGrainedDeserialize(quantizer).convert({key: bias}, model=model, full_layer_name=key)[key]
        gate, up = bias.chunk(2, dim=1)
        torch.testing.assert_close(actual, torch.stack((gate, up), dim=-1).flatten(1))

    def test_ops_refuse_a_list_of_several_tensors(self):
        """The loader hands a single-source tensor in a one-element list; an op that unwraps it must not
        silently keep the first of several."""
        from transformers.integrations.finegrained.conversions import (
            FineGrainedInputScalesSplit,
            FineGrainedPackedBlocks,
            FineGrainedQuantize,
            FineGrainedViewPackedInt8,
        )

        tensor = torch.zeros(2, 4, 32, 16, dtype=torch.uint8)
        for op in (
            FineGrainedPackedBlocks(None),
            FineGrainedViewPackedInt8(None),
            FineGrainedInputScalesSplit(None),
            FineGrainedQuantize(None),
        ):
            with self.subTest(op=type(op).__name__), self.assertRaises(ValueError):
                op.convert({"experts.down_proj": [tensor, tensor]})
        # a single target, re-keyed by the op before anything reads it
        with self.assertRaises(ValueError):
            FineGrainedViewPackedInt8(None).convert({"x": [tensor, tensor]}, target_patterns=["experts.down_proj"])


@require_torch
class FineGrainedLoaderChainTest(unittest.TestCase):
    """A core conversion chain and an on-the-fly quantization in the same load: the permute has to
    run on the full-precision tensor, and only the finegrained module of the three gets a scale."""

    def test_on_the_fly_quantization_blocks_each_projection_whole(self):
        """The gate|up interleave runs on the quantized weight and its grid, so every 32-row block holds rows of
        ONE projection, scaled by its own magnitude: a gate four orders below its up half keeps its values.
        Both stack layouts: a per-expert checkpoint concatenated by the model's converter, and one stored
        transposed `(E, H, 2I)`, which the quantization reads as rows."""
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize

        torch.manual_seed(0)
        gate, up, down = torch.randn(E, I, H) * 1e-5, torch.randn(E, I, H), torch.randn(E, H, I)
        dense = {"model.layers.0.self_attn.q_proj.weight": torch.randn(H, H)}
        per_expert = {
            f"model.layers.0.mlp.experts.{expert}.{proj}.weight": tensor[expert]
            for expert in range(E)
            for proj, tensor in (("gate_proj", gate), ("up_proj", up), ("down_proj", down))
        }
        transposed = {
            "model.layers.0.mlp.experts.gate_up_proj": torch.cat([gate, up], dim=1).transpose(1, 2).contiguous(),
            "model.layers.0.mlp.experts.down_proj": down.transpose(1, 2).contiguous(),
        }
        config = Fp8Config(weight_block_size=(32, 32))
        for layout, checkpoint, mapping in (("concatenated", per_expert, None), ("transposed", transposed, [])):
            with self.subTest(layout=layout):
                model = _MoeModel(_moe_config())
                if layout == "transposed":
                    model.model.layers[0].mlp.experts = _TransposedExperts(model.config)
                model, info, saved = _load_and_save(
                    {**dense, **checkpoint}, config, model, mapping, pre_quantized=False
                )
                self.assertEqual(info.conversion_errors, {})
                experts = model.model.layers[0].mlp.experts
                reloaded, info, _ = _load_and_save(saved, config)
                self.assertEqual((info.missing_keys, info.unexpected_keys, info.conversion_errors), (set(), set(), {}))
                restored = reloaded.model.layers[0].mlp.experts
                for name in ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv"):
                    torch.testing.assert_close(
                        getattr(restored, name).float(), getattr(experts, name).float(), rtol=0, atol=0
                    )

                self.assertTrue(experts.holds_interleaved_gate_up)
                for half, expected in ((0, gate), (1, up)):
                    held = FineGrainedDequantize(None)._dequantize_one(
                        experts.gate_up_proj[:, half::2], experts.gate_up_proj_scale_inv[:, half::2], torch.float32
                    )
                    error = ((held - expected).norm() / expected.norm()).item()
                    self.assertLess(error, 0.1, f"half {half}")

    def test_a_chunked_rope_permute_quantizes_only_the_finegrained_target(self):
        from transformers.integrations.finegrained import FineGrainedLinear
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        n_heads = 2
        head_dim = 4
        in_dim = 4
        out_dim = n_heads * head_dim
        block_size = (4, 4)

        class RopeProjector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim))

        class RopeSelfAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = FineGrainedLinear(
                    in_dim, out_dim, quantization_config=Fp8Config(weight_block_size=block_size)
                )
                self.k_proj = RopeProjector()
                self.v_proj = RopeProjector()

        class RopeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = RopeSelfAttn()

        class RopeModel(PreTrainedModel):
            base_model_prefix = "model"

            def __init__(self, config):
                super().__init__(config)
                self.layers = torch.nn.ModuleList([RopeLayer()])
                self.post_init()

            def _init_weights(self, module):
                pass  # every weight here comes from the state dict, and `normal_` has no fp8 kernel

        config = PreTrainedConfig()
        config.num_attention_heads = n_heads
        model = RopeModel(config)

        raw_q = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0],
                [0.5, -0.5, 0.5, -0.5],
                [-1.0, 1.0, -1.0, 1.0],
                [-0.5, 0.5, -0.5, 0.5],
                [1.0, 1.0, -1.0, -1.0],
                [0.5, 0.5, -0.5, -0.5],
                [-1.0, -1.0, 1.0, 1.0],
                [-0.5, -0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        raw_k = torch.arange(out_dim * in_dim, dtype=torch.float32).reshape(out_dim, in_dim)
        raw_v = torch.arange(out_dim * in_dim, dtype=torch.float32).reshape(out_dim, in_dim) + 100.0
        raw_qkv = torch.cat([raw_q, raw_k, raw_v], dim=0)
        state_dict = {"layers.0.self_attn.qkv_proj.weight": raw_qkv.clone()}

        quantizer_cls = type(
            "FineGrainedHfQuantizer",
            (),
            {
                "__init__": lambda self, bs=block_size: setattr(
                    self, "quantization_config", SimpleNamespace(weight_block_size=bs, scale_fmt="float")
                ),
                "param_needs_quantization": lambda self, _model, param_name: param_name.endswith("q_proj.weight"),
                "get_quantize_ops": lambda self: FineGrainedQuantize(self),
                "member_of": lambda self, _module: None,
                "pre_quantized": False,
            },
        )
        quantizer = quantizer_cls()

        weight_mapping = [
            WeightConverter(
                "self_attn.qkv_proj.weight",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0), PermuteForRope(permute_layer_names=["q_proj", "k_proj"])],
            )
        ]
        load_config = LoadStateDictConfig(weight_mapping=weight_mapping, hf_quantizer=quantizer)
        loading_info, _ = convert_and_load_state_dict_in_model(model, state_dict, load_config)

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        permute_op = PermuteForRope(permute_layer_names=["q_proj", "k_proj"])
        permute_op.config = model.config
        expected_q = permute_op._apply(raw_q)
        expected_k = permute_op._apply(raw_k)
        expected_v = raw_v  # V carries no RoPE, so the permute names only q and k

        model_state = model.state_dict()
        self.assertFalse(torch.allclose(raw_k, expected_k))
        torch.testing.assert_close(model_state["layers.0.self_attn.k_proj.weight"], expected_k)
        torch.testing.assert_close(model_state["layers.0.self_attn.v_proj.weight"], expected_v)

        q_weight_key = "layers.0.self_attn.q_proj.weight"
        scale_key = "layers.0.self_attn.q_proj.weight_scale_inv"
        self.assertIn(scale_key, model_state)
        self.assertEqual(model_state[q_weight_key].dtype, torch.float8_e4m3fn)
        self.assertEqual(model_state[q_weight_key].shape, torch.Size((out_dim, in_dim)))
        self.assertEqual(model_state[scale_key].dtype, torch.float32)
        self.assertEqual(
            model_state[scale_key].shape,
            torch.Size((out_dim // block_size[0], in_dim // block_size[1])),
        )

        dequantized_q = FineGrainedDequantize(None)._dequantize_one(
            model_state[q_weight_key], model_state[scale_key], output_dtype=torch.float32
        )
        torch.testing.assert_close(dequantized_q, expected_q, rtol=1e-2, atol=1e-2)
