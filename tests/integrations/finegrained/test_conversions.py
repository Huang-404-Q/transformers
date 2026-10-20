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
from unittest import mock

import torch

import transformers.integrations.finegrained.layers as fg
from transformers.integrations.moe import use_experts_implementation
from transformers.testing_utils import require_torch

from .utils import (
    E,
    H,
    I,
    _Cfg,
    _format_config,
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
