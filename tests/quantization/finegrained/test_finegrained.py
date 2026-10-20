# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""The fine-grained quantizer (`FineGrainedConfig`): a plan mixing the format quantizers, the checkpoints that
read into one, and the conversions it builds. Each format's own coverage lives in its folder."""

import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

from torch import nn

from transformers import DeepseekV4Config
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import (
    TestCasePlus,
    require_torch,
)
from transformers.utils.quantization_config import (
    BlockScaledConfig,
    FineGrainedConfig,
    Fp8Config,
    Mxfp4Config,
)

from ...integrations.finegrained.utils import (
    _conversion_model,
    _quantizer_for,
)
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    FINEGRAINED,
    FORMATS,
)


@require_torch
class FineGrainedPlanTest(unittest.TestCase):
    """`FineGrainedConfig`: a plan of formats per module group, how a checkpoint that mixes formats reads into one,
    and what each plan entry loads."""

    def test_a_legacy_expert_dtype_becomes_the_two_entries_it_meant(self):
        """DeepSeek-V4 ships `quant_method="fp8"` and declares its mxfp4 experts on the MODEL
        config, because a flat config cannot say "the experts differ". `shared_experts` is DENSE
        despite the name, which is why the pattern matches a path segment and not a substring."""
        from transformers.quantizers.quantizers_utils import plan_entry_for

        config = DeepseekV4Config(expert_dtype="fp4").adapt_quantization_config(Fp8Config(scale_fmt="ue8m0"))
        self.assertIsInstance(config, FineGrainedConfig)
        for module, expected in (
            ("model.layers.3.mlp.experts", "mxfp4"),
            ("model.layers.3.mlp.experts.gate_up_proj", "mxfp4"),
            ("model.layers.3.self_attn.q_proj", "fp8"),
            ("model.layers.3.mlp.shared_experts.up_proj", "fp8"),
        ):
            self.assertEqual(plan_entry_for(config.dtype_plan, module).method.value, expected, module)
        # without one, block-FP8 stays block-FP8, in DeepSeek's layout
        plain = DeepseekV4Config().adapt_quantization_config(Fp8Config())
        self.assertEqual((type(plain), plain.checkpoint_format), (Fp8Config, "deepseek"))

    def test_deepseek_block_scales_load_onto_their_slot(self):
        """DeepSeek's checkpoints name a block scale `scale`: its producer layout renames it onto `weight_scale_inv`."""
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        quantizer = _quantizer_for(DeepseekV4Config().adapt_quantization_config(Fp8Config()))
        quantizer.pre_quantized = True
        renamings = quantizer.get_checkpoint_renamings()
        self.assertTrue(all(isinstance(c, WeightRenaming) for c in renamings))
        renamed, _ = rename_source_key("layers.0.attn.wq_a.scale", renamings, [])
        self.assertEqual(renamed, "layers.0.attn.wq_a.weight_scale_inv")

    def test_deepseek_v4_takes_one_config_whether_its_checkpoint_is_from_the_hub_or_a_save(self):
        """The checkpoint is read for the model, and a format config lends its loading attributes to the plan it
        becomes, so the class a user passes does not depend on whether the checkpoint still says `fp8` (the Hub) or
        already `finegrained` (a `save_pretrained`)."""
        from transformers.quantizers.auto import AutoHfQuantizer

        model_config = DeepseekV4Config(expert_dtype="fp4")
        hub = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}
        saved = model_config.adapt_quantization_config(AutoQuantizationConfig.from_dict(hub)).to_dict()
        for checkpoint in (hub, saved):
            resolved = model_config.adapt_quantization_config(AutoQuantizationConfig.from_dict(checkpoint))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                merged = AutoHfQuantizer.merge_quantization_configs(resolved, Fp8Config(dequantize=True))
            self.assertIsInstance(merged, FineGrainedConfig)
            self.assertTrue(merged.dequantize)

    def test_an_expert_dtype_in_the_quantization_config_reads_as_the_same_plan(self):
        """DeepSeek-V4.1 declares its FP4 experts in the quantization config instead of the model config: the same
        plan, block-FP8 at the checkpoint's block over W4A4 MXFP4 experts, whichever config the load is handed."""
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        shipped = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
        model_config = DeepseekV4Config()
        model_config.quantization_config = shipped
        for passed in (AutoQuantizationConfig.from_dict(shipped), Fp8Config(dequantize=True)):
            with self.subTest(passed=type(passed).__name__):
                config = model_config.adapt_quantization_config(passed)
                self.assertIsInstance(config, FineGrainedConfig)
                experts, rest = config.dtype_plan[EXPERTS_PATTERN], config.dtype_plan[".*"]
                # W4A4: the weights' own activation format
                self.assertEqual((type(experts), experts.activation_format), (Mxfp4Config, None))
                self.assertEqual(
                    (type(rest), rest.weight_block_size, rest.scale_fmt),
                    (Fp8Config, passed.weight_block_size, passed.scale_fmt),
                )
                # the loading options are the plan's, not an entry's
                self.assertEqual((config.dequantize, rest.dequantize), (passed.dequantize, False))

    def test_expert_dtype_reaches_a_config_quantizing_on_the_fly(self):
        """The checkpoint an on-the-fly quantization writes declares the model config's `expert_dtype`, and is read
        back with it: the experts are quantized to MXFP4 then, so the save loads as what it holds."""
        from transformers.quantizers.auto import get_hf_quantizer
        from transformers.quantizers.block_scaled_utils import BlockScaledQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        model_config = DeepseekV4Config(expert_dtype="fp4")
        with mock.patch.object(BlockScaledQuantizer, "validate_environment"):
            quantizer, model_config, _ = get_hf_quantizer(model_config, Fp8Config(), None, True, {})
        self.assertIsInstance(quantizer.quantization_config, FineGrainedConfig)
        self.assertIsInstance(quantizer.quantization_config.dtype_plan[EXPERTS_PATTERN], Mxfp4Config)
        self.assertFalse(quantizer.pre_quantized)

    def test_the_base_config_is_not_a_format(self):
        with self.assertRaises(TypeError):
            BlockScaledConfig()

    def test_ep_plans_use_masked_routing_for_every_format(self):
        configs = [spec.config_cls() for spec in FORMATS]
        configs.append(FineGrainedConfig(dtype_plan={r"\.experts($|\.)": Mxfp4Config(), ".*": Fp8Config()}))
        for config in configs:
            for router in ("gate", "router"):
                with self.subTest(config=type(config).__name__, router=router):
                    model = nn.Module()
                    model.config = SimpleNamespace(_experts_implementation=None)
                    model.layers = nn.ModuleList([nn.Module()])
                    model.layers[0].mlp = nn.Module()
                    setattr(model.layers[0].mlp, router, nn.Module())
                    model.layers[0].mlp.experts = nn.Module()
                    plan = {
                        "layers.*.mlp.experts": "ep_dispatch_experts",
                        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj": "grouped_gemm",
                    }
                    quantizer = _quantizer_for(config)
                    quantizer.update_parallel_plan(model, plan)
                    expected = {
                        "layers.*.mlp.experts": "moe_tp_experts",
                        f"layers.*.mlp.{router}": "ep_router",
                        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj": "grouped_gemm",
                    }
                    self.assertEqual(plan, expected)
                    quantizer.update_parallel_plan(model, plan)
                    self.assertEqual(plan, expected)


@require_torch
class FineGrainedPlanConversionTest(unittest.TestCase):
    """The conversions a plan of several formats builds."""

    @staticmethod
    def _split(conversions):
        from transformers.core_model_loading import WeightConverter, WeightRenaming

        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        return renamings, [c for c in conversions if isinstance(c, WeightConverter)]

    def test_a_rewritten_model_converter_keeps_its_placement(self):
        """Qwen4-Exp concatenates its ~95 GiB n-gram table on CPU (`force_cpu`); the rewrite that adds the scales
        keeps it there, both ways the quantizer rewrites a converter."""
        from transformers.core_model_loading import MergeModulelist, WeightConverter

        for dequantize in (False, True):
            with self.subTest(dequantize=dequantize):
                model_converter = WeightConverter(
                    "ngram_embeddings.*.weight", "ngram_embedding.weight", [MergeModulelist(dim=0)], force_cpu=True
                )
                quantizer = _quantizer_for(Fp8Config(dequantize=dequantize))
                (rewritten, *_) = quantizer.update_weight_conversions([model_converter], _conversion_model(quantizer))
                self.assertTrue(rewritten.force_cpu)


class FineGrainedTinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = FINEGRAINED


class FineGrainedLoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = FINEGRAINED
