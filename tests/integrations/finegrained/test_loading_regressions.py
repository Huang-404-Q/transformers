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
"""CPU loading, inference, and saving against local quantized checkpoints with real model mappings."""

import os
import unittest

import torch
from safetensors.torch import load_file, save_file

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    NVFP4Config,
    Qwen2MoeConfig,
)
from transformers.integrations.finegrained.conversions import FineGrainedDequantize
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import FineGrainedConfig, Fp8Config

from .utils import _ChainAssertions


def _quantized_weights(model):
    """The dequantized weights the tiny checkpoints ship quantized: `q_proj` and both expert stacks."""
    suffixes = ("q_proj.weight", "experts.gate_up_proj", "experts.down_proj")
    weights = {name: tensor for name, tensor in model.state_dict().items() if name.endswith(suffixes)}
    assert len(weights) == len(suffixes), sorted(weights)
    return weights


@require_torch
class FineGrainedLoadingRegressionTest(_ChainAssertions, unittest.TestCase):
    def _checkpoint(self, directory, moe=False, mixed=False, modelopt_mixed=False):
        config_class = Qwen2MoeConfig if moe else LlamaConfig
        config = config_class(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            **({"num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 32} if moe else {}),
        )
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
        # Saving the model first supplies the architecture's checkpoint key layout, including split experts.
        model.save_pretrained(directory)
        filename = os.path.join(directory, "model.safetensors")
        state = load_file(filename)
        quantized = {}
        for name, tensor in state.items():
            if tensor.ndim != 2 or name.endswith(("embed_tokens.weight", "lm_head.weight", "mlp.gate.weight")):
                quantized[name] = tensor
                continue
            if modelopt_mixed and ".q_proj." not in name and ".mlp.experts." not in name:
                quantized[name] = tensor
                continue
            stem = name.removesuffix("weight")
            if modelopt_mixed and ".q_proj." in name:
                quantized[name] = torch.full_like(tensor, 2, dtype=torch.float8_e4m3fn)
                quantized[stem + "weight_scale"] = torch.tensor(1 / 128)
                quantized[stem + "input_scale"] = torch.tensor(1.0)
                continue
            ct = mixed and ".q_proj." in name
            quantized[stem + ("weight_packed" if ct else "weight")] = torch.full(
                (tensor.shape[0], tensor.shape[1] // 2), 0x22, dtype=torch.uint8
            )
            quantized[stem + "weight_scale"] = torch.ones(
                tensor.shape[0], tensor.shape[1] // 16, dtype=torch.float8_e4m3fn
            )
            quantized[stem + ("weight_global_scale" if ct else "weight_scale_2")] = torch.tensor(
                64.0 if ct else 1 / 64
            )
        config.quantization_config = (
            FineGrainedConfig(
                dtype_plan={r"\.q_proj$": NVFP4Config(checkpoint_format="compressed-tensors"), ".*": NVFP4Config()},
                modules_to_not_convert=["lm_head", "mlp.gate"],
            )
            if mixed
            else NVFP4Config(modules_to_not_convert=["lm_head", "mlp.gate"])
        )
        if modelopt_mixed:
            config.quantization_config = {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                    "model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
                },
            }
        config.save_pretrained(directory)
        save_file(quantized, filename, metadata={"format": "pt"})

    def test_mixed_block_sizes_use_the_current_modules_grid(self):
        # Both block sizes produce a 2x2 scale grid for 192x192 weights, but different boundaries.
        quantizer = FineGrainedHfQuantizer(
            FineGrainedConfig(
                dtype_plan={
                    "a$": Fp8Config(weight_block_size=(96, 96)),
                    "b$": Fp8Config(weight_block_size=(128, 128)),
                },
                dequantize=True,
            )
        )
        model = torch.nn.Module()
        model.a = torch.nn.Linear(192, 192, bias=False)
        model.b = torch.nn.Linear(192, 192, bias=False)
        scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        for name, block in (("a", 96), ("b", 128)):
            with self.subTest(module=name):
                result = FineGrainedDequantize(quantizer).convert(
                    {"weight": torch.ones(192, 192).to(torch.float8_e4m3fn), "weight_scale_inv": scales},
                    full_layer_name=f"{name}.weight",
                    model=model,
                )[f"{name}.weight"]
                expected = scales.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:192, :192]
                torch.testing.assert_close(result, expected)
