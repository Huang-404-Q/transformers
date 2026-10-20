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
"""Normalize producer metadata without silently changing module targets or quantization formats."""

import unittest

from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import NVFP4Config, read_compressed_tensors

from ...integrations.finegrained.utils import _compressed_tensors


@require_torch
class BlockScaledCheckpointReaderTest(unittest.TestCase):
    """modelopt and compressed-tensors NVFP4 exports read as `NVFP4Config`, skip-lists bounded to the modules they name."""

    def test_compressed_tensors_checkpoints_we_do_not_serve_stay_with_it(self):
        """Integer, per-channel FP8 and sparse checkpoints are compressed-tensors' own."""
        from transformers.utils.quantization_config import read_compressed_tensors

        int4 = {"num_bits": 4, "type": "int", "group_size": 128, "strategy": "group"}
        channel = {"num_bits": 8, "type": "float", "strategy": "channel"}
        nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16}
        for checkpoint in (
            _compressed_tensors(int4, fmt="pack-quantized"),
            _compressed_tensors(channel, fmt="float-quantized"),
            _compressed_tensors(nvfp4, sparsity_config={"format": "sparse-24-bitmask"}),
        ):
            self.assertIsNone(read_compressed_tensors(checkpoint))


class QuantizationConfigReaderRegistryTest(unittest.TestCase):
    def test_a_registered_reader_reads_its_method(self):
        from unittest import mock

        from transformers.quantizers import auto, register_quantization_config_reader

        with mock.patch.dict(auto.AUTO_QUANTIZATION_CONFIG_READERS):
            register_quantization_config_reader("some-tool")(
                lambda config: NVFP4Config() if config.get("fp4") else None
            )
            self.assertIsInstance(
                AutoQuantizationConfig.from_dict({"quant_method": "some-tool", "fp4": True}), NVFP4Config
            )
            self.assertTrue(auto.AutoHfQuantizer.supports_quant_method({"quant_method": "some-tool"}))
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_quantization_config_reader("some-tool")(lambda config: None)
            # a reader returning None leaves the method to its own config class, here none
            with self.assertRaisesRegex(ValueError, "Unknown quantization type"):
                AutoQuantizationConfig.from_dict({"quant_method": "some-tool"})


class FineGrainedConfigGroupReaderTest(unittest.TestCase):
    def _groups(self, reverse=False, identical=False):
        nvfp4 = {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float", "group_size": 16}}
        fp8 = {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        }
        groups = [("nvfp4", nvfp4), ("other", nvfp4 if identical else fp8)]
        return dict(reversed(groups) if reverse else groups)

    def test_compressed_tensors_keeps_its_backend_for_conflicting_group_targets(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.assertIsNone(
                    read_compressed_tensors(
                        {"format": "float-quantized", "config_groups": self._groups(reverse=reverse)}
                    )
                )


NVFP4 = {"num_bits": 4, "type": "float", "group_size": 16}
MODELOPT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


@require_torch
class ProducerSkipListTest(unittest.TestCase):
    """Producers name skipped modules as globs (modelopt's `exclude_modules` / `ignore`, a trailing `.` for a
    subtree) or exact names and `re:` patterns (compressed-tensors): each skips exactly the modules it names.
    Passed through as regexes the dots match any character and the star is greedy, so "model.layers.1.*" would
    also take layers 10-19."""
