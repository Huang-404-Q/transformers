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

from transformers.testing_utils import (
    require_torch,
)
from transformers.utils.quantization_config import (
    BlockScaledConfig,
    Fp8Config,
)

from ...integrations.finegrained.utils import (
    _conversion_model,
    _quantizer_for,
)


@require_torch
class FineGrainedPlanTest(unittest.TestCase):
    """`FineGrainedConfig`: a plan of formats per module group, how a checkpoint that mixes formats reads into one,
    and what each plan entry loads."""

    def test_the_base_config_is_not_a_format(self):
        with self.assertRaises(TypeError):
            BlockScaledConfig()


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
