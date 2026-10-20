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
"""What the `finegrained` integration hands `kernels-community/finegrained-kernels`: the kernel bundle is
mocked, so these pin the loader and the call contract (As-positional / no-block_size / b_global_scale) without
a GPU."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.deepgemm as deepgemm
import transformers.integrations.finegrained.layers as fg
from transformers.integrations.finegrained import (
    finegrained_linear,
    load_finegrained_kernel,
)
from transformers.testing_utils import require_torch

from ..integrations.finegrained.utils import fake_finegrained_kernel, patch_finegrained_kernel


@require_torch
class FineGrainedLoaderTest(unittest.TestCase):
    def test_loader_requires_every_symbol(self):
        kernel, _ = fake_finegrained_kernel()
        del kernel.matmul_grouped
        # Mock auto-creates attributes; force the miss
        kernel.matmul_grouped = None
        with patch_finegrained_kernel(kernel), self.assertRaises(ImportError) as ctx:
            load_finegrained_kernel()
        self.assertIn("matmul_grouped", str(ctx.exception))

    def test_loader_requires_only_the_symbols_it_calls(self):
        """A build with exactly the bundle's symbols loads: nothing the integration never calls is required."""
        kernel = SimpleNamespace(**{name: object() for name in fg.FineGrained.__dataclass_fields__})
        with patch_finegrained_kernel(kernel):
            self.assertIsInstance(load_finegrained_kernel(), fg.FineGrained)

    def test_loader_is_compile_safe_cold(self):
        """Cold path: the compiled call is first to load, so the opaque loader node runs its full
        body under compile and must return None, never the bundle (`torch.* op returned
        non-Tensor`). The loader has no arch gate, so nothing here fakes a device."""
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        with patch_finegrained_kernel(kernel):
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_is_compile_safe_when_warm(self):
        """Warm path, which is the production order: eager warm-up, then compile. The loader hits
        its short-circuit at trace time — the branch that must also return None."""
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        with patch_finegrained_kernel(kernel):
            load_finegrained_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_binds_all_symbols(self):
        kernel, _ = fake_finegrained_kernel()
        with patch_finegrained_kernel(kernel):
            bundle = load_finegrained_kernel()
        self.assertIs(bundle.matmul_2d, kernel.matmul_2d)
        self.assertIs(bundle.get_supported_act_fns, kernel.get_supported_act_fns)


@require_torch
class FineGrainedLinearMarshallingTest(unittest.TestCase):
    def _run(self, **linear_kwargs):
        kernel, rec = fake_finegrained_kernel()
        with patch_finegrained_kernel(kernel), mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            x = torch.randn(3, 5, 64, dtype=torch.bfloat16)
            w = torch.randn(32, 64).to(torch.float8_e4m3fn)
            ws = torch.randn(1, 1, dtype=torch.float32)
            out = finegrained_linear(x, w, ws, **linear_kwargs)
        return out, rec.calls["matmul_2d"][-1]

    def test_scale_rides_positionally_and_block_size_dies(self):
        out, call = self._run(block_size=[128, 128])
        # A is flattened 2D; As slot (positional 3rd) is the activation scale = None here
        self.assertEqual(call.args[0].shape, (15, 64))
        self.assertIsNone(call.args[2])
        self.assertEqual(call.args[3].dtype, torch.float32)
        self.assertNotIn("block_size", call.kwargs)
        self.assertEqual(out.shape, (3, 5, 32))

    def test_static_activation_scale_is_As(self):
        scale = torch.tensor(0.5)
        _, call = self._run(activation_scale=scale)
        self.assertIs(call.args[2], scale)

    def test_nvfp4_global_and_weight_only_format(self):
        g = torch.tensor(2.0)
        _, call = self._run(weight_global_scale=g, activation_format="bf16")
        self.assertIs(call.kwargs["b_global_scale"], g)
        self.assertEqual(call.kwargs["activation_format"], "bf16")
