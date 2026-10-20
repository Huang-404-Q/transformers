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
"""Integration tests for the DeepGEMM linear / experts dispatches.

Two layers, both mocking only what needs a Hopper/Blackwell GPU + a JIT CUDA toolkit +
`kernels-community/deep-gemm`:

* `DeepGemmLoaderTest` mocks the environment probes and `lazy_load_kernel` to drive
  `load_deepgemm_kernel` with no GPU/toolkit — asserting it gates correctly (raises when a
  precondition is unmet, returns the bundle otherwise, honours the per-call `requires_sm100` gate)
  and stays torch-compile safe.
* `DeepGemmForwardTest` mocks only the loaded-kernel ops (the `DeepGEMM` bundle callables:
  `per_token_cast_to_fp8`, `fp8_fp4_matmul`, the grouped GEMMs, the Mega MoE ops) to return
  correctly shaped tensors, and runs the REAL public forwards so their scale-factor coercion /
  TMA-aligned grouped layout / weight-selection / dtype-cast glue executes for real (device-agnostic,
  so it runs on CPU); it then asserts the tensors handed to the kernel are what the kernel expects
  (packed int32 UE8M0 SFs, int32 grouped layout, `(qtensor, sf)` operand tuples, recipes, transformed
  Mega MoE weights, ...).

Arch-gated paths are exercised by faking the device (`is_sm100()` reads availability, then the
capability): the SF-packing / TMA-alignment / psum-layout code it selects is pure tensor arithmetic,
so mocking both to a Blackwell device drives those paths anywhere, a CPU-only runner included.
"""

import contextlib
import inspect
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from parameterized import parameterized

import transformers.integrations.deepgemm as dg
import transformers.integrations.finegrained.layers as fg
from transformers.integrations.deepgemm import (
    deepgemm_bf16_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_greater_or_equal,
    torch_device,
)

from ..integrations.finegrained.utils import fake_finegrained_kernel, patch_finegrained_kernel
from .test_utils import make_experts


def _add_one(x, *args, **kwargs):
    return x + 1


# ── Fake kernels (a "good" one exposing every symbol the loader resolves, plus a variant missing one) ──


class _FakeDeepGemmKernel:
    fp8_fp4_gemm_nt = staticmethod(_add_one)
    m_grouped_fp8_fp4_gemm_nt_contiguous = staticmethod(_add_one)
    m_grouped_fp8_fp4_gemm_nn_contiguous = staticmethod(_add_one)
    m_grouped_bf16_gemm_nt_contiguous = staticmethod(_add_one)
    m_grouped_bf16_gemm_nn_contiguous = staticmethod(_add_one)
    per_token_cast_to_fp8 = staticmethod(_add_one)
    transform_sf_into_required_layout = staticmethod(_add_one)
    transform_weights_for_mega_moe = staticmethod(_add_one)
    get_symm_buffer_for_mega_moe = staticmethod(_add_one)
    fp8_fp4_mega_moe = staticmethod(_add_one)

    @staticmethod
    def get_mk_alignment_for_contiguous_layout():
        return 128


class _DeepGemmKernelMissingSymbol(_FakeDeepGemmKernel):
    fp8_fp4_mega_moe = None  # a resolved symbol that comes back as None


@require_torch
class DeepGemmLoaderTest(unittest.TestCase):
    def setUp(self):
        dg._DEEPGEMM = None
        self.addCleanup(setattr, dg, "_DEEPGEMM", None)

    def _env(
        self,
        *,
        kernels_available=True,
        cuda_available=True,
        capability=(9, 0),
        toolkit_present=True,
        nvcc_present=True,
        nvcc_version=(12, 9),
        kernel=_FakeDeepGemmKernel,
    ):
        stack = contextlib.ExitStack()
        # a REAL fake toolkit on disk (bin/nvcc touched iff nvcc_present) — faking the
        # filesystem state instead of file APIs keeps global machinery untouched
        # (patching `os.path.isfile` corrupts e.g. setuptools' distutils import shim
        # under torch.compile).
        cuda_home = None
        if toolkit_present:
            cuda_home = stack.enter_context(tempfile.TemporaryDirectory())
            if nvcc_present:
                os.makedirs(os.path.join(cuda_home, "bin"), exist_ok=True)
                open(os.path.join(cuda_home, "bin", "nvcc"), "w", encoding="utf-8").close()
        stack.enter_context(mock.patch.object(dg, "is_kernels_available", return_value=kernels_available))
        # Fake a "CUDA + `capability`" environment for the loader *only* (scoped to its call stack): the
        # loader's availability/arch gate passes, while torch.compile / inductor still see the real
        # device — so the compile-safety test runs under the default (inductor) backend on any host, not
        # just SM90+. The real fallbacks are lazy: only non-loader callers reach them (inductor, under
        # compile), never the CPU-only gating tests (where querying real CUDA would raise).
        real_is_available = torch.cuda.is_available
        real_capability = torch.cuda.get_device_capability

        def _in_loader():
            # The loader's only capability query happens inside `is_deepgemm_loadable` (which the gating
            # tests also call directly), so fake the device whenever that frame is on the stack. Inductor's
            # own capability query (under compile, outside that frame) still reads the real device.
            return any(f.function == "is_deepgemm_loadable" for f in inspect.stack())

        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                "is_available",
                side_effect=lambda *a, **k: cuda_available if _in_loader() else real_is_available(),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda *a, **k: capability if _in_loader() else real_capability(),
            )
        )
        stack.enter_context(mock.patch.object(dg, "_get_cuda_home", return_value=cuda_home))
        stack.enter_context(mock.patch.object(dg, "_get_nvcc_version", return_value=nvcc_version))
        stack.enter_context(mock.patch.object(dg, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = dg.load_deepgemm_kernel()
        self.assertIsInstance(bundle, dg.DeepGEMM)
        self.assertEqual(bundle.m_alignment, 128)

    @parameterized.expand(
        [
            ("valid_env", {}, None),
            ("no_kernels", {"kernels_available": False}, "kernel unavailable"),
            ("no_cuda", {"cuda_available": False}, "requires CUDA"),
            ("bad_arch", {"capability": (8, 0)}, "requires Hopper"),
            ("no_toolkit", {"toolkit_present": False}, "needs a CUDA toolkit"),
            ("no_nvcc", {"nvcc_present": False}, "compiles with nvcc"),
            ("unreadable_nvcc", {"nvcc_version": None}, "could not read its CUDA version"),
            ("old_nvcc", {"nvcc_version": (12, 0)}, "too old"),
        ]
    )
    def test_is_deepgemm_loadable(self, _name, env_kwargs, pattern):
        # The single gating source: valid env -> True; each unmet precondition -> False, or the specific
        # `ImportError` when `raise_error=True` (what the loader uses).
        with self._env(**env_kwargs):
            self.assertEqual(dg.is_deepgemm_loadable(), pattern is None)
        if pattern is not None:
            with self._env(**env_kwargs), self.assertRaisesRegex(ImportError, pattern):
                dg.is_deepgemm_loadable(raise_error=True)

    @parameterized.expand(
        [
            ("unloadable_env", {"cuda_available": False}, "requires CUDA"),
            ("kernel_load_fails", {"kernel": None}, "Failed to load"),
            ("missing_symbols", {"kernel": _DeepGemmKernelMissingSymbol}, "missing required symbols"),
        ]
    )
    def test_loader_raises(self, _name, env_kwargs, pattern):
        # The loader delegates gating to `is_deepgemm_loadable(raise_error=True)`, then resolves symbols;
        # confirm each failure surfaces through `load_deepgemm_kernel`.
        with self._env(**env_kwargs), self.assertRaisesRegex(ImportError, pattern):
            dg.load_deepgemm_kernel()

    def test_loader_is_compile_safe(self):
        # Cold path: the compiled call is first to load, so the opaque loader node runs its full body
        # under compile and must return None, never the bundle (`Unsupported: torch.* op returned
        # non-Tensor`). Default (inductor) backend; `_env` fakes CUDA+SM90 only for the loader so
        # torch.compile sees the real device — no GPU required.
        with self._env():
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

            out = run(torch.zeros(3, device=torch_device))
        self.assertTrue(torch.equal(out, torch.ones(3, device=torch_device)))

    def test_loader_is_compile_safe_when_warm(self):
        # Warm path (production order: eager warmup, then compile). The loader runs its arch/env checks
        # every call, then hits the cache short-circuit — which must also return None, not the bundle.
        with self._env():
            dg.load_deepgemm_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

            out = run(torch.zeros(3, device=torch_device))
        self.assertTrue(torch.equal(out, torch.ones(3, device=torch_device)))

    def test_is_deepgemm_loadable_is_compile_safe(self):
        # Regression guard: `is_deepgemm_loadable` sits in the compiled `fp8_linear` gate, so it must fold
        # to a constant (via `@assume_constant_result`); tracing its env probe would break the graph.
        torch.compiler.reset()

        @torch.compile(fullgraph=True)
        def run(x):
            return x + 1 if dg.is_deepgemm_loadable() else x - 1

        run(torch.zeros(3, device=torch_device))  # a graph break / traced probe would raise here

    def test_uint8_scale_containers_use_exponents_on_both_architectures(self):
        scales = torch.tensor([[126, 127, 128, 129]], dtype=torch.uint8).repeat(2, 1)
        hopper = dg._coerce_sf_for_kernel(scales, is_sm100=False)
        torch.testing.assert_close(hopper, torch.tensor([[0.5, 1.0, 2.0, 4.0]]).repeat(2, 1))
        blackwell = dg._coerce_sf_for_kernel(scales, is_sm100=True)
        torch.testing.assert_close(blackwell, scales.view(torch.int32))


# ── Capturing fake DeepGEMM bundle ─────────────────────────────────────────────
#
# Every op records its call and returns a rightly-shaped output; the matmul-family ops zero their
# provided output buffer in place (the real kernels write it in place) so downstream reductions stay
# finite. Scale tensors follow the real recipe dtype: UE8M0 (`float8_e8m0fnu`) when `use_ue8m0`, else
# `float32` — so `_coerce_sf_for_kernel`'s dtype-driven branches execute for real.


def _make_bundle(captured):
    def per_token_cast_to_fp8(x, *, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False):
        cols = -(-x.size(-1) // gran_k)
        sf_dtype = torch.float8_e8m0fnu if use_ue8m0 else torch.float32
        sf = torch.ones(x.size(0), cols, dtype=torch.float32, device=x.device).to(sf_dtype)
        captured.setdefault("per_token_cast", []).append(
            {"x": x, "use_ue8m0": use_ue8m0, "gran_k": gran_k, "use_packed_ue8m0": use_packed_ue8m0}
        )
        return x.to(torch.float8_e4m3fn), sf

    def fp8_fp4_matmul(lhs, rhs, out, *, recipe=None):
        captured.setdefault("matmul", []).append({"lhs": lhs, "rhs": rhs, "out": out, "recipe": recipe})
        out.zero_()

    def _grouped(name):
        def op(lhs, rhs, out, grouped_layout, *, recipe=None, use_psum_layout=False):
            captured.setdefault("grouped", []).append(
                {
                    "name": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "out": out,
                    "grouped_layout": grouped_layout,
                    "recipe": recipe,
                    "use_psum_layout": use_psum_layout,
                }
            )
            out.zero_()

        return op

    def transform_sf_into_required_layout(sf, dim_a, dim_b, *, recipe, num_groups):
        captured.setdefault("transform_sf", []).append(
            {"sf": sf, "dim_a": dim_a, "dim_b": dim_b, "recipe": recipe, "num_groups": num_groups}
        )
        return sf

    def transform_weights_for_mega_moe(gate_up_pair, down_pair):
        captured["transform_weights_in"] = (gate_up_pair, down_pair)
        # Preserve the `[E_local, 2*I, *]` leading dims the forward reads back; tag the payloads so we
        # can prove the module parameters were overwritten with exactly these transformed tensors.
        gate_up_w, gate_up_sf = gate_up_pair
        down_w, down_sf = down_pair
        new_gate_up = torch.full_like(gate_up_w, 7)
        new_down = torch.full_like(down_w, 9)
        out = ((new_gate_up, gate_up_sf), (new_down, down_sf))
        captured["transform_weights_out"] = out
        return out

    def get_symm_buffer_for_mega_moe(
        process_group, *, hidden, num_topk, num_experts, num_max_tokens_per_rank, intermediate_hidden
    ):
        captured["symm_buffer_kwargs"] = {
            "process_group": process_group,
            "hidden": hidden,
            "num_topk": num_topk,
            "num_experts": num_experts,
            "num_max_tokens_per_rank": num_max_tokens_per_rank,
            "intermediate_hidden": intermediate_hidden,
        }
        buf = types.SimpleNamespace(
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            x=torch.zeros(num_max_tokens_per_rank, hidden, dtype=torch.float8_e4m3fn, device=torch_device),
            x_sf=torch.zeros(num_max_tokens_per_rank, -(-hidden // 32), dtype=torch.float32, device=torch_device),
            topk_idx=torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.long, device=torch_device),
            topk_weights=torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.float32, device=torch_device),
        )
        captured["symm_buffer"] = buf
        return buf

    def fp8_fp4_mega_moe(y, gate_up_pair, down_pair, symm_buffer, *, activation_clamp=None):
        captured["mega_moe"] = {
            "y": y,
            "gate_up_pair": gate_up_pair,
            "down_pair": down_pair,
            "symm_buffer": symm_buffer,
            "activation_clamp": activation_clamp,
        }
        y.zero_()

    return dg.DeepGEMM(
        fp8_fp4_matmul=fp8_fp4_matmul,
        grouped_fp8_fp4_matmul_nt=_grouped("fp8_fp4_nt"),
        grouped_fp8_fp4_matmul_nn=_grouped("fp8_fp4_nn"),
        grouped_bf16_matmul_nt=_grouped("bf16_nt"),
        grouped_bf16_matmul_nn=_grouped("bf16_nn"),
        per_token_cast_to_fp8=per_token_cast_to_fp8,
        transform_sf_into_required_layout=transform_sf_into_required_layout,
        transform_weights_for_mega_moe=transform_weights_for_mega_moe,
        get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        m_alignment=128,
    )


@require_torch
class DeepGemmForwardTest(unittest.TestCase):
    """Drives the real public forwards with only the loaded-kernel ops mocked."""

    def setUp(self):
        dg._DEEPGEMM = None
        self.addCleanup(setattr, dg, "_DEEPGEMM", None)

    @contextlib.contextmanager
    def _bundle(self, *, is_sm100):
        captured = {}
        bundle = _make_bundle(captured)
        # The forwards read the arch via `is_sm100()`, so fake the device to the requested arch: lets
        # SM100 dispatch/packing run on this SM80 box and drives the Hopper-rejection guards. `[0]` is
        # all `is_sm100()` reads of the capability, and it reaches that only when CUDA is available —
        # so a CPU-only runner needs both faked, or every arch-gated branch below goes untested.
        capability = (10, 0) if is_sm100 else (9, 0)
        with (
            mock.patch.object(dg, "load_deepgemm_kernel", return_value=bundle),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
        ):
            yield captured

    # ── deepgemm_fp8_fp4_linear ────────────────────────────────────────────────

    @require_torch_greater_or_equal("2.7")  # torch.float8_e8m0fnu (UE8M0) landed in 2.7
    def test_linear_fp4_sm100_packs_ue8m0_scales(self):
        # FP4 weights (int8): SM100-only. Scales arrive UE8M0 and must reach the kernel packed into
        # int32 (4 K-bytes -> 1 int32) with the `(1, 1, gran_k=32)` recipe.
        input = torch.randn(4, 128, dtype=torch.bfloat16, device=torch_device)
        weight = torch.randint(-8, 8, (256, 64), dtype=torch.int8, device=torch_device)
        weight_scale = torch.ones(256, 4, dtype=torch.float32, device=torch_device).to(
            torch.float8_e8m0fnu
        )  # (N, K/32)
        with self._bundle(is_sm100=True) as captured:
            deepgemm_fp8_fp4_linear(input, weight, weight_scale)

        cast = captured["per_token_cast"][0]
        self.assertTrue(cast["use_ue8m0"] and cast["use_packed_ue8m0"])
        self.assertEqual(cast["gran_k"], 32)

        (matmul,) = captured["matmul"]
        self.assertEqual(matmul["recipe"], (1, 1, 32))
        (_, act_sf), (_, w_sf) = matmul["lhs"], matmul["rhs"]
        # UE8M0 bytes packed 4->1 into int32 for the kernel's `(INT, 1, gran_k)` branch.
        self.assertEqual(act_sf.dtype, torch.int32)
        self.assertEqual(w_sf.dtype, torch.int32)
        self.assertEqual(w_sf.shape, (256, 1))

    def test_linear_rejects_float32_scales_on_sm100(self):
        # Regression for #47030: rounding a checkpoint's float32 scales up to UE8M0 without requantizing
        # silently corrupts the output on SM100. `fp8_linear` skips DeepGEMM up front for this combo, but
        # this guards the backstop inside the forward itself (which also protects the experts / megamoe
        # paths, that have no such gate): the forward must reject float32 scales before any kernel op runs.
        input = torch.randn(4, 128, dtype=torch.bfloat16, device=torch_device)
        weight = torch.randn(256, 128, device=torch_device).to(torch.float8_e4m3fn)
        weight_scale = torch.ones(2, 1, dtype=torch.float32, device=torch_device)
        with self._bundle(is_sm100=True) as captured:
            with self.assertRaisesRegex(NotImplementedError, "float32 scale-factor path"):
                deepgemm_fp8_fp4_linear(input, weight, weight_scale, block_size=(128, 128))
        self.assertEqual(captured, {})  # raised before the kernel ran

    # ── deepgemm_bf16_experts_forward ──────────────────────────────────────────

    def _bf16_hidden(self, experts, *, num_tokens=3, top_k=2):
        # hidden = the projection's input dim: (E, out, H) non-transposed, (E, H, out) transposed.
        proj = experts.gate_up_proj if experts.has_gate else experts.up_proj
        hidden = proj.shape[1] if experts.is_transposed else proj.shape[-1]
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3]], device=torch_device)[:num_tokens, :top_k]
        top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device=torch_device)
        return hidden_states, top_k_index, top_k_weights

    def test_bf16_experts_kernel_inputs_sm90(self):
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        up, down = captured["grouped"]
        # Non-transposed -> NT kernel for both projections.
        self.assertEqual(up["name"], "bf16_nt")
        self.assertEqual(down["name"], "bf16_nt")
        # Weights are forwarded as-is (gate_up for the up proj, down for the down proj).
        self.assertIs(up["rhs"], experts.gate_up_proj)
        self.assertIs(down["rhs"], experts.down_proj)
        # SM90 grouped layout: int32 per-row expert id, `-1` in padding rows the kernel skips.
        self.assertEqual(up["grouped_layout"].dtype, torch.int32)
        self.assertEqual(up["grouped_layout"].dim(), 1)
        self.assertFalse(up["use_psum_layout"])
        self.assertTrue((up["grouped_layout"] == -1).any())  # alignment padding present
        self.assertEqual(set(up["grouped_layout"].unique().tolist()), {-1, 0, 1, 2, 3})
        # Up-proj output is the padded (rows, 2I) buffer; hidden activation padded to the same rows.
        rows = up["out"].shape[0]
        self.assertEqual(up["out"].shape, (rows, 32))
        self.assertEqual(up["lhs"].shape, (rows, 8))

    def test_bf16_experts_sm100_uses_psum_cumsum_layout(self):
        # SM100 grouped layout is the cumsum of per-expert aligned counts (not a per-row id vector).
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=True) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        up = captured["grouped"][0]
        self.assertTrue(up["use_psum_layout"])
        layout = up["grouped_layout"]
        self.assertEqual(layout.dtype, torch.int32)
        self.assertEqual(layout.numel(), experts.num_experts)  # one boundary per expert
        # Monotonic non-decreasing cumsum, each entry a multiple of the M-alignment (128).
        self.assertTrue(torch.all(layout[1:] >= layout[:-1]))
        self.assertTrue(torch.all(layout % 128 == 0))

    def test_bf16_experts_marshals_bias_by_expert(self):
        # Bias is gathered per routed pair (up_bias[expert_ids]) and scattered into the padded rows.
        # With the fake matmul zeroing its output, the (non-gated) down input is `act_fn(padded bias)`,
        # so its column-sum equals `act_fn(per-pair gathered up_bias)` summed over all pairs (padding rows
        # contribute act_fn(0)=0; order-invariant).
        experts = make_experts(num_experts=4, hidden=8, inter=16, has_gate=False, has_bias=True)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        down_act = captured["grouped"][1]["lhs"]
        expected = experts.act_fn(experts.up_proj_bias[top_k_index.reshape(-1)]).sum(dim=0)
        self.assertTrue(torch.allclose(down_act.sum(dim=0).float(), expected.float(), atol=1e-2))

    def test_bf16_experts_rejects_non_bf16_hidden_states(self):
        experts = make_experts()
        hidden_states = torch.randn(3, 8, dtype=torch.float16, device=torch_device)
        top_k_index = torch.zeros(3, 2, dtype=torch.long, device=torch_device)
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device=torch_device)
        with self.assertRaisesRegex(ValueError, "requires bfloat16 hidden states"):
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

    # ── deepgemm_fp8_fp4_experts_forward ───────────────────────────────────────

    def _fp8_hidden(self, num_tokens=3, hidden=8, top_k=2):
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3]], device=torch_device)[:num_tokens, :top_k]
        top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device=torch_device)
        return hidden_states, top_k_index, top_k_weights

    # ── deepgemm_fp8_fp4_megamoe_experts_forward ───────────────────────────────

    def _make_megamoe(self, *, num_experts=4, hidden_dim=64, inter=32, swiglu_limit=None):
        gate_up = torch.randint(
            -8, 8, (num_experts, 2 * inter, hidden_dim // 2), dtype=torch.int8, device=torch_device
        )
        down = torch.randint(-8, 8, (num_experts, hidden_dim, inter // 2), dtype=torch.int8, device=torch_device)
        return types.SimpleNamespace(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=inter,
            gate_up_proj=torch.nn.Parameter(gate_up, requires_grad=False),
            down_proj=torch.nn.Parameter(down, requires_grad=False),
            gate_up_proj_scale_inv=torch.nn.Parameter(
                torch.ones(num_experts, 2 * inter, hidden_dim // 32, device=torch_device).to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
            down_proj_scale_inv=torch.nn.Parameter(
                torch.ones(num_experts, hidden_dim, inter // 32, device=torch_device).to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
            config=types.SimpleNamespace(swiglu_limit=swiglu_limit),
            has_post_expert_norm=False,
        )

    @require_torch_greater_or_equal("2.7")  # torch.float8_e8m0fnu (UE8M0) landed in 2.7
    def test_megamoe_requires_fp4_weights_and_process_group(self):
        hidden_states = torch.randn(3, 64, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.randint(0, 4, (3, 2), dtype=torch.long, device=torch_device)
        top_k_weights = torch.rand(3, 2, dtype=torch.float32, device=torch_device)

        # Non-FP4 (non-int8) expert weights are rejected — Mega MoE is FP4-packed only.
        module = self._make_megamoe()
        module.gate_up_proj = torch.nn.Parameter(module.gate_up_proj.data.to(torch.bfloat16), requires_grad=False)
        with self._bundle(is_sm100=True):
            with self.assertRaisesRegex(NotImplementedError, "FP4-packed expert weights"):
                deepgemm_fp8_fp4_megamoe_experts_forward(
                    module,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    process_group=types.SimpleNamespace(size=lambda: 1),
                )

        # Missing process_group (EP group) is rejected — required for the symm-buffer rendezvous.
        module2 = self._make_megamoe()
        with self._bundle(is_sm100=True):
            with self.assertRaisesRegex(ValueError, "requires a .process_group."):
                deepgemm_fp8_fp4_megamoe_experts_forward(module2, hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def _megamoe_flag_set(module):
        return getattr(module, "_megamoe_transformed", False)

    def test_every_arm_refuses_hidden_states_that_are_not_bfloat16(self):
        """Each arm builds its intermediates and its output in bfloat16, so anything else changes
        the dtype the caller gets back. The check belongs to the guard, not to one forward: it
        used to be copied into two of the three and missing from Mega MoE, which allocates its
        output bfloat16 regardless."""
        import transformers.integrations.deepgemm as dg

        class _Experts(torch.nn.Module):
            activation_scheme = "dynamic"

        arms = (
            dg.deepgemm_bf16_experts_forward,
            dg.deepgemm_fp8_fp4_experts_forward,
            dg.deepgemm_fp8_fp4_megamoe_experts_forward,
        )
        for forward in arms:
            with self.subTest(forward=forward.__name__), self.assertRaises(ValueError) as caught:
                forward(_Experts(), torch.zeros(2, 4, dtype=torch.float16), None, None)
            self.assertIn("bfloat16", str(caught.exception))

    def test_megamoe_refuses_a_post_expert_norm(self):
        """Mega MoE fuses the routing-weighted reduce, so a per-expert output norm has nowhere to go."""
        import transformers.integrations.deepgemm as dg

        module = torch.nn.Module()
        module.has_post_expert_norm = True
        with self.assertRaises(NotImplementedError):
            dg._assert_no_post_expert_norm(module)

    def test_sm100_guard_takes_non_gated_experts(self):
        # The arch guard reads the weight dtype from `down_proj`, which gated and non-gated experts both hold
        forward = dg.deepgemm_experts_guards(sm100=True)(lambda module, hidden_states: "ran")
        module = mock.Mock(
            spec=["down_proj", "down_proj_scale_inv", "has_post_expert_norm"],
            down_proj=torch.zeros(1, 1, 1, dtype=torch.float8_e4m3fn),
            down_proj_scale_inv=torch.ones(1, 1, 1),
            has_post_expert_norm=False,
        )
        hidden_states = torch.zeros(1, 1, dtype=torch.bfloat16)
        with mock.patch.object(dg, "is_sm100", return_value=False):
            self.assertEqual(forward(module, hidden_states), "ran")


@require_torch
class DeepGemmFineGrainedDispatchTest(unittest.TestCase):
    """`prefers_deepgemm_linear` carries two independent gates: a correctness one (a pre-swizzled scale is not
    readable as row-major) and an SM100 perf one. Block-FP8 scales have no swizzled layout, so the first never
    fires for the shape the second catches."""

    def _routed_to(self, *, sm100, scale_ndim, trains=False, fp4=False, input_dtype=torch.bfloat16):
        """Which backend a block-FP8 (or FP4) linear reaches, given arch, scale layout, input dtype and whether
        the call needs a gradient."""
        kernel, _ = fake_finegrained_kernel()
        w = torch.randint(-8, 8, (32, 32), dtype=torch.int8) if fp4 else torch.randn(32, 64).to(torch.float8_e4m3fn)
        s = torch.randn(*([1, 1, 1, 1, 1][:scale_ndim] if scale_ndim > 2 else [1, 1]))
        with (
            patch_finegrained_kernel(kernel),
            mock.patch.dict(os.environ, {"TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR": "0"}),
            mock.patch.object(dg, "is_deepgemm_loadable", return_value=True),
            mock.patch.object(dg, "is_sm100", return_value=sm100),
            mock.patch.object(fg, "deepgemm_fp8_fp4_linear") as deepgemm_linear,
        ):
            x = torch.randn(4, 64, dtype=input_dtype, requires_grad=trains)
            fg.finegrained_linear(x, w, s, block_size=[128, 128])
        return "deepgemm" if deepgemm_linear.called else "triton"

    @parameterized.expand(
        [
            ("sm100_never_prefers_deepgemm", {"sm100": True}, "triton"),
            ("pre_sm100_uses_deepgemm", {}, "deepgemm"),
            # DeepGEMM has no backward pass
            ("a_call_needing_a_gradient", {"trains": True}, "triton"),
            # DeepGEMM's FP4 kernel needs SM100 and its linear takes half-precision inputs: both stay on Triton
            # up front instead of attempting DeepGEMM and falling back on every call
            ("fp4_before_sm100", {"fp4": True}, "triton"),
            ("full_precision_input", {"input_dtype": torch.float32}, "triton"),
            # the correctness gate holds on any arch, not just where the perf gate does
            ("swizzled_scales", {"scale_ndim": 5}, "triton"),
        ]
    )
    def test_routing(self, _, kwargs, backend):
        self.assertEqual(self._routed_to(**{"sm100": False, "scale_ndim": 2, **kwargs}), backend)
