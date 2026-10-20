<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fine-grained

The block-scaled formats are low-precision
weights with one scale per block or group of values. They share one integration and one kernel bundle,
[finegrained-kernels](https://huggingface.co/kernels-community/finegrained-kernels), and each has its own config that
quantizes a whole model. [`FineGrainedConfig`] mixes them in one model. This page covers what they share; the format
pages cover each format's layouts and activation formats.

## Requirements

```bash
pip install --upgrade accelerate "kernels>=0.17.2,<0.18"
```

- Quantized compute needs a CUDA GPU with compute capability 8.9 or newer (Ada, Hopper, Blackwell) or an Intel XPU.
- Without a GPU, or below the required compute capability, a pre-quantized checkpoint is dequantized to bf16 as it
  loads. Quantizing a full-precision checkpoint on the fly raises an error without a GPU, and below the required
  compute capability loads the model unquantized.
- The kernels are loaded from the Hub on the first forward. A missing or incompatible `kernels` package is not caught
  at load time: the first forward raises an `ImportError`.

## Load, quantize and save

A pre-quantized checkpoint carries its quantization config, so it loads with no config. Pass a format config to
[`~PreTrainedModel.from_pretrained`] to quantize a full-precision checkpoint as it loads, and save it with
[`~PreTrainedModel.save_pretrained`]; the saved model reloads with no config.

A config passed with a pre-quantized checkpoint does not requantize it. Its `activation_format` picks the kernels the
weights run on, and its loading options apply:

- `dequantize=True` loads the weights dequantized to full precision, for example to fine-tune them.
- `modules_to_not_convert` replaces the checkpoint's list of modules kept in full precision; left unset, the
  checkpoint's list stays.

The format options (block size, scale format, activation scheme) are the checkpoint's own. A passed value that
differs from them is ignored with a warning.

## Mix formats in one model

[`FineGrainedConfig`]'s `dtype_plan` maps module name patterns to the format config each matching module is
quantized in. A typical plan holds the MoE experts, which carry most of the weights, in a 4-bit format and every other
linear in 8 bits.

- A pattern names the modules it matches with `re.search`. A targeted entry wins over the `".*"` catch-all, whatever
  order the plan is written in, since a plan saved to `config.json` loses its order. Two targeted entries matching one
  module raise an error.
- A module no entry matches, or that `modules_to_not_convert` names, stays in full precision.
- The loading options (`dequantize`, `modules_to_not_convert`) belong to the plan, not to its entries.

## Tensor and expert parallelism

Pass a [`DistributedConfig`] to [`~PreTrainedModel.from_pretrained`] to shard a quantized model; each
scale is sharded with its weight. Quantized experts use [masking and all-reduce](../expert_parallelism) expert
parallelism instead of token dispatch, so `ep_size` must equal `tp_size`.

## Kernel autotuning

The kernels tune each new input shape and cache the chosen configuration. For correctness tests or quick experiments,
set `FINEGRAINED_AUTOTUNE_CONFIG=fast` before the kernels load to shrink the search. Leave it unset when measuring
performance. See the [kernel autotuning documentation](https://huggingface.co/kernels-community/finegrained-kernels#autotuning).
