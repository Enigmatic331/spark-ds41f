# TensorFold-compatible DeepSeek vision arithmetic

## Scope

This experiment adds one opt-in arithmetic mode to the generic DeepSeek-V4
vision tower. The default constructor, CPU-built RoPE and attention path are
unchanged. When selected, the mode constructs the existing DeepSeek RoPE
expression directly on the input tensor's device, preserves the checkpoint's
BF16 QKV projection and BF16 RoPE rounding, widens those rounded values to
FP32, forces PyTorch's memory-efficient SDPA backend, and rounds the attention
output once to the input dtype before the output projection. It also preserves
the BF16 intermediate between the vision MLP's FP32 SiLU and FP32 multiply,
matching the observed TensorFold lane instead of leaving that boundary to a
packed activation kernel/compiler path.

The four-patch experiment applies after the 641-patch vLLM series recorded by
`experiments/2026-10-26-spark-ds41f-candidate/1107/upstreams.lock.json` over
[Local Inference Lab vLLM revision
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`](https://github.com/local-inference-lab/vllm/commit/19f2c20ed4d64ae22bd009ceacadd85e2a81e354).
The imported Christopher stack is preserved at
[`59f523c543cb61e5a2d0df616cb4efa459677b35`](https://github.com/christopherowen/spark-ds41f/commit/59f523c543cb61e5a2d0df616cb4efa459677b35)
inside the audited checkout
[`998b21a0b9ecdc5c61698be89e7318bef417116e`](https://github.com/christopherowen/spark-ds41f/commit/998b21a0b9ecdc5c61698be89e7318bef417116e).

## Provenance

- The generic tower is [Local Inference Lab vLLM](https://github.com/local-inference-lab/vllm)
  Apache-2.0 code ([pinned license](https://github.com/local-inference-lab/vllm/blob/19f2c20ed4d64ae22bd009ceacadd85e2a81e354/LICENSE)),
  itself marked as a port of DeepSeek's MIT-licensed reference vision
  implementation. The checkpoint/reference pin is
  [`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277),
  with its [MIT license at that revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/LICENSE).
- The compatibility arithmetic follows the served-lane recording documented
  by [`bertholomus/TensorFold@6062066f65a05a7799c09b9aadde0517ded82105`](https://github.com/bertholomus/TensorFold/commit/6062066f65a05a7799c09b9aadde0517ded82105).
  TensorFold is [Apache-2.0](https://github.com/bertholomus/TensorFold/blob/6062066f65a05a7799c09b9aadde0517ded82105/LICENSE)
  and attributes its extracted PyTorch memory-efficient attention kernel to
  PyTorch under the [BSD-3-Clause license](https://github.com/pytorch/pytorch/blob/main/LICENSE).
  This experiment invokes installed PyTorch; it does not copy that kernel.
  No TensorFold source or kernel binary is copied by this patch stack.
- The device-explicit RoPE change is a local adaptation of the DeepSeek
  expression already present in vLLM. It reproduces the official inference
  process's `torch.set_default_device("cuda")` arithmetic without changing the
  process-wide default device. The diagnosis and adaptation were AI-assisted
  and reviewed against local raw-stage probes; no additional DeepSeek or
  TensorFold source was copied.
- The explicit vision activation is a local, AI-assisted adaptation of the
  TensorFold arithmetic recorded by paired GB10 stage probes: FP32 SiLU, BF16
  rounding, FP32 multiply by the BF16 up branch, then BF16 output rounding.
  Claude reviewed the diagnosis; the raw paired tensor hashes, rather than the
  model review, are the authoritative evidence. No TensorFold source or binary
  was copied.
- `_encode_images_torch` is locally adapted and refactored from upstream
  vLLM's existing non-packed DeepSeek image loop. The environment selector and
  its Torch/aligner factory adapters are newly implemented local AI-assisted
  integration. The selector tests added by patch `0004` are also newly
  implemented locally; they change no serving source.
- The combined code is a local adaptation over those sources. It is not
  represented as original TensorFold work or as ready for an upstream pull
  request.

## Root-cause evidence

The two 65x103-patch fixtures already matched TensorFold at the patch embedding
boundary. The former vLLM path then built RoPE on CPU and copied it to CUDA, so
its first mismatch was `cos`. Running the same existing DeepSeek expression
with all tensor sources created on CUDA matched TensorFold exactly:

- `cos` SHA256:
  `2de6257af4ec7f2e0279c71c3d0162db824b7ed7cbae9a6a8f0e25b306a34925`
- `sin` SHA256:
  `210e00fc3e2ae6b2f69c08e8b8c3b3e6ed10800b1cda7d088e7aabaf3c807a8e`

These are raw FP32 tensor-byte hashes, not semantic approximations. A corrected
token-major-layout trace then showed that QKV, rotated/widened Q/K/V, FP32
attention, the BF16 attention result, output projection and residual all
already matched TensorFold byte-for-byte. The first remaining difference was
the MLP activation. After making its BF16 intermediate explicit, both
65x103-patch Etherscan fixtures matched TensorFold at all 102 recorded stages,
including final rows and 794-token aligned spans, with zero differing BF16
values. This proves encoder parity for those fixtures; it does not replace the
strict end-to-end served-output gate.

## Quality implications

This mode intentionally changes vision embeddings and must remain opt-in until
every later raw stage and multimodal regression test passes. It may also change
vision latency and memory use because RoPE is constructed on CUDA and attention
inputs and output are FP32. The explicit activation also materializes FP32 and
BF16 intermediates and disables compiler fusion for that small module.
Text-only execution and default generic vision execution are unaffected.

## Activation boundary and patch order

Apply the files in `vllm/series` order to the pinned, fully materialized vLLM
tree. Patch `0001` adds the opt-in RoPE/attention arithmetic to the generic
tower without changing its default. Patch `0002` records the local V4.1
selector used by the audited deployment:
`VLLM_DSV41_VISION_BACKEND=torch` selects the compatibility tower and passes
`tensorfold_compat_attention=True`; the unset/default value continues to select
TileLang. Patch `0003` uses that same opt-in flag to select the explicit BF16
vision activation. The Torch lane deliberately rejects tensor-parallel vision
and therefore must be launched with `--mm-encoder-tp-mode data`.

Patch `0002` includes the pre-existing local Torch selector/factories and the
per-image runner adapted from upstream vLLM's non-packed DeepSeek loop; only
its `tensorfold_compat_attention=True` constructor argument was added during
this vision-arithmetic diagnosis. The selector/factories are local AI-assisted
integration, while the runner retains its upstream lineage. Neither is
TensorFold authorship. Patch `0004` adds tests for both selector branches,
invalid values, data-parallel rejection, the compatibility constructor flag,
and per-image slicing without changing production code.

The audited before/after selector hashes are:

- imported 1107 selector: `ecf43fa91334ad768ee5c15386f939532cbfd734c8182a1494c8eb472fe640d0`
- local Torch compatibility selector: `f4c899fde8117caaab1b18fc506389e0e504ce6cdb6c7ee19cad3100ba08c4d0`

The audited generic tower hashes are:

- imported generic tower: `b7218cc112fb6377f5bf7628e2dad22148e26f014c554b9b45d8398a5f7884ec`
- RoPE/attention-only tower: `a99b4dd1a5b6b89ad6b492cc26f4e91c37d0ff285023c41680fe42dd521d0998`
- final activation-compatible tower: `54f6fe4b4ae0d8a2876f101d725e5ab6e4ea522c813fde4d6f5ea718232279e7`

## Prepared validation

The first 24 focused cases cover arithmetic opt-in/default behavior: the
default generic tower omits a RoPE device, retains the ordinary attention
wrapper and retains vLLM's ordinary `SiluAndMul`; compatibility mode passes the
patch embedding's device to all three RoPE tensor constructors, uses FP32
Q/K/V, enters only `SDPBackend.EFFICIENT_ATTENTION`, restores BF16 before
`wo`, and selects the explicit staged-BF16 activation. They also check
dtype/shape rejection and a BF16 input that distinguishes the staged result
from arithmetic that retains SiLU in FP32 through the multiply. Those 24 cases
do **not** by themselves exercise `VLLM_DSV41_VISION_BACKEND` or the per-image
runner.

Patch `0004` adds six focused CPU cases for exactly that integration boundary:
default TileLang selection, explicit Torch selection, invalid value rejection,
both data-parallel factories (including the TensorFold compatibility flag),
and exact per-image patch slicing. The complete 30-case file passed on DGX02
in the isolated native runtime on 2026-11-13; CUDA execution was not required.
Applying patches `0001` through `0003` reproduces the deployed source and its
original focused-test hash:

- generic tower: `54f6fe4b4ae0d8a2876f101d725e5ab6e4ea522c813fde4d6f5ea718232279e7`
- selector: `f4c899fde8117caaab1b18fc506389e0e504ce6cdb6c7ee19cad3100ba08c4d0`
- focused tests: `532d646b1bd440f8ece36c3c25dcaacd369d36b90a343eafd96ddbe0a8174461`

Applying test-only patch `0004` changes only the focused-test file to SHA256
`778abe33967b0d0e26ea204e42e780bc1ed13538f2b5564fa2638aa0de273bd0`.
The retained DGX02 receipt is
`$HOME/spark-ds41f-native-20261113/diagnostics/vision-selector-unit-K4dOMfcC/`;
it records 30/30 passing cases and the same deployed generic-tower and selector
hashes above. The original 24/24 claim and the additional six selector/runner
cases are therefore distinct rather than conflated.

The guarded GB10 raw-stage diagnostic subsequently proved byte identity at all
102 recorded stages and at the final rows/spans for both Etherscan fixtures.
Post-fix normal-static serving passed the **original archived Etherscan scorer**
7/7. E4 then loaded the exact TensorFold spans on all four ranks and reproduced
the same candidate trajectory twice; both also passed the original scorer, for
9/9 retained normal-plus-E4 passes. E5 reversed the two spans, proved exact
post-scatter placement across all 1,588 placeholder rows on every rank, and
changed both the reasoning and visible answer in the expected semantic
direction. E5 is a causal positive control, not another scorer pass.

The later, locally added same-region wording supplement remains 0/9 on those
candidate normal/E4 outputs, while current TensorFold controls are 4/4. The
candidate identifies the correct header placeholder as "missing" rather than
using the supplement's clipped/narrowed wording. This is a reproducible
downstream output/runtime wording difference and means full output parity is
**not** claimed, but E4/E5 show that it is not a broken encoder, image order,
placeholder scatter, or unused visual span. The complete post-fix archived
debug/concurrency campaign has not yet run; the isolated results must not be
presented as that full campaign or as promotion qualification.

This source bundle does not itself claim production qualification. Separate
site evidence exists for the DSpark-static lane, concurrency, long-text-plus-
image and restart staging checks, but the complete original archived campaign,
real authenticated clients, sustained soak and final same-port canaries remain
release concerns outside this patch stack. Source parity is not service
promotion.

## Upstream status

Local experiment only. Before any upstream proposal, verify duplicate work,
ownership and attribution, review the exact diff, run the required vLLM tests
and model evaluations, and have a human submitter understand every changed line.
