# TensorFold / spark-ds41f kernel-parity investigation

Status: active diagnostic; not a qualified release and not a prompt workaround.

Diagnostic patch: `vllm/0001-test-dsv41-add-bounded-kernel-parity-traces.patch`,
exported from AI-assisted commit `644b00ca0` on the exact imported patch head
`2756478b60b90700b9918b457aeb38657da637b5`. It is diagnostic tooling, not the
eventual causal fix.

Head arithmetic A/B: `vllm/0002-test-dsv41-compare-served-and-torch-head-logits.patch`,
exported from AI-assisted commit `2e1965c99`. It computes a torch FP32 projection
from the exact unpacked BF16 head beside the served TileLang projection without
using the torch result for sampling.

Sequence-parallel trace alignment:
`vllm/0003-test-dsv41-trace-sequence-parallel-target-rows.patch`, exported from
AI-assisted commit `c8c77f132`. It maps the absolute target position onto the
owning rank's local prefill slice; without this, full-prefill tracing could
silently omit the target even while replicated decode tracing worked.

Independent head control:
`vllm/0004-test-dsv41-add-independent-fp64-head-control.patch`, exported from
AI-assisted commit `8c81c6fe2` after Claude Code challenged the shared gather
path. It computes only token rows 16 and 469 on their owning shard in CPU FP64,
before any distributed gather, from the exact same hidden row and unpacked BF16
weight. This is diagnostic evidence, not a sampling override.

Startup/request separation:
`vllm/0005-test-dsv41-gate-traces-until-request-replay.patch`, exported from
AI-assisted commit `823aa58ad`. It keeps model profiling and warmup rows out of
the evidence by requiring a validated owner-only sentinel that the guarded
runner creates only after readiness and launch-identity checks.

Sparse-attention boundary trace:
`vllm/0006-test-dsv41-trace-sparse-attention-boundary.patch`, exported from
AI-assisted commit `e89d37d3c`. It captures target-row index-query bytes and
head weights, layer-20 candidate blocks, per-layer selected sparse keys,
attention metadata and pre/post-projection attention outputs. It is bounded
diagnostic instrumentation and does not affect the sampled result.

Sparse-attention operand trace:
`vllm/0007-test-dsv41-trace-sparse-attention-operands.patch`, exported from
AI-assisted commit `17021eca2`. It records only the locked row's layer-20/21
attention Q/KV operands, selected physical slots and selected raw cache records
to distinguish cache-content drift from decode/extend sparse-MLA schedule drift.
It is diagnostic-only and does not alter attention or sampling.

Exact-mode CED bypass:
`vllm/0008-feat-dsv41-add-exact-mode-CED-bypass.patch`, exported from
AI-assisted commit `c32911467`. It adds an opt-in environment switch that
disables context-encoder/decoder row compaction for a causal A/B. Default
behavior is unchanged. This is an experiment, not yet a qualified fix.

Exact retained-state warm-up:
`vllm/0009-feat-dsv41-warm-CED-replay-for-exact-retained-states.patch`, exported
from AI-assisted commit `d7b1b5a17`. Its opt-in mode executes 127 warm-up rows
before the normal 128-row decoder prefill, then returns to the unchanged
128-row decode window. The locked case remained 5/6, disproving the initial
one-window exactness assumption. Default behavior is unchanged.

Decoder-depth propagation fix:
`vllm/0010-fix-dsv41-propagate-CED-warmup-through-decoder-depth.patch`, exported
from AI-assisted commit `0a18f5c87`. It accounts for the 127-row SWA receptive
field growth across all 20 replayed decoder layers, using a 2,541-row replay
span before returning to the unchanged 128-row decode path. Claude Code
independently challenged the one-window derivation and agreed with this
dependency frontier; only local traces and end-to-end tests count as evidence.
Default behavior is unchanged; qualification is pending.

## Locked acceptance case

- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash` revision
  `dba1be0a40aa45a94ad051997016db3960a90277`, existing weights only.
- Input: the original six-finding screenshot request and image bytes from the
  preserved qualification receipt. The request, chat template, token IDs,
  sampling, parser and scoring oracle must remain unchanged.
- Sampling: target-only greedy, no speculative decoding, no prompt policy,
  no response-format rewrite and no post-processing repair.
- Required result: the engine must emit all six expected findings without
  steering. The change must also pass the locked text, vision, tool, streaming,
  concurrency and restart gates.

## Diagnostic matrix

Run each cell twice and require bitwise repeatability before comparing engines.

| Engine | Full-prefix prefill | True token-by-token decode |
|---|---:|---:|
| TensorFold served source `09c2b6cd504ba7909524cd5e6b2acca533d606f1` | required | required |
| spark-ds41f user fork at this branch | required | required |

At the common 88-generated-token prefix, capture only numerical state needed
for diagnosis: per-layer attention input, per-layer MoE input, router choices
and weights, final normalized head input, and logits for the competing tokens
plus the local top set. Do not persist prompt or image bodies, model weights,
credentials, or unrelated requests.

Treat token IDs and hidden-state traces as sensitive derived data even though
the harness omits prompt/image/response bodies; keep all trace files owner-only.

The first layer with a material error-growth jump is a localization clue, not
proof by itself. A causal fix must remove that jump in the intended kernel/path,
change the locked end-to-end result to 6/6, and survive the broader regression.

## Current evidence

- Both engines share the first 88 generated tokens. At token 89 TensorFold
  selects token 469 (` or`) while spark-ds41f selects token 16 (`.`).
- The spark-ds41f target logit margin is close: token 16 exceeds token 469 by
  about 0.019 in natural decode. Full-prefix replay increases that margin to
  about 0.168, so prefill/decode parity is itself under test.
- The target row remains byte-identical between natural decode and a
  teacher-forced full prefill through layer 21's attention input. Layer-20
  candidate blocks, candidate lengths, index query bytes/scales/weights,
  selected sparse positions and sliding-window metadata are also byte-identical.
  The first difference appears in layer 21's sparse-MLA output on every rank.
  Its selected raw cache differs only in the oldest 39 of 128 rows; the newest
  89 are exact. Position tracing tied this directly to the CED replay floor.
  The head implementations agree within roughly 3.4e-5 logit, excluding the
  head and candidate chooser.
- The causal replay ladder is now complete. Ordinary 128-row replay and a
  255-row one-window warm-up both score 5/6. Full decoder prefill and the
  2,541-row decoder-depth warm-up both score 6/6 and emit the same complete
  1,336-token sequence. The latter returns to ordinary 128-row decode after
  prefill and is the implemented engine fix for this locked case.
- Replaying TensorFold's full reasoning through spark-ds41f produces 6/6;
  replaying spark-ds41f reasoning produces 5/6. Forcing only the first divergent
  token does not fix the outcome.
- Torch vision vectors and prompt token IDs were byte-identical offline. In-engine
  merged embeddings, positions and masks still require direct confirmation.
- The known routed-MoE BF16-boundary correction changes later tokens but did not
  repair this case. It remains correct independently, but is not credited as
  the present accuracy fix.

## Promotion rule

No prompt addition, system-message policy, response-format omission, decoding
steering or output repair can satisfy this investigation. No production switch
occurs from this experiment alone. A candidate is promotable only after the
locked case reaches 6/6 for the causal engine change and the full release gates
pass.

## Provenance

- Upstream serving base: Local Inference Lab vLLM commit
  `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` and the repository's recorded
  Christopher Owen patch stack.
- TensorFold comparison: upstream/adapted TensorFold, served source commit
  `09c2b6cd504ba7909524cd5e6b2acca533d606f1`.
- Model/reference behavior: DeepSeek checkpoint/revision above and its existing
  reference inference sources.
- Diagnostic harness and any eventual fix: newly implemented in this branch,
  AI-assisted by Codex. Claude Code supplied an independent diagnostic review;
  its opinion is not treated as test evidence.
- No upstream pull request is proposed. Before any publication, ownership,
  licensing, attribution, imported layers and the exact submitted diff must be
  reviewed explicitly.
