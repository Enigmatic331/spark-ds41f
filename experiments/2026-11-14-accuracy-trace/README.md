# Accuracy trace on the native four-Spark 1107 client trial

The current native engine has a reproducible screenshot omission. The usual renderer improves its frequency on the tested image, but does not eliminate it. No figure below estimates general model accuracy. November 14, Malaysia time.

## Visible accuracy on paired requests

All screenshot calls used the frozen October 18 debug image, user message and scorer, with normal low reasoning, temperature 0.6 and top-p 0.95. Each of the 14 requests returned HTTP 200, the expected model, zero cached input tokens, a normal stop and one SSE terminator. Only the seed and the presence of the ordinary text format varied; each request had a fresh cache salt. The seeds are paired request settings. Serving and batch state can also affect outputs.

| Seed | Usual request, format omitted | Explicit plain text |
|---|---|---|
| 0 | 6/6 | 5/6 |
| 1 | 6/6 | 6/6 |
| 2 | 6/6 | 5/6 |
| 3 | 5/6 | 5/6 |
| 4 | 6/6 | 5/6 |
| 5 | 6/6 | 5/6 |
| 6 | 6/6 | 6/6 |

The only failed check was the clipped Value column. A separate seed-3 replay of the usual request produced the exact same visible-answer hash. The output identified the To column, broken icons and horizontal scrollbar but omitted the explicit Value-column annotation. The identical historical seed-0 TensorFold request passed. No TensorFold sweep at seeds 1–6 is available.

On three additional existing TensorFold image requests, both engines correctly read the invoice number and amount, identified the cat and its direction of gaze, and matched all three colors to the correct stack heights. Original prompts, image bytes, generation settings and per-request limits were replayed. The historical TensorFold used the same checkpoint revision. On six checkable prompts from the November 11 text validation both engines returned the requested prime numbers, TCP explanation, string reversal, five rivers, explanation of seasons and translations. Both answers to the string-reversal task contained a correct complete function followed by a clipped optional third example under the 160-token cap. Scoring used the user-visible answers, with six task rubrics recorded before the new calls. The other two original prompts were subjective and excluded before generation.

A new paired panel used 16 small synthetic dashboards from one template with shuffled rows and 4 task types: identifying an active issue, correctly reporting a healthy board, copying a specified current code despite an unrelated note, and exactly transcribing a note. Both Christopher request formats scored 16/16 at temperature zero and produced the same visible answer on every item. This is a within-engine control on four task families, not 16 independent populations or a TensorFold comparison.

## Request-format correction

The site middleware's `_should_stage_policy` returns false for every non-null `response_format`. The explicit OpenAI spelling `{"type":"text"}` consequently removes the generic visual-evidence instruction on otherwise eligible image turns. The debug request then has 1,060 input tokens versus 1,136 on the usual path. This difference is introduced before model execution.

`0001-ground-explicit-text-like-default.patch` applies to the site module and tests at fork commit `be65a39c`. The patch treats omitted and explicit plain text identically. The separate `structured_outputs` guard and the exclusions for actual constrained output formats are retained. Nine unit tests passed. On the exact fixture the patched selector produced identical messages in both plain-text forms; the live tokenizer returned the previously frozen 1,136 IDs, SHA-256 `4f1a1d1cdc313b743b5c3279351735605d518f2a047cd104f88e73ce2bdebde4`. Replaying those patched messages on the running model with explicit text and seed 0 passed 6/6. This was an input replay. The active client-trial processes have not loaded the patch.

## Source and arithmetic at the first greedy disagreement

The independent greedy control used the same checkpoint, image and prompt IDs and disabled speculation. After 88 matching generated tokens, Christopher ranked the period ahead of “ or” by 0.0190773 log-probability units; TensorFold emitted “ or”. Those two Christopher scores both come from Christopher. TensorFold did not expose the two raw scores in that run. DeepSeek's reference `ParallelHead.forward` projects the final normalized hidden vector onto the vocabulary. TensorFold's native path projects BF16 weights into FP32 logits and gathers the rank slices. Christopher's TileLang path uses a packed BF16 vocabulary matrix with an exception table, FP32 accumulation and a separate sum for exceptions. At temperature zero the selection paths choose a maximum. The input to the head and the head's reduction are both possible points of divergence; the trace does not localize a first bad layer.

We independently reconstructed the 128-token layer-zero sliding window for the archived common prefix on CPU from the site's own Safetensors, real token IDs and checkpoint parameters. Layer zero has no Engram or compressed KV. At these eight late positions the sliding windows contain text tokens only. Shared inputs were reconstructed, not captured from running ranks; the CPU implementation models the semantic arithmetic rather than reproducing the GB10 instruction schedule.

| One-variable comparison | Reconstructed result |
|---|---|
| Put the learned attention sink into the initial softmax state versus add it after summing KV, keeping the same Q/K/V and segment states. | Across all 64 heads at positions 1140–1147, 1/262,144 output coordinates differed in BF16 and 0/262,144 differed at the subsequent MXFP8 projection input. This narrows this hypothesis only for those inputs and layer. |
| Round the weighted routed-expert activation to BF16 before MXFP8, keeping the weighted activation and equal-input CPU projection routines fixed. | At position 1147, 452/13,824 MXFP8 activation bytes and one scale group differed across six selected experts. 27,159/30,720 individual BF16 expert down-projection coordinates differed. Largest absolute difference was 0.00110. These are intermediate differences, not wrong answers or directly comparable to the final logit gap. |

The currently loaded `tilelang/moe.py` hashes to `1332bf6c6ad649266df8896a60a9bf851e684e333716fa8ecbee66fa3907ca36`. The local materialized source is different and already incorporates Christopher's correction. The active 1107 routed SwiGLU goes from the weighted FP32 result into MXFP8 without the reference BF16 cast. DeepSeek's `Expert.forward` explicitly casts before `w2`; TensorFold's native `gu_store` rounds its equivalent result before storing. Christopher's #1030 addresses this arithmetic discrepancy. Earlier live four-rank trials with #1030 alone and with the 0799 shared-expert ordering correction still returned the same token at position 88 and scored 5/6. With #1030 the generated sequence first diverged from the unpatched candidate at position 169. This concrete implementation difference does not explain the screenshot's first **observed selection** on its own.

For causal diagnosis, both engines' real normalized head inputs at the same forced prefix and their vocabulary scores for IDs 16 and 469 would distinguish a head-only error from upstream hidden-state drift. Recompute the two dot products in FP64 on each vector, then compare rank-local layer outputs if the divergence is already present in the vectors. For accuracy, freeze a broader paired workload with hidden expected answers, including coding/tool tasks and annotated and unannotated visual cases. Compare wins and losses on distinct tasks across engines before assigning significance to a change. A matching token stream is not an answer-quality oracle.

## Provenance

DeepSeek supplied the model and released reference code. The site checkpoint is the existing 1017 revision with the documented Huihui output-projection transplant. The native serving stack is Christopher Owen's 1107 recipe over imported vLLM/TileLang and collective dependencies. TensorFold source was read at `09c2b6cd504ba7909524cd5e6b2acca533d606f1`. The established Torch vision adaptation, runtime policy and new plain-text fix are AI-assisted site adaptations. Christopher Owen authored the earlier numerical patches. The historical fixtures and target-only traces were reused from existing runs. The new input comparisons and CPU reconstruction are AI-assisted diagnostics. Claude reviewed the causal interpretation of the format experiment. No changes to a shared endpoint, checkpoint, or upstream project were made in this investigation.
