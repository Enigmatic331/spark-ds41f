# Target-exact verifier result

## Verdict

**REJECTED; TensorFold restored.** The verifier passed its sampled-identity and
speed gates, but the locked seed-zero screenshot was less accurate than the
historical TensorFold result. Per the operator's quality-first rule, the
Christopher candidate was not retained.

As of `2026-11-14T21:55:00+08:00`, the unchanged client endpoint is again
`http://10.80.2.2:8888/v1`, model `DeepSeek-V4.1-Flash-1017-v4`, served by the
preserved four-rank TensorFold deployment with five streams and a
1,048,576-token context limit.

## Gates

| Gate | Result |
|---|---|
| GPU commit-kernel test | **PASS.** The TileLang target-prefix commit kernel passed direct int32 and int64 draft-token tests on DGX02. |
| Real restart/canary | **PASS.** Text, SQL, tools and continuation, remote-media refusal, two simultaneous six-image requests with text/tool traffic, and a 32,842-token-plus-six-image request completed. |
| Serial sampled identity | **PASS, 8/8.** Target-exact and speculation-off produced byte-identical visible content, hidden reasoning, token counts and finish behavior for five sampled full-frame calls, one greedy full-frame call, a header crop and a region-assisted call. |
| Simple voxel oracle | **PASS, 3/3.** Every call returned red=3, green=2 and blue=1. |
| Locked sampled screenshot | **FAIL versus TensorFold.** The target-exact/serial seed-zero answer found the red positive value and clipped Value column, but missed the clipped header search field and the requirement to report at least three issues. The historical matched TensorFold seed-zero answer passed all five checks. No prompt or response-format steering was used. |
| Matched performance | **PASS.** The verifier was within +0.31%/-0.11% of the retained Christopher prefill/decode measurements and remained substantially faster than TensorFold beyond short prompts. |
| Recovery | **PASS.** TensorFold restore, streamed terminal `[DONE]`, tool-call arguments, real image recognition, exact model identity and launcher hashes passed. The legacy minute watchdog reported the cluster healthy; all trial loops, cron entries and rank deadlines are disarmed. |

The byte identity with speculation disabled is the key causal result. The
quality difference is not caused by draft acceptance after this patch. It is
the serial Christopher target trajectory itself, so the remaining TensorFold
divergence lies in the target model's numerical/kernel path or sampler contract,
not in multi-token speculative commit logic.

## Matched speed

| Actual prompt tokens | Target-exact PP | Existing Christopher PP | TensorFold PP |
|---:|---:|---:|---:|
| 1,431 | 2,441 tok/s | 2,433 tok/s | 2,535 tok/s |
| 5,580 | 4,500 tok/s | 4,491 tok/s | 3,481 tok/s |
| 22,048 | 6,733 tok/s | 6,712 tok/s | 3,808 tok/s |
| 44,076 | 7,317 tok/s | 7,298 tok/s | 3,827 tok/s |
| Fixed-sequence decode | 178.74 tok/s | 178.93 tok/s | 148.52 tok/s |

At 44,076 prompt tokens target-exact was `+0.25%` versus the existing
Christopher block verifier and `+91.20%` versus TensorFold. Decode was `-0.11%`
versus Christopher and `+20.35%` versus TensorFold. The speed gate therefore
did not cause the rejection.

## Receipts

- `target-exact-performance.json`: matched cold-prefill and fixed-sequence decode.
- `sampled-visual-20261114T213700/results.json`: target-exact visual run.
- `specoff-visual-20261114T214930/results.json`: serial visual run.
- `restart-canary-0eg9ZNX8/release-probes.json`: API/tool/multimodal canary.

No application prompt, credential or user history is present in these
receipts. The answer text retained beside the visual receipts is generated
only from the hash-pinned synthetic test images.

## Provenance

- **Upstream/imported:** Christopher Owen's vLLM 1107 materialized deployment
  baseline (`f6d685d557328fbae4cc3ee9b9befe285d28eb99`) and its vLLM,
  TileLang, CUDA and collective layers retain their original provenance.
- **Existing adapted layers:** the site's ARM64 packaging, Torch vision, API
  compatibility/media policy, exact-CED stack and recovery controls predate
  this verifier and remain separately attributed.
- **Behavioral reference:** TensorFold's public exact-prefix semantics were
  inspected under Apache-2.0. No TensorFold source or binary was copied into
  the patch.
- **New work:** vLLM commit
  `6ae956399820e56dbcdfd2a4fbf15016b30d317e` and experiment commit
  `3b1f13f0` are AI-assisted private site work.
- **Independent review:** Claude Code was invoked, but returned no review
  because its configured USD budget was exceeded. No agreement from Claude is
  claimed; source inspection and matched execution are the authority.

This assembled diff is experimental. It is not represented as wholly
original, upstream-ready or suitable for a pull request without a separate
ownership, licensing, attribution and exact-diff review.
