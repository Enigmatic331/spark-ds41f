# Native four-Spark staging results

## Screenshot fix, November 14

Christopher Owen's 1107 candidate with the site's Torch vision path could read the pink “Value column clipped →” annotation in the archived debug screenshot. Under the existing low-reasoning setting it omitted that detail when producing the final JSON. Changing the private serving policy to carry relevant visual evidence into the final answer fixed that behavior on the tested four-rank configuration. The client submitted the same PNG, messages and sampling controls to the normal streaming Chat Completions endpoint.

| Check | Result on the 16-slot staging boot |
|---|---|
| Previously failing debug image | Both uncached replays passed all six historical checks. Both visible JSON answers explicitly quoted the annotation and associated it with the clipped rightmost column. Before the change both static-DSpark replays missed it. |
| October 18 Etherscan campaign | All 16 responses passed the original scorer across nine waves with one to four concurrent requests. Each ended normally with one SSE terminator and used zero cached tokens. |
| Region-specific review | All nine visual comparisons also reported the search control and its clipping in the same region. The identical-image control reported no regression. |
| Three new dashboards | Three simultaneous low-reasoning requests correctly reported an empty paper tray, a paused backup and an empty issue list for the healthy dashboard. None of the dashboard labels is present in the policy. |
| Direct API suite | All ten passed, including streamed text, reasoning, tool call and tool-result continuation, image data URL, image pair, malformed image, text after images and six-image Etherscan. This suite used the private direct endpoint and its existing no-auth contract. |

The policy adds an internal system message on eligible image turns. On this exact request it added 76 tokens, from 1,060 to 1,136. The image span stayed at 794 tokens. A diagnostic assertion expected 77 additional tokens because it used a different offline prompt. That assertion made the first controller exit with failure even though every functional gate passed. Separate replay of the exact archived message through the installed renderer reproduced 1,136. The first controller receipt is preserved unchanged.

This addresses an observed answer omission with an explicit serving policy. A causal low-level decoder defect has not been demonstrated. Typed response-format requests, assistant continuations, cached input IDs and non-default image detail do not activate this policy. Count the affected inputs and evaluate each deployment's workloads before relying on the result.

| Reproducibility record | SHA-256 or commit |
|---|---|
| Policy fork commit | `692b1288c9c31ee5b00b5cd0c052db142946b9d4` |
| Rank-zero live policy module | `6a373dd7c4a1c10f6fdca8c4e8d8e1d544f91b91097d3d7dca21f7801d795ac5` |
| Unmodified October 18 scorer | `84e2e83c77000a8941d54f23ea2266c3c7eeffa1351fe21f4a22825285b3f32e` |
| Debug fixture PNG | `b2c7109fa155eecd1c96c78bc78a3d233ecae45f1b3dae1d16fb62412de6dc64` |
| Visible debug JSON, both replays | `5997505eb9bfd4444081c1d8f6d00693a18ee62790ab215ae9ea8f59201e6633` |
| Diagnostic receipt / complete archive / three dashboards | `cf7b07a70912b77ac4df82ff23a1b417cf346b993d0744492d020d1dadc7d190` / `ec54f7d825a4fca493fb6d107a48fc4bd2341138bdc310274e121a24364ac039` / `2048c6fbb87caa574866a4d16d00e72cf1ff9c17bf226c3c77ec55df376e6aff` |

## Sustained 17-slot trial

The same renderer change on 17 slots passed the full original archive (16/16) and all nine search-field region checks. Three concurrent held-out dashboards, original/relabelled/erased annotation controls, and live streaming/non-streaming assistant continuations all passed. The tokenizer API returned 1,136 IDs matching the offline reference byte for byte. Both isolated debug cases produced the same final JSON reported above, with 0.465 and 0.446 seconds to first output and 11.00 and 11.39 seconds end to end. The one matching TensorFold control measured 1.34 and 15.92 seconds respectively.

Over 1,802 seconds the site exercised 142 waves with SQL, two-turn synthetic transfer lookups, six-image comparisons, and long text plus images. All 1,278 streamed calls and 142 expected URL denials passed. Every wave returned to zero active and queued requests. The head's idle active allocation remained at 84.00 GiB. The first 32,919-token plus six-image request was cold (7.25 seconds to first output). Subsequent long requests benefited from prefix reuse.

The tool call's p95 time to first output rose from **0.40 seconds solo to 6.15 seconds** with two concurrent image prompts. Four of 142 mixed tool streams had at least one output gap above three seconds; the longest was **4.47 seconds**. All completed correctly. There is no identical app-level p95 series on TensorFold. The closest older six-image/one-coder baseline measured a worst resident gap of 0.78–0.80 seconds on two runs, with different resident counts and prompts.

At the end of this boot the local-file URL negative returned 500 before accessing the path. That response blocks promotion of the tested media variant. The v2 connector maps disabled file input to a validation error; this branch runs only for local-file media. The intended 16-resident plus one image case did not send any requests on this boot because the wrapper looked in the wrong directory for its invoice fixture. Subsequent release decisions must use an actual protected run of that cell.

### Final media revision

On a fresh 17-slot staging boot using the revised guard, both trusted HTTPS fixtures answered correctly in visible content and stopped normally. Untrusted host, altered query and disabled local file each returned 400. Etherscan passed 16/16, region checks 9/9, and both isolated screenshot replays explicitly returned the annotated column. Every one of the 16 distinct resident streams matched its solo output byte for byte and continued after the six-image request's first token. The image request scored 5/5 with first output in 3.357 seconds and completion in 4.117 seconds. The longest observed resident event gap was 2.845 seconds during prefill and 0.143 seconds during recovery. All 17 requests were uncached.

## Provenance

The model and tokenizer originate with DeepSeek. The deployed site checkpoint uses Huihui native FP8 output-projection transplants in 13 layers. The engine is Christopher Owen's 1107 recipe and imported vLLM, TileLang, B12X, FlashInfer and NCCL/SparkNet dependencies. The native ARM64 build, Torch vision adaptation, rank integration and recovery controls are site adaptations. The grounding module, new dashboards and assessment are AI-assisted site work. The historical images, prompts and scorer come from the site's existing October 18 suite. Claude reviewed the diagnostic interpretation. These numbers describe private staging, with no upstream contribution or public client cutover implied.
