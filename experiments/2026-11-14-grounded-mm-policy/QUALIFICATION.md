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

## Provenance

The model and tokenizer originate with DeepSeek. The deployed site checkpoint uses Huihui native FP8 output-projection transplants in 13 layers. The engine is Christopher Owen's 1107 recipe and imported vLLM, TileLang, B12X, FlashInfer and NCCL/SparkNet dependencies. The native ARM64 build, Torch vision adaptation, rank integration and recovery controls are site adaptations. The grounding module, new dashboards and assessment are AI-assisted site work. The historical images, prompts and scorer come from the site's existing October 18 suite. Claude reviewed the diagnostic interpretation. These numbers describe private staging, with no upstream contribution or public client cutover implied.
