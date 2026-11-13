# Where the debug screenshot detail is lost

The archived synthetic debug PNG contains a small pink callout reading “Value column clipped →”. The accompanying text supplies the CSS, HTML and console evidence. It never supplies the callout text. The archived scorer requires the final JSON to identify the clipped Value column.

On native release 1107, Torch vision, TP4 and target-only greedy decoding, the model generated the exact label during reasoning and omitted it from the visible JSON. The JSON correctly identified horizontal overflow, the clipped To column, the missing icon, the CSS cause and the repairs. The final score was 5/6. The TensorFold control included the label in its visible JSON and scored 6/6.

We decoded the recorded token IDs using the checkpoint vocabulary, split at ID 128822 (`</think>`), and removed EOS. On each arm, the first segment reproduced the reasoning-field hash and the second reproduced the exact returned content. The observed omission is present in the generated answer.

| Greedy configuration | Exact label mentions before `</think>` | Mention in visible JSON | Output tokens |
|---|---:|---|---:|
| TensorFold | 4 | yes | 1,456 |
| Christopher 1107 | 1 | no | 1,460 |
| 1107 with Christopher's #1030 routed-expert fix | 2 | no | 1,407 |
| 1107 with #1030 and Christopher's 0799 shared-expert fix | 2 | no | 1,407 |

All four arms repeated identically within their configuration. The two patched arms also produced identical token sequences to each other. In the combined arm, a separate archived request using SSE, seed 0, temperature 0.6 and top-p 0.95 produced the label in reasoning only. The server returned a normal stop and one SSE terminator. With thinking disabled for the request the visible JSON passed all six checks once. The release gate retains the original reasoning behavior.

Three new full-frame controls asked the model to transcribe the callout on the combined build. It returned the original words. Replacing the words with “Nonce column hidden” at the same coordinates and font size changed the transcription to that phrase. Erasing only the lettering produced EMPTY as requested. These tests used greedy sampling and direct-answer mode. All were uncached and completed normally. Some extra diagnostic requests overlapped the greedy replay on this boot.

We independently ran both vision paths on the exact debug PNG (SHA256 `b2c7109fa155eecd1c96c78bc78a3d233ecae45f1b3dae1d16fb62412de6dc64`) on a single SM80 GPU. All 50 common stage hashes matched. The 770 aligner rows and the 794-token completed image span were byte-identical. The span SHA256 was `5153b20c60addfd69502715c8830a17cf6879ee5b1dc0a122e99ee88a2026220`. Live transcriptions above exercised the four-rank serving path.

## History intervention

We supplied the original image and prompt to the unpatched 1107 target. We gave it either engine's captured reasoning, the closed thinking delimiter and the same one-token JSON opener. Tokenization and the actual completion both confirmed the expected prompt IDs. Both requests were uncached, used greedy decoding and finished at EOS. With TensorFold's history, Christopher generated a JSON answer passing 6/6. With its own history, it generated a JSON answer scoring 5/6. The normalized opener removed the original candidate's Markdown fence in both arms. The underlying text of the supplied histories and their lengths differed.

Forcing just the first divergent token after 88 shared output IDs produced a 5/6 answer in both arms. At that step, full-prefill and incremental decode both select ID 16. The log-probability margin over ID 469 changes from 0.0191 in decode to 0.1684 in full-prefill. These results localize the semantic difference to the model's generated history at this fixture. They do not identify the first causally relevant layer or a validated change to the default serving profile.

The history intervention also exposed a separate protocol defect. With thinking already closed in an assistant-prefilled request, the non-streaming endpoint returned the entire JSON continuation in the reasoning field. We scored the generated IDs to isolate this from content routing. A minimal alphabet continuation reproduced it. SSE returned the final two letters in content, while the non-streaming endpoint returned the same letters in reasoning and empty content. Ordinary, unprefilled archived requests place the generated final answer correctly.

## Provenance

The model and tokenizer originate with DeepSeek. Both services load the same site checkpoint, based on DeepSeek 1017 with Huihui native FP8 attention-output weights transplanted in 13 layers. The native trial uses the imported Christopher Owen 1107 recipe and its vLLM, TileLang and collective dependencies, with the site Torch vision adaptation. Christopher Owen authored both numerical corrections in his experimental patch stacks. The offline TensorFold comparison uses its imported vision implementation at `8ce1b5a528b37fdb843ed590c7341459f1c171b9` with a site diagnostic adapter. The screenshot and scorer come from the site's October 18 synthetic corpus. The counterfactuals, checks and write-up are AI-assisted site work. Claude reviewed the experimental interpretation.
