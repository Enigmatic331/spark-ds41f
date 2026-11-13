# 0003 continued-answer field — offline evidence

The tested patch is available on the fork in commit `72f65dd7`. Production continues to run TensorFold.

- Patch: `0003-continued-answer-field.patch`
- SHA-256: `cdbe65c231f0c43a4ad2be90dc27369e596c1128c292d6e28a96aa633bfda3fc`
- Active serving source before patch: SHA-256
  `dec67c6e32f98e4e00159ae452ad5ade190d52f2487ec0fcc98ac015410edfe9`

## Source-confirmed cause

SSE passes rendered prompt token IDs into `DelegatingParser.parse_delta()`,
which observes the template's reasoning-end token before classifying generated
tokens. The non-streaming site called `DelegatingParser.parse()` with generated
text alone, so DeepSeek V4.1 restarted in its default reasoning state and
returned a continued visible suffix in `reasoning` with empty `content`.

The patch supplies a parser-only copy of the exact reasoning-end token already
present in the prompt. It does not change response text or response token IDs.
It is limited to the exact active call chain:

`ParserManager.get_parser.<locals>._Parser` →
`DeepSeekV41ParserReasoningAdapter` → `DeepSeekV41Parser`.

Activation additionally requires an assistant `continue_final_message`,
nonempty assistant content/tool calls, `is_reasoning_end(prompt_ids)`, and an
actual configured end-token ID in the rendered prompt. No request-text marker
search is performed.

## Results

The patch passed `git apply --check` against the clean fork source and both
materialized native source trees. An isolated copy of the exact active serving
source was patched and loaded controller-side with real vLLM DeepSeek parser
components, a deterministic tokenizer, and GPU visibility disabled.

| Case | Reasoning | Content | Reasoning-token count |
|---|---|---|---:|
| Unpatched non-streaming | `D E` | `None` | 3 |
| Patched non-streaming | `None` | `D E` | 0 |
| Existing SSE | `None` | `D E` | 0 |

Six synthetic controls passed: content-only continuation/SSE parity, open
reasoning, normal and reasoning-disabled turns, DSML tools, another parser
family with the same markers, literal user sentinel without a special close,
and close-then-new-think reopening.

An additional CPU replay exercised the complete `OpenAIServingChat.chat_completion_full_generator` using the active serving source with 0003 applied (SHA-256 `3a0e1785a373004aaca157c4e5d5cb45065a377b6a98d8dc033d911837881267`). With a synthetic tokenizer and a supplied request-output fixture carrying generated IDs 420, 446 and EOS, the real response object contained `D E` in content, null reasoning, three completion tokens and zero reasoning tokens. This test ends before HTTP serialization. The actual live cluster test reproduced the original defect before patching.

## Provenance

The surrounding serving/parser implementation is upstream Apache-2.0 vLLM
material. This narrow site adaptation and its tests are newly implemented,
AI-assisted work in this experiment. No third-party patch was copied or
flattened; upstream publication suitability has not been asserted.
