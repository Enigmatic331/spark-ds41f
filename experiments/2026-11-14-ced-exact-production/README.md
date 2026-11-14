# DeepSeek V4.1 exact CED production patch stack

This directory is the clean deployment patch stack extracted from the numerical
diagnosis. It deliberately excludes every trace hook and diagnostic-only model,
router, logits, and quantization change in `../2026-11-14-kernel-parity/`.

## Provenance

- Imported baseline: Christopher Owen/vLLM materialized build commit
  `2756478b60b90700b9918b457aeb38657da637b5`.
- Newly implemented, AI-assisted changes: commits `09641bae4`, `8aa3f9a9d`,
  `6e4cf616f`, and `2ba18e91d`, exported byte-for-byte as `vllm/1` through
  `vllm/4` with authorship and commit metadata preserved.
- Independent review: Claude Code challenged the decoder-depth derivation;
  local traces and live end-to-end tests, not that review, are the evidence.
- No claim is made that this stack is upstream-ready. Ownership, licensing,
  attribution, and an intended upstream diff must be reviewed separately before
  proposing a pull request.

Apply the files listed in `vllm/series` to the exact imported baseline with
`git am`. The feature is opt-in through `VLLM_DS41_CED_EXACT_WARMUP=1`; default
behavior remains unchanged.

## Causal result

Christopher's bounded CED replay retained 128 prompt rows for each of 20 decoder
layers. The correct causal frontier is `20 * (128 - 1) + 1 = 2,541` rows. The
patch stack propagates this frontier through model-state replay and uses the same
value for scheduler prefix-cache lookup, while ordinary decode remains 128 rows.

Production-profile qualification on four DGX Sparks used DSpark-static, Torch
vision, the requested `FULL_AND_PIECEWISE` profile (runtime downshifted to
`FULL_DECODE_ONLY` as before), and the unchanged original screenshot request:

- original screenshot: 6/6, token hash
  `532beafbef880c9125a8dddff742051903612b05304b3c5f0581da22371ea231`;
- 28,698-token cache pair: cached tokens `0 -> 26,112`, identical output-token
  and visible-content hashes;
- four independent concurrent text requests: 4/4;
- SQL, tool call plus continuation, render/reasoning defaults, and remote-media
  refusal: all passed;
- simultaneous two vision requests plus text and tool requests: all passed;
- 32,842-token prompt plus six images: all visual assertions passed in 7.34 s.

Sanitized live evidence is retained outside the repository in
`spark-ds41f-native-audit-20261112/results/kernel-parity-trace-o0sFKPXT/`.
