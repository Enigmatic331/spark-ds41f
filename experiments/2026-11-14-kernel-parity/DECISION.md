# Decision record

## Candidate cause

Confirmed for the locked Christopher miss: the 128-row CED replay starts the
decoder without enough earlier SWA state for the dependency frontier to
propagate through all 20 replayed decoder layers. The first observed mismatch
is layer-21 attention; the head and prompt/vision inputs are not the cause.

## Disproved causes

- Prompt-policy steering: changes the input and is not an engine fix.
- Single divergent token: forcing either branch after the common prefix still
  gives spark-ds41f 5/6.
- Offline vision side-encoder vector mismatch: captured vectors were identical.
- Routed-MoE BF16-boundary omission as the sole cause: the corrected four-rank
  build still produced 5/6 on the locked case.

## Fix

Opt-in patch `0010-fix-dsv41-propagate-CED-warmup-through-decoder-depth.patch`
extends prompt replay to 2,541 rows: the final 128 rows plus 19 earlier
127-row SWA frontiers. Decode returns to the ordinary 128-row path. Default
behavior is unchanged.

Patch `0011-fix-dsv41-preserve-exact-CED-warmup-across-prefix-hi.patch`
closes the prefix-cache path: cached global/encoder pages may be reused only
up to the point that leaves the full 2,541-row private decoder suffix to
rebuild. Decoder/drafter pages remain private. This is required for exact mode;
otherwise a cache hit still caps replay at 128 rows and bypasses the fix.

The trace-free four-commit extraction passed the original 6/6 request, a real
26,112-token prefix hit with token-identical output, independent concurrency,
tool, text, and mixed/long vision gates under the production DSpark-static and
CUDA-graph profile. See `../2026-11-14-ced-exact-production/`.

## Validation

The unchanged greedy request produced this causal ladder:

- ordinary 128-row replay: 5/6;
- 255-row one-window warm-up: 5/6;
- full decoder prefill: 6/6;
- 2,541-row propagated warm-up: 6/6 and bit-identical 1,336-token output to
  full decoder prefill.

Receipts are retained locally under `kernel-parity-trace-VFb1PAyw`,
`kernel-parity-trace-uka8UtK4` and `kernel-parity-trace-vHKvH36Z`. This proves
the locked fix, not general quality or production qualification. The active
release remains the previously qualified release17 until broader gates run.
