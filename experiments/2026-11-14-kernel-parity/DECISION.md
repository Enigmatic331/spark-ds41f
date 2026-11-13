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
