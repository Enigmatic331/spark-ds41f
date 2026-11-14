# Target-exact DSpark verification

This experiment tests a narrowly scoped speculative verifier for the site's
four-rank DeepSeek V4.1 Flash deployment. Each target-logit row is sampled from
the existing position-keyed target RNG stream. Draft tokens are committed only
while they exactly equal those target samples; the first mismatch commits the
target token, and a fully matched block commits the bonus target token.

## Provenance

- **Imported build baseline:** Christopher Owen's materialized vLLM 1107 build
  tree, recorded locally as commit `f6d685d557328fbae4cc3ee9b9befe285d28eb99`
  with tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`. That commit is an
  imported deployment baseline, not a claim of authorship.
- **Existing site adaptations:** the native ARM64 deployment, Torch vision
  integration, API compatibility layer and exact-CED patch stack remain
  separate layers and retain their existing provenance.
- **Behavioral reference:** the exact-prefix contract was independently adapted
  from the public behavior documented and inspected in
  [TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0. No TensorFold
  Zig, CUDA, Metal, Python source or binary was copied into this patch.
- **New implementation:** vLLM commit
  `6ae956399820e56dbcdfd2a4fbf15016b30d317e` (tree
  `ec7b7928da4de2c27b7bf57974c4de0ec8d71e16`) is AI-assisted site work. It adds the opt-in `target_exact`
  verifier, fail-closed configuration checks and focused GPU tests.
- **Review:** Claude was asked for an independent design review. Its response,
  if received, is advisory; local source inspection and matched execution tests
  are the acceptance authority.

This patch is experimental and is not represented as original work for the
imported layers, ready for an upstream pull request, or suitable for publication
without a separate ownership, license, attribution and exact-diff review.

## Qualification gate

1. Fixed-seed speculative output must be token-identical to the serial target
   sampler under the same request and sampling parameters.
2. The locked visual fixture must pass without prompt or response-format
   steering, followed by text, tool, reasoning and mixed image/text checks.
3. One/four/five-stream decode and matched prefill must not materially regress
   from the current Christopher profile and must remain worthwhile versus the
   recorded TensorFold control.
4. Any material quality or speed failure restores TensorFold.

