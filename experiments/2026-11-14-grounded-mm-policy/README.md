# Private grounded multimodal answer policy

Status: experimental, opt-in, and not deployed or qualified.

This overlay adds one fixed, generic server policy to the model input only for
a fresh Chat Completions turn whose final caller message contains ordinary
default-detail image URLs and text. It is intended to test whether Christopher's
native DSv4.1 candidate can consistently carry visual evidence from reasoning
into its final answer without changing the model, decoder, official image
encoder, API request schema, tools, auth, or caller sampling controls.

The fixed `v1` policy is:

> Within the caller's requested scope and format, for image-based answers,
> carry the relevant visible evidence you identified into the final response.
> When an image contains an annotation that explains a finding, include its
> text with that finding. For comparisons, tie each change to its specific
> region. Report only what the images support, do not add unrequested details,
> and follow the requested format.

There is no expected answer, benchmark vocabulary, fixture identity, prompt
hash, or case-specific branch in the implementation.

## Activation and rendering boundary

Only rank 0 should load the middleware, and only in a guarded staging boot:

```text
SPARK_DS41F_GROUNDED_MM_POLICY=v1
--middleware spark_grounded_mm_policy.GroundedMultimodalPolicyMiddleware
```

The experiment directory must be importable without changing the canonical
runtime tree. The guarded launcher uses it as rank 0's working directory, which
places the module on `sys.path` even under `python -S`; headless ranks do not
load it.

The ASGI middleware is a pass-through. At initialization it idempotently wraps
`OnlineRenderer.preprocess_chat`. On an eligible request the wrapper passes the
unchanged validated request plus a new local message list to the original
renderer. The new list begins with one internal `system` message containing the
policy, followed by the exact caller message objects in their original order.
Caller system/developer/user instructions therefore remain later and more
specific. The original request and its messages are not mutated and the policy
does not appear as caller-authored user text in the API transcript. It *does*
change the rendered model input, and prompt/input logging can therefore contain
the policy.

The ordinary DeepSeek V4.1 renderer then performs its existing special-token
validation, chat formatting, image parsing, and official vision encoding. The
overlay does not route to another model or alter reasoning effort, tools,
response parsing, sampling parameters, or decoding.

## Fail-closed exclusions

The policy is not staged for text-only or history-only images, continuation
prefills, explicit non-generation templates, response formats/structured
outputs, decode-side reused prompt IDs, cached UUIDs, explicit non-default image
detail, embeds/PIL objects, `input_image` aliases, audio/video, prompt embeds,
unknown modalities, mixed non-image modalities, or a final non-user turn.
There is deliberately no lexical transcription detector. Exact transcription
and other narrow caller instructions remain later in the prompt, and the policy
itself requires the caller's scope/format and forbids unrequested details.

Run the offline sanity suite with:

```sh
python3 test_grounded_mm_policy.py
```

These tests prove activation boundaries, caller-object immutability, ordering,
idempotence, exact opt-in, and wrapper plumbing. They do not prove model quality.
The archived normal request, a held-out screenshot, a counterfactual, exact
transcription, tools, and schema controls must be exercised against a guarded
staging endpoint before any promotion decision.

## Provenance

The checkpoint/tokenizer are DeepSeek's. The candidate runtime is the imported
Christopher Owen deployment over an imported/adapted vLLM materialization. This
module, its packaging, tests, and policy wording are new private site work and
AI-assisted/AI-reviewed. It copies no third-party implementation and is not an
upstream-ready contribution. Existing runtime and dependency authorship remains
unchanged.
