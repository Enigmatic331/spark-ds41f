# Native client compatibility investigation

Status: experimental; not promoted or fully qualified for DSH/Sembang.

Development destination: [Enigmatic331/spark-ds41f](https://github.com/Enigmatic331/spark-ds41f),
branch `native-client-compat`. The checkout's `origin` and default push remote
point to that fork; `upstream` retains Christopher Owen's repository.

This captures the narrow reasoning-field adaptation from a Dockerless TP4 trial
using existing Safetensors over NFS. It does not contain a native deployment
installer or change the promoted recipe. Cluster configuration and raw local
service records remain outside this public experiment.

## Source provenance

- Deployment base: [Christopher Owen's spark-ds41f at 998b21a0](https://github.com/christopherowen/spark-ds41f/commit/998b21a0b9ecdc5c61698be89e7318bef417116e).
- Underlying serving source: [Local Inference Lab vLLM at 19f2c20e](https://github.com/local-inference-lab/vllm/commit/19f2c20ed4d64ae22bd009ceacadd85e2a81e354),
  followed by the imported 1107 vLLM series named in `upstreams.lock.json`.
- Existing vLLM code and the deployment project retain their Apache-2.0 licenses
  and original history. Other imported dependencies retain their own notices.
- This patch is a local adaptation of two existing response serializers. The
  patch packaging, tests, and investigation notes were AI-assisted. Claude
  reviewed the alias proposal and later challenged the early-stop diagnosis.
- No flattened dependency source or imported patch stack is added by this change.
  This is a fork experiment, not an upstream pull request or an original engine.

## Reasoning field correction

The tested frontend produced `reasoning`, but the existing client contract
expects `reasoning_content`. A model response could therefore contain reasoning
while a legacy client failed to display or preserve it. The patch emits both
fields with equal values whenever the serialized `reasoning` field is non-null,
for full chat messages and streaming deltas. It preserves content, tool calls,
and the canonical reasoning field. It changes response serialization only.

Apply `0001-preserve-reasoning-content.patch` **after** the pinned 1107 source
series, within its prepared vLLM tree. It is intentionally not added to the
promoted lock or release series. Verify applicability with `git apply --check`
before applying. The exact source file hashes are:

| Path below vLLM tree | Before SHA-256 | After SHA-256 |
|---|---|---|
| `vllm/entrypoints/openai/chat_completion/protocol.py` | `979097239d30cb56b4ddb1e1d582efbf37e7ccedd354963cff761b3f46731750` | `957b2ecf0742a53debaa5465fe92ef56bc5e93396db3bc521ade137d954ea51e` |
| `vllm/entrypoints/generate/base/protocol.py` | `cbf29ae008b7b51234d74e602f1395b5cbef660fa79e65eedab8228e9c49c9ee` | `e5d674a1679d9562b81d8def59c465b8b343b02178719f0b2cf10d5d6a011a9c` |

Run the CPU-only regression suite with the already installed Pydantic 2 runtime:

```sh
python test_reasoning_compat.py --vllm-source /path/to/prepared/vllm
```

It tests the actual serializer method bodies without importing vLLM or CUDA.
Coverage includes omitted, null, empty and populated reasoning, serialization
flags, full JSON and dictionary output, content and tool preservation, field
selection, and nested envelopes. Negative controls remove the alias and require
the regression check to detect the original problem. This does not substitute
for complete API serialization or actual application tests.

Verification on 2026-11-13: 136 serialization cases passed with Pydantic 2.13.5,
and both negative controls detected the omitted legacy field. The patch also
passed reverse applicability checking against the two deployed trial files;
removing only this delta reconstructs the recorded before hashes. These checks
required no inference service launch or GPU imports.

The earlier native trial passed direct SSE reasoning, a basic reasoning task,
tools and continuation, screenshot/image fixtures, and a controlled restart
after this adaptation. Actual DSH/Sembang end-to-end qualification is still open.

## Early-ending answer: observation versus cause

In a synthetic mixed-prefill test, four resident requests each enumerated 2,048
different consecutive integers while a separate 48,548-token request prefills.
One resident returned exactly the first 1,948 integers, then normal `stop` and
`[DONE]`. Offline reconstruction matched its recorded SHA-256. The maximum
stream event gap in that cell was 1.099 s; all residents progressed. A different
cell showed a 4.049 s pause and complete answers.

This is a failed answer-completeness assertion, not proof that the scheduler or
speculative decoder caused premature termination. Ordinary model early stopping
is still plausible. The follow-up controls below supersede the original
uncertainty about whether load was necessary. No generation/scheduler fix is included.

## Controlled replay, November 13

The four resident request bodies match the historical fingerprints exactly.
The focal request yielded the same 1,948-number prefix, byte for byte, in all
six Christopher runs. Every peer returned all 2,048 numbers in every wave.
Every stream completed with HTTP 200, one terminal marker, and normal stop.

| Condition | Repeats | Focal integers (requested 2,048) | Worst resident event gap |
|---|---:|---:|---:|
| Christopher, alone | 2 | 1,948 both times | 0.048 s |
| Christopher, four residents | 2 | 1,948 both times | 0.075 s |
| Christopher, four plus long prefill | 2 | 1,948 both times | 1.033 s |
| TensorFold, identical request alone | 2 | 1,548 both times | See receipts |

The new injected prompts used distinct nonces (51,529 and 53,018 actual tokens,
8,192 cached in each, versus 48,548 and 8,192 in the original). Both needle
answers passed, and all residents progressed during both prefills. The measured
gaps are between client SSE events, not kernel scheduling measurements.
TensorFold also returned an exact prefix with clean termination, but a different
length. This fails the task oracle on both engines. It rules out competing
decoders or long prefill as a necessary cause for this Christopher result.
Neither model-chosen EOS nor a backend-specific decoding cause is proven; no
speculation-off or terminal-logit comparison was made.

The original test changed the resident prompts along with the prefill size and
used one failure name for output mismatch, starvation, and excessive gaps.
The adapted local harness now holds resident requests fixed across prefill sizes
at each concurrency and records each failing assertion separately. Historical
records remain unchanged: reclassifying the old failure yields only
`resident_output_mismatch`. Reports still fail task quality for wrong answers.
The opt-in `local-harness-gates.patch` targets the external site qualification
harness, **not** the vLLM runtime. The new AI-assisted replay/comparison tools
and sanitized receipts are beside this file. Numeric bodies are retained locally.

The direct reasoning/tool/vision suite passed on the replay boot. Actual
DSH/Sembang routes, image interference, long-running soak, and production
qualification remain open. The controller restored the known-good service.
During this window the original baseline stop launcher was found to remove its
containers, defeating the rank-local restart fallback. The new opt-in baseline
control helper verifies all four container IDs, image, and auto-remove flags;
the corrected cutover confirmed all original IDs survived in stopped state.
The full multi-node deadline expiry was not deliberately exercised.

## Fix and retest queue

1. Audit the shared test harness and separate incomplete output, stalls,
   disconnects, timeouts and crashes in results. Preserve complete synthetic
   fixtures and add independently checkable realistic tasks.
2. Replay the exact failed prompt alone and with identical concurrent load;
   repeat and obtain a targeted baseline control. If load-dependent, capture
   stop-token provenance, batch/request mapping and speculative acceptance
   before changing one configuration variable at a time.
3. Complete reasoning on/off and effort-level coverage, history round trips,
   tools after reasoning, and the actual DSH and Sembang routes and parsers.
4. Complete mixed image/text concurrency, image URL/error handling, prefill
   pause/recovery comparisons, cancellation, and long-context coverage.
5. Measure matched realistic coding, contract analysis, chat and tool workloads,
   with repeated latency/throughput measurements and equivalent memory budgets.
6. Verify restart, rollback coordination, external monitoring and sustained soak
   before promotion. Present actual-client results before switching production.

The native trial was rolled back after its mandatory output assertion failed.
Successful synthetic throughput alone is not a release qualification.
