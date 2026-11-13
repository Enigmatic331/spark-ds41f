# Christopher native cutover plan

The existing compiled TP4 binaries are the launch candidate. Verify their
manifest on all four hosts; rebuild natively with the recorded pins only if the
native sources, toolchain, or artifact identity change. Use the original
Safetensors read-only and keep all model and service data in place.

1. Match production defaults before benchmarking: thinking enabled at low
   effort unless overridden, output limit 131,072, existing direct model ID
   and auth, base64 image acceptance and refusal of remote image fetches.
   Inspect actual rendered token IDs and sampling parameters and consume an
   actual streamed answer. The compatibility alias alone does not cover defaults.
2. Run targeted matched TensorFold and candidate fixtures for a contract
   event, a coding artifact with an executable result oracle, and multi-turn
   tools. Report correctness and wall time independently. No repeated baseline
   restart or broad baseline qualification.
3. In a bounded four-rank window, verify independent watchdogs and an idle
   baseline, then stop only the verified existing processes and use staging.
   The baseline client endpoint is unavailable during this shared-GPU window.
   Run real image concurrency, long text plus an image, streamed reasoning,
   tools, cancellation, and 32K/128K/256K context. Test fixed independently
   verified resident prompts at 4/8/16 with long prefills; mark cache counts
   and pauses separately from content. Keep all pre-existing failed gates.
4. Before promotion, an authenticated DSH/Tokios test route and a Sembang
   test user/provider must exercise the candidate: text, reasoning, images,
   tools and continuation, usage and cancellation. No substitution of a
   connector's worker credential for a client key, or production rerouting
   represented as a staging test. Record request IDs without prompt bodies.
5. Under the same recovery guard run a 30–60 minute realistic mixed soak
   and native restart. Require no errors, persistent stalls, OOM, or leaks;
   compare task success and interactive latency with baseline. Verify the
   endpoint outside the candidate process.
6. Only after the required dispositions and gates pass, start the candidate
   on the exact client endpoint and repeat the authenticated text/image/tool
   flows. Retain the original recovery launcher. On a failed or unavailable
   mandatory gate leave the verified original service online. The previous
   counting oracle failed on both backends: its repeated result rules out
   load as a necessary cause but is not an absolute quality pass or waiver.

Estimated clean qualification window: 60–120 minutes after access and setup,
including soak, restart and final checks. A failure, missing client access,
or an unresolved mandatory quality decision precludes a promised live time.
The new plan/probes are AI-assisted site integration over the upstream sources.

## November 13 execution checkpoint

Production is on the restored baseline. The opt-in native launcher defaults
and media guard in `service-defaults.json` and patch 0002 were exercised in
native TP4 serving. The pre-existing compiled artifacts verified on all ranks;
no dependency/environment upgrade or recompile of unchanged CUDA code was needed.
The original imported files were preserved by hash on every rank. The standalone
health adapter below was not part of the GPU-tested builds.

| Gate | Result |
|---|---|
| Real chat request with omitted reasoning effort | Prompt IDs equal explicit low; high, disabled, all named aliases and max verified through normal one-token completions |
| Output limit | 131,072 set in the launcher; budget function checked on four cases, one-token client limit checked live. No 131K output was generated |
| Media | Patched sync and async remote-URL guard tested before I/O; remote image rejected live, data images passed. No external image fetch was enabled |
| Coding / synthetic receipt and tools | Three matched SQL executions and three tool-call-plus-result cases passed on both engines. First matched candidate medians 0.355/1.338 s; baseline 0.442/1.676 s. Small samples, not production p95 |
| Multimodal | Two simultaneous six-image requests alongside SQL and tool work passed; 32,842-token text-plus-six-image request passed on both corrected-probe boots |
| Natural EOS / cancellation | Five independent requests passed. In the isolated disconnect check one running request drained to zero running/waiting in 0.038 s, and a fresh completion passed |
| Context | Retrieved a randomly placed reference at 32,815 / 131,116 / 262,192 prompt tokens; respective TTFT 3.98 / 16.22 / 34.07 s. Synthetic filler, not a quality guarantee for arbitrary documents |
| Load / soak | Matrix stopped during its first solo calibration with `finish_reason_mismatch`. No 4/8/16 mixed cell or soak pass. That run did not retain the actual finish reason; evidence capture fixed, retest pending |
| Actual DSH / Sembang | Not run: the known connector is paired to production and the documented Sembang client key reference is not accessible to this account. Use a verified staging route and authorized client identity before promotion |
| Health JSON | Live default returned empty HTTP 200. New opt-in `native_health.py` has offline tests only. Needs a staging boot, honest telemetry under load, reset-peak semantics, and failure propagation verified before putting it on the client port |

Failed attempts are retained. The first probed an optional scale-out rendering
endpoint and got HTTP 404; it was replaced with normal chat requests returning
prompt token IDs. The second relied solely on the abort counter, which did not
increment on client disconnect. Verified source removes the request state before
handling later engine outputs. The corrected test observes idle → one active →
zero active/queued, with minimum output greater than possible during its short
observation, and then tests a new completion. The abort counter remained zero.

`client_acceptance.py` is for an already authenticated test route; no endpoint,
key, or staging provider is created by it. Use scoped file references rather
than chat-pasted credentials. Receipts confirm API behaviour, not route
attribution: correlate the request with the intended staging engine.

For the **next isolated build only**, install `native_health.py` in the private
site-packages, and pass `--middleware native_health.HealthMiddleware` and
`--worker-extension-cls native_health.HealthWorker`. Its worker method reports
head-worker Torch CUDA allocator statistics, not the whole machine; it caches
reads for 2 s and must be benchmarked under polling. Reject it if RPC probes
delay decode or if unavailable telemetry is misreported as healthy. Do not
promote it based on the four mocked middleware tests.

Full published receipt summaries are in `release-receipts.json`. The adapted
edge harness is in `release_edge.py`; site helpers supply the existing fixture
client. Remaining steps are live health compatibility, diagnose/complete the
fixed-input load matrix, a 30–60 minute soak, authenticated app checks, and
final same-endpoint restart/rollback verification. There is no go-live ETA
until the client paths are available and these gates are green.
