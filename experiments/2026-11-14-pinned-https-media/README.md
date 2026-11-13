# Process-local exact HTTPS media allowlist

Status: **experimental staging integration; live qualification is recorded by
the release runner, not here**.

`spark_pinned_https_media.py` is a generic rank-0 process overlay. At import it
parses `SPARK_DS41F_MEDIA_HTTPS_ALLOWLIST_JSON`, validates a unique list of one
to sixteen exact canonical HTTPS URLs, and freezes that list for the process
lifetime. It wraps `MediaConnector._assert_url_in_allowed_media_domains` in
memory. Because canonical `file:` dispatch bypasses that method and otherwise
raises `RuntimeError` when no local path is configured, the overlay also wraps
`_load_file_url` in data-only mode to raise `ValueError` before any path
resolution/open. That gives the normal client-error API envelope while the
canonical connector remains SHA256
`fdd722e172d644d077dc21fb35bcab2be2f70c814ddbf47c24d287af3724a80e`
on every rank.

The guard requires:

- an exact operator-listed URL (no wildcard or prefix expansion);
- HTTPS on implicit or explicit port 443;
- no userinfo, query or fragment;
- the connector's built-in `--allowed-media-domains` list to equal the hosts
  derived from the frozen exact URL list;
- `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`;
- existing `SPARK_DS41F_MEDIA_DATA_ONLY=1` to remain enabled.

Every other HTTP(S) URL is rejected before cache lookup or connection dispatch.
Data URLs keep their existing behavior. File URLs remain unavailable because
`--allowed-local-media-path` is unset. With site data-only mode disabled, the
canonical connector behavior is delegated unchanged.

## 17-slot staging configuration

Start rank 0 with this directory as its working directory. Do not alter
`PYTHONPATH`. Add the following repeatable middleware and built-in domain args
alongside the existing grounded-answer middleware:

```text
--allowed-media-domains raw.githubusercontent.com
--middleware spark_pinned_https_media.PinnedHttpsMediaPolicyMiddleware
```

Keep these environment values:

```text
SPARK_DS41F_MEDIA_DATA_ONLY=1
VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
SPARK_DS41F_MEDIA_HTTPS_ALLOWLIST_JSON=["https://raw.githubusercontent.com/christopherowen/spark-ds41f/998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/2026-09-25-vision/fixtures/shapes.png","https://raw.githubusercontent.com/christopherowen/spark-ds41f/998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/2026-09-25-vision/fixtures/card.png"]
```

The JSON value is operator configuration for this boot, not fixture identity
embedded in the feature. `staging-settings.json` is the machine-readable copy.

Rollback is process-local: stop the staged rank-0 process and restart without
this middleware/config. There is no connector source to restore. Before and
after the window, verify the canonical connector remains at the pinned SHA.

## Evidence and tests

The controller independently fetched both reviewed URLs with
`ssl.create_default_context()` hostname/certificate validation, port 443, and
no redirect implementation. Both returned 200 over TLS 1.3 and exactly matched
the repository copies:

- `shapes.png`: 38,601 bytes, SHA256
  `5f2eddceb0ca8205e0cbef34c5bd5ea5f317c317e523304b6df3a9635c7e80b5`;
- `card.png`: 372,357 bytes, SHA256
  `f0638ec4318deb5c373d4fd588f57280a3a5a1d189d5696a88c4c45ad80e0a7d`.

Offline policy test:

```bash
/home/enigmatic331/models/gemma4-cmp170hx-lab/.venv/bin/python \
  test_pinned_https_media.py \
  --connector /home/enigmatic331/models/spark-ds41f-native-audit-20261112/runtime-src/vllm/vllm/multimodal/media/connector.py \
  --overlay spark_pinned_https_media.py
```

It executes the canonical sync/async URL entry methods around the process-local
wrapper, tests both allowed URLs, rejects scheme/host/subdomain/userinfo/port/
query/fragment/path/branch/encoding variants before fake network I/O, checks
redirect fail-closed behavior, checks invalid operator configs, verifies data,
file and opt-out behavior, and hashes the canonical file before/after. A second
test imports the overlay through the real installed `MediaConnector` and proves
the nonexistent file sentinel is rejected as `ValueError` before the original
loader.

Controller TLS/content check:

```bash
/home/enigmatic331/models/gemma4-cmp170hx-lab/.venv/bin/python \
  verify_pinned_https_fixture.py \
  --fixture-root /home/enigmatic331/models/spark-ds41f-review-20261112/experiments/2026-09-25-vision/fixtures
```

Staged API check (use `--no-auth` only on the explicitly no-auth staging port):

```bash
python3 probe_staged_media_policy.py \
  --base-url http://STAGING:PORT/v1 --model MODEL --no-auth
```

The staged gate gives each request 2,048 completion tokens with explicit low
reasoning, `temperature=0`, and seed 17. It passes only when both exact URL
image requests return 200, terminate with `finish_reason=stop`, and their final content passes the original
known-answer checks: shapes must include red/circle/blue/square-or-rectangle,
`SPARK3`, and `42`; card must include `DeepSeek` and `173`. All three negative
cases (loopback untrusted host, same-host redirect-shaped query, and a
nonexistent file sentinel) must return 400 or 422. The negative URLs cannot
create external egress even if the guard regresses.

## Provenance

- **Imported interface:** the Apache-2.0 vLLM `MediaConnector` in Christopher
  Owen's promoted 1107 lineage `59f523c543cb61e5a2d0df616cb4efa459677b35`.
- **Upstream fixture content:** Christopher Owen's `spark-ds41f` commit
  `998b21a0b9ecdc5c61698be89e7318bef417116e`; authorship is not claimed here.
- **New current-work layer:** the process-local wrapper, offline tests, probes,
  configuration and notes are newly implemented with AI assistance.
- **Not copied:** no TensorFold source or binary code is present.

This bounded experiment is not represented as upstream-ready. Publication
would still require ownership/licensing review and a deliberately generalized
test plan.
