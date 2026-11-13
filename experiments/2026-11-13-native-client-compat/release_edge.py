#!/usr/bin/env python3
"""Staging-only natural-EOS, cancellation, and long-context gates.

Provenance: newly implemented deployment qualification glue, AI-assisted.  It
imports the locally composed candidate client but no upstream spark-ds41f code.
It records only hashes, token counts, timing, and pass/fail metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
from pathlib import Path
import secrets
import sys
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from typing import Any


# The archived qualification harness supplies ApiClient and base_chat. Import it
# only after main() has validated the explicitly supplied path and endpoint.


def require_capture(capture: Any, sentinel: str) -> dict[str, Any]:
    summary = capture.safe_summary(MODEL)
    exact = capture.content.strip() == sentinel
    passed = (
        capture.status == 200
        and capture.error is None
        and capture.done_count == 1
        and capture.finish_reason == "stop"
        and summary["model_exact"]
        and summary["completion_tokens"] >= 1
        and exact
    )
    summary["exact_sentinel"] = exact
    summary["passed"] = passed
    if not passed:
        raise RuntimeError(f"stream gate failed: {capture.case_id}")
    return summary


def natural_eos_five(client: ApiClient) -> dict[str, Any]:
    barrier = threading.Barrier(5)

    def one(index: int) -> dict[str, Any]:
        sentinel = f"NATIVE_EOS_{index}_OK"
        body = base_chat(
            MODEL,
            [{"role": "user", "content": f"Reply with exactly {sentinel}."}],
            48,
        )
        capture = client.stream_chat(
            body,
            f"natural-eos-{index}",
            barrier=barrier,
            timeout=300,
        )
        return require_capture(capture, sentinel)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        runs = list(pool.map(one, range(1, 6)))
    return {"passed": all(run["passed"] for run in runs), "runs": runs}


def cancellation_recovery(client: ApiClient) -> dict[str, Any]:
    def metrics() -> tuple[float, float, float]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(BASE_URL.removesuffix("/v1")+"/metrics", timeout=5) as res:
            lines = res.read(4 * 1024 * 1024).decode().splitlines()
        def gauge(name,extra=""):
            series=[float(line.split("} ",1)[1].split()[0]) for line in lines
                    if line.startswith(name+"{") and extra in line and f'model_name="{MODEL}"' in line]
            if not series: raise RuntimeError("required staging metrics unavailable")
            return sum(series)
        return (gauge("vllm:num_requests_running"),gauge("vllm:num_requests_waiting"),
                gauge("vllm:request_success_total",'finished_reason="abort"'))

    running,waiting,aborts_before=metrics()
    if running or waiting: raise RuntimeError("disconnect test needs idle staging")
    body = base_chat(
        MODEL,
        [
            {
                "role": "user",
                "content": (
                    "List consecutive integers beginning at 100000, one per line, "
                    "and continue until the response limit."
                ),
            }
        ],
        131072,
    )
    # Suppress natural EOS/length completion during this explicitly cancelled
    # synthetic request. Confirm active occupancy after receiving stream bytes.
    body.update(min_tokens=32768,ignore_eos=True)
    raw = json.dumps(body, separators=(",", ":"))
    url = urlsplit(BASE_URL)
    conn_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = conn_type(url.hostname, url.port, timeout=120)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=raw,
        headers={"content-type": "application/json", "accept": "text/event-stream"},
    )
    response = connection.getresponse()
    prefix = response.read(4096)
    status = response.status
    observed_running=0
    try:
        end=time.monotonic()+12
        while time.monotonic()<end:
            observed_running,waiting,_=metrics()
            if observed_running==1 and waiting==0: break
            time.sleep(.2)
        else: raise RuntimeError("could not observe isolated active request")
    finally:
        response.close()
        connection.close()
    closed_after_s = time.monotonic() - started
    if status != 200 or b"data:" not in prefix:
        raise RuntimeError("cancellation stream did not begin")

    # Allow the server to observe the disconnect, then require an ordinary
    # request to make progress.  The subsequent concurrency phase is the
    # stronger cleanup/leak check.
    abort_deadline = time.monotonic() + 30
    while time.monotonic() < abort_deadline:
        running, waiting, aborts_after = metrics()
        if running==0 and waiting==0:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("active generation did not drain after disconnect")
    drain_s=time.monotonic()-started-closed_after_s
    sentinel = "NATIVE_POST_CANCEL_OK"
    capture = client.stream_chat(
        base_chat(
            MODEL,
            [{"role": "user", "content": f"Reply with exactly {sentinel}."}],
            48,
        ),
        "post-cancel-recovery",
        timeout=300,
    )
    recovery = require_capture(capture, sentinel)
    return {
        "passed": recovery["passed"],
        "cancel_http_status": status,
        "cancel_prefix_bytes": len(prefix),
        "closed_after_s": closed_after_s,
        "server_abort_delta": aborts_after - aborts_before,
        "observed_running_before_close": observed_running,
        "running_after_close": running,
        "waiting_after_close": waiting,
        "drain_s": drain_s,
        "recovery": recovery,
    }


def long_context(client: ApiClient, targets: list[int]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for target in targets:
        sentinel = "REF_" + secrets.token_hex(6).upper()
        pos = target * 3 // 5
        content = ("Synthetic archive " + secrets.token_hex(6) + ":" + (" record" * pos)
                   + f"\nThe only signed reference is {sentinel}.\n"
                   + (" record" * (target - pos))
                   + "\nReturn the signed reference from the archive. Output only its exact value.")
        capture = client.stream_chat(
            base_chat(MODEL, [{"role": "user", "content": content}], 64),
            f"context-{target}",
            timeout=1800,
        )
        summary = require_capture(capture, sentinel)
        observed = int(summary["prompt_tokens"])
        size_ok = int(target * 0.95) <= observed <= int(target * 1.05)
        summary.update(
            {
                "target_prompt_tokens": target,
                "observed_prompt_tokens": observed,
                "target_within_five_percent": size_ok,
                "passed": summary["passed"] and size_ok,
            }
        )
        if not size_ok:
            raise RuntimeError(f"context tokenizer floor missed: {target}:{observed}")
        runs.append(summary)
    return {"passed": all(run["passed"] for run in runs), "runs": runs}


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> int:
    global ApiClient, base_chat, BASE_URL, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-targets", default="32768,131072,262144",
        help="comma-separated nominal prompt token targets; 1M is intentionally not default",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing network I/O without --execute")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("use a new output file")
    sys.path.insert(0, str(args.harness_dir.resolve()))
    from candidate_qualification import ApiClient, base_chat, normalize_base_url
    BASE_URL, MODEL = normalize_base_url(args.base_url), args.model
    if urlsplit(BASE_URL).path != "/v1":
        raise SystemExit("expected an explicit staging /v1 API base")
    targets = [int(value) for value in args.context_targets.split(",")]
    if not targets or any(value < 32768 or value > 1048576 for value in targets):
        raise SystemExit("invalid context targets")

    result: dict[str, Any] = {
        "schema": "spark-ds41f-native-edge-gates/v1",
        "provenance": {
            "newly_implemented": True,
            "ai_assisted": True,
            "historical_suite_claimed": False,
        },
        "candidate": {"base_url": BASE_URL, "model": MODEL},
        "status": "RUNNING",
        "started_epoch": int(time.time()),
        "prompt_bodies_logged": False,
        "phases": {},
    }
    write_result(output, result)
    client = ApiClient(BASE_URL, MODEL, None, 1800)
    try:
        result["phases"]["natural_eos_five"] = natural_eos_five(client)
        write_result(output, result)
        result["phases"]["cancellation_recovery"] = cancellation_recovery(client)
        write_result(output, result)
        result["phases"]["long_context"] = long_context(client, targets)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
        write_result(output, result)
        return 1
    result["finished_epoch"] = int(time.time())
    write_result(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
