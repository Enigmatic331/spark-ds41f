"""Replay known synthetic resident requests; record output and timing separately.

AI-assisted experiment using the operator's archived candidate_qualification.py
as the request constructor. No model or site-specific credentials are included.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace


def assess_sequence(content, expected):
    values = [int(s) for s in re.findall(r"-?\d+", content)]
    mismatch = next((i for i, (v, w) in enumerate(zip(values, expected)) if v != w), None)
    if mismatch is not None:
        outcome = "wrong_value"
    elif len(values) < len(expected):
        outcome = "short_prefix"
    elif len(values) > len(expected):
        outcome = "extra_values"
    else:
        outcome = "exact"
    return dict(count=len(values), expected=len(expected), sequence_exact=values == expected,
                prefix_exact=values == expected[:len(values)], first_mismatch=mismatch,
                sequence_outcome=outcome)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness-dir", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--condition", choices=["alone", "four", "four-prefill"], required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    sys.path.insert(0, str(a.harness_dir.resolve()))
    import candidate_qualification as q
    src = json.loads(a.original.read_text())
    cell, = (c for c in src["phases"]["prefill_interference"]["cells"]
             if c["label"] == "text-32768-n4")
    model = src["candidate"]["model"]
    if not a.execute:
        raise SystemExit("requires --execute")
    if a.base_url != src["candidate"]["base_url"] and a.condition != "alone":
        raise SystemExit("only the isolated control may use a different endpoint")
    ns = SimpleNamespace(args=SimpleNamespace(model=model))
    specs = []
    for i, index in enumerate(cell["resident_case_indexes"]):
        body, oracle = q.Runner.number_sequence_body(ns, index, src["configuration"]["decoder_items"],
                                                    prefix=cell["resident_prompt_prefix"])
        if q.body_fingerprint(body) != cell["resident_decoders"][i]["request_fingerprint"]:
            raise ValueError("historical resident request fingerprint mismatch")
        if a.condition != "alone" or i == 2:
            specs.append((index, body, oracle))
    os.umask(0o077)
    a.out.mkdir(exist_ok=False, parents=True)
    client = q.ApiClient(a.base_url, model, None, 210)
    events = [threading.Event() for _ in specs]
    barrier = threading.Barrier(len(specs) + 1)
    start = time.monotonic()
    rows = []
    def record(cap, oracle):
        row = cap.safe_summary(model)
        if re.fullmatch(r"[0-9,\s]+", cap.content):
            (a.out / (cap.case_id + ".txt")).write_text(cap.content)
        row.update(assess_sequence(cap.content, oracle))
        row.update(finished_s=cap.finished-start,
                   transport_ok=cap.status==200 and cap.done_count==1 and cap.error is None
                   and cap.models=={model} and len(cap.response_ids)==1
                   and cap.finish_reason in {"stop", "length"})
        return row
    injected = None
    warm = None
    window = None
    nonce = hashlib.sha256(f"new-controlled-prefill-{a.repeat}".encode()).hexdigest()[:12]
    if a.condition == "four-prefill":
        # Populate shared prefix as in the original run (reported 8192 cached).
        head, key = q.Runner.long_prefill_body(ns, 16384, nonce+"-warm")
        cap = client.stream_chat(head, "prefix-primer")
        warm = cap.safe_summary(model)
        if cap.content.strip()!=key or cap.error or cap.done_count!=1:
            raise RuntimeError("prefix primer failed")
        long_body, key = q.Runner.long_prefill_body(ns, 32768, nonce+"-long")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futs = [pool.submit(client.stream_chat,body, f"resident-{idx}", barrier=barrier,
                            first_output_event=ev) for (idx,body,_),ev in zip(specs,events)]
        barrier.wait(timeout=15)
        if a.condition == "four-prefill":
            if not all(ev.wait(timeout=15) for ev in events):
                raise RuntimeError("resident streams not producing output")
            time.sleep(1)
            begin = time.monotonic()
            cap = client.stream_chat(long_body, "injected")
            end = cap.first_output or cap.finished
            injected = dict(cap.safe_summary(model), needle_exact=cap.content.strip()==key)
            window = [begin,end]
        for f, (_,_,oracle) in zip(futs,specs):
            cap = f.result(timeout=210)
            row = record(cap,oracle)
            if window:
                row.update(gaps_during=q.gap_stats(cap.event_times,*window),
                           progress_during=any(window[0]<=t<=window[1] for t in cap.event_times))
            rows.append(row)
    result = dict(condition=a.condition,repeat=a.repeat,model=model,rows=rows,
                  injected=injected,primer=warm,started_utc_epoch=time.time()-(time.monotonic()-start))
    (a.out/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(dict(condition=a.condition,repeat=a.repeat,counts=[r["count"] for r in rows],
                          exact=[r["sequence_exact"] for r in rows],
                          transport=[r["transport_ok"] for r in rows])),flush=True)
    if not all(r["transport_ok"] for r in rows) or (injected and
            (not injected["needle_exact"] or injected["done_count"]!=1 or injected["error"])):
        raise SystemExit("transport/injection failure")


if __name__ == "__main__":
    main()
