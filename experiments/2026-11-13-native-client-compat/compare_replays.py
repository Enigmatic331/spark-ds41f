"""Compare the same resident under load against its own isolated controls.

The task oracle and the scheduler comparison answer different questions. A
repeatable wrong answer remains wrong, even if the scheduler preserves it.
AI-assisted experiment; reads only the numeric-fixture runner's result JSON.
"""
import argparse
import json
from pathlib import Path


def compare(results, target="resident-34170", gap_limit=5.0):
    records = {}
    fingerprints = set()
    models = set()
    groups = set()
    for result in results:
        condition, repeat = result["condition"], result["repeat"]
        key = condition, repeat
        if key in records:
            raise ValueError("duplicate condition/repeat")
        rows = result["rows"]
        focal, = (r for r in rows if r["case_id"] == target)
        if len(rows) != (1 if condition == "alone" else 4) or len({r["case_id"] for r in rows}) != len(rows):
            raise ValueError("wrong resident set")
        if condition not in {"alone", "four", "four-prefill"}:
            raise ValueError("unknown condition")
        if condition != "alone":
            groups.add(tuple(sorted((r["case_id"], r["request_fingerprint"]) for r in rows)))
        fingerprints.add(focal["request_fingerprint"])
        models.add(result["model"])
        records[key] = (result, focal)
    conditions = ("alone","four","four-prefill")
    if set(records) != {(c,r) for c in conditions for r in (1,2)}:
        raise ValueError("requires exactly two waves of every condition")
    if len(fingerprints) != 1 or len(models) != 1 or len(groups) != 1:
        raise ValueError("unmatched resident request or model")
    isolated = {records["alone",r][1]["content_sha256"] for r in (1,2)}
    digest = next(iter(isolated)) if len(isolated) == 1 else None
    cells = []
    for (condition,repeat),(result,focal) in sorted(records.items()):
        rows = result["rows"]
        injected = result["injected"]
        injection_ok = (injected is None if condition != "four-prefill" else
                        bool(injected and injected["needle_exact"] and injected["http_status"]==200
                             and injected["done_count"]==1 and injected["error"] is None
                             and injected["model_exact"] and injected["finish_reason"]=="stop"))
        timing_ok = None
        if condition == "four-prefill":
            timing_ok = all(r["progress_during"] and r["gaps_during"]["max_s"] is not None
                            and r["gaps_during"]["max_s"]<=gap_limit for r in rows)
        cells.append(dict(condition=condition, repeat=repeat, focal_count=focal["count"],
                          transport_ok=all(r["transport_ok"] for r in rows),
                          injected_ok=injection_ok, during_prefill_timing_ok=timing_ok,
                          exact_all_residents=all(r["sequence_exact"] for r in rows),
                          focal_matches_isolation=None if digest is None else focal["content_sha256"]==digest))
    return dict(model=next(iter(models)), request_fingerprint=next(iter(fingerprints)),
                isolated_hash=digest, cells=cells,
                same_focal_answer_across_conditions=all(c["focal_matches_isolation"] is True and
                    c["transport_ok"] and c["injected_ok"] for c in cells),
                full_task_pass=all(c["exact_all_residents"] and c["transport_ok"] and c["injected_ok"]
                                   and c["during_prefill_timing_ok"] is not False for c in cells))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    print(json.dumps(compare([json.loads(f.read_text()) for f in args.directory.glob("*/result.json")]),
                     sort_keys=True, indent=2))
