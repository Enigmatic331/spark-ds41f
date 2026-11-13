"""Run the archived Etherscan visual campaign with its own prompts and scorer."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from uuid import uuid4
from visual_region_check import search_field_clipped


ARCHIVED_SHA = "84e2e83c77000a8941d54f23ea2266c3c7eeffa1351fe21f4a22825285b3f32e"


def run(args):
    source = args.etherscan_dir.parent / "run_benchmark.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != ARCHIVED_SHA:
        raise ValueError("archived visual campaign changed")
    if not (source.parent/"raw").is_dir():
        raise ValueError("expected archived fixture/result tree")
    os.umask(0o077)
    # Reviewed module. Its only module-level mkdir targets the verified existing
    # raw directory. Its historical main/preflight and its writers are unused.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("archived_etherscan_visual", source)
    archive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    sys.path.insert(0, str(args.harness_dir.resolve()))
    import candidate_qualification as q
    client = q.ApiClient(args.base_url, args.model, None, 200)
    waves = [("c1-negative", ["negative"])]
    for kind in ("visual", "debug"):
        waves.extend((f"c1-{kind}-r{i}", [kind]) for i in (1, 2))
    waves.extend((f"c2-r{i}", ["visual", "debug"]) for i in (1, 2))
    waves += [("c3", ["visual", "visual", "debug"]), ("c4", ["visual", "visual", "visual", "debug"])]
    if getattr(args, "visual_control", False):
        waves = [(f"c1-visual-r{i}", ["visual"]) for i in (1, 2)]
    data = {"status":"RUNNING", "source_sha256":ARCHIVED_SHA, "waves":[]}
    with args.output.open("x") as f: json.dump(data,f)
    def save(): args.output.write_text(json.dumps(data, indent=2)+"\n")
    def one(kind, index):
        msgs, _, _ = archive.image_messages(kind)
        body = dict(model=args.model, messages=msgs, max_tokens=4096, temperature=0.6, top_p=0.95,
                    seed=0, stream=True, stream_options={"include_usage":True}, cache_salt=uuid4().hex)
        c = client.stream_chat(body, f"{label}-{kind}-{index}", barrier=barrier)
        checks = archive.score(kind, c.content)
        additional = {}
        if kind == "visual":
            additional["search_clipping_in_same_region"] = search_field_clipped(checks.get("parsed") or {})
        row = c.safe_summary(args.model)
        passed = bool(checks["pass"] and all(additional.values()) and c.status==200 and not c.error and c.done_count==1
                      and c.finish_reason=="stop" and row["model_exact"] and row["completion_tokens"]>0)
        if not passed and getattr(args, "private_failure_dir", None) is not None:
            # Only this script's hash-pinned synthetic archive can reach here.
            # Store the visible JSON locally for human adjudication. Exclude reasoning.
            folder = args.private_failure_dir
            folder.mkdir(mode=0o700, parents=False, exist_ok=True)
            if folder.is_symlink() or folder.stat().st_mode & 0o077:
                raise ValueError("private failure folder must be owner-only")
            with (folder/(row["client_request_id"]+".json")).open("x") as out:
                json.dump({"case":label,"content":c.content,"checks":checks["checks"]},out)
        return dict(kind=kind, checks=checks["checks"], archived_score_passed=checks["pass"],
                    additional_checks=additional, passed=passed, **row)
    for label, kinds in waves:
        barrier = threading.Barrier(len(kinds))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(kinds)) as pool:
            rows = list(pool.map(one, kinds, range(len(kinds))))
        data["waves"].append({"label":label,"rows":rows})
        if not all(row["passed"] for row in rows) and not getattr(args, "visual_control", False):
            data.update(status="FAIL", failed=label)
            save()
            return 2
        save()
    passed = all(r["passed"] for wave in data["waves"] for r in wave["rows"])
    data["status"]="PASS" if passed else "FAIL"
    save()
    return 0 if passed else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for s in ("harness-dir","etherscan-dir","output"):
        p.add_argument("--"+s, type=Path, required=True)
    for s in ("model","base-url"):
        p.add_argument("--"+s, required=True)
    p.add_argument("--visual-control",action="store_true",help="Two isolated archived visual comparisons only")
    p.add_argument("--private-failure-dir",type=Path,help="Keep only failed synthetic fixture answers locally for review")
    sys.exit(run(p.parse_args()))
