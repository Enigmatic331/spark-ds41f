"""Repeat the checked coding, tools, and multimodal workloads for 30 minutes."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from release_probes import probe
from archived_visual import run as archived_visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "model"):
        parser.add_argument("--" + name, required=True)
    for name in ("harness-dir", "etherscan-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.harness_dir.resolve()))
    import candidate_qualification as q
    client = q.ApiClient(args.base_url.removesuffix("/v1"), args.model, None, 10)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    archive_args = SimpleNamespace(**vars(args), output=args.output_dir/"archived-visual.json",
                                   private_failure_dir=args.output_dir/"local-synthetic-failures")
    if archived_visual(archive_args) != 0:
        (args.output_dir/"status.json").write_text(json.dumps({
            "status":"NOT_STARTED", "reason":"archived visual qualification failed", "waves":0})+"\n")
        return 2
    start = time.monotonic()
    state = {"status": "RUNNING", "waves": 0, "started_epoch": time.time()}
    def save():
        state["elapsed_s"] = time.monotonic() - start
        (args.output_dir/"status.json").write_text(json.dumps(state, indent=2)+"\n")
    save()
    try:
        while time.monotonic() - start < 1800:
            if time.time()+240+max(0,1800-(time.monotonic()-start)) >= args.deadline:
                raise RuntimeError("insufficient protected time for soak and recovery")
            index = state["waves"]
            suite = SimpleNamespace(**vars(args), case_offset=index*100, render=False, repeats=1,
                multimodal=True, output=args.output_dir/f"wave-{index:04d}.json")
            if probe(suite) != 0:
                raise RuntimeError(f"workload failed in wave {index}")
            health = client.json_request("GET", "/health")
            if health.status != 200 or health.data.get("status") != "ok":
                raise RuntimeError("external health check failed")
            with (args.output_dir/"health.jsonl").open("a") as out:
                out.write(json.dumps({"elapsed_s":time.monotonic()-start,
                    "memory":health.data["memory"],"live":health.data["live"]})+"\n")
            state["waves"] += 1
            save()
    except Exception as exc:
        state.update(status="FAIL", reason=str(exc)[:180])
        save()
        return 2
    state["status"] = "PASS"
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
