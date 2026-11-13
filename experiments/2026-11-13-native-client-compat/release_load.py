"""Fixed-resident scheduler matrix; synthetic fixtures, no full response bodies."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from uuid import uuid4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("harness-dir","output","etherscan-dir"):
        p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("base-url","model"):
        p.add_argument("--"+arg,required=True)
    p.add_argument("--deadline",type=int,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    runner=q.Runner.__new__(q.Runner)
    runner.args=SimpleNamespace(model=a.model,decoder_items=640,resident_case_base=0,
        resident_prefix="fixed-resident-qualification", max_decoder_gap_s=5.0,timeout=180,
        injection_delay=1,decoder_start_timeout=30)
    runner.client=q.ApiClient(q.normalize_base_url(a.base_url),a.model,None,180)
    runner.run_nonce=uuid4().hex;runner.etherscan=a.etherscan_dir
    doc=dict(status="RUNNING",isolated=[],cells=[],scope="synthetic scheduler diagnostic; task quality remains independent")
    os.umask(0o077)
    with a.output.open("x") as f: json.dump(doc,f)
    def save():
        with tempfile.NamedTemporaryFile(dir=a.output.parent,delete=False,mode="w") as f:
            json.dump(doc,f,sort_keys=True,indent=2)
        os.replace(f.name,a.output)
    iso={}
    try:
        for i in range(16):
            if time.time()+300>=a.deadline: raise RuntimeError("window_expiring")
            b,expected=runner.number_sequence_body(i,runner.args.decoder_items,prefix=runner.args.resident_prefix)
            c=runner.client.stream_chat(b,f"isolated-{i}")
            r=c.safe_summary(a.model);r["exact"]=runner.sequence_exact(c.content,expected)
            doc["isolated"].append(r);save()
            runner.assert_stream(c,finishes={"stop"})
            if not r["exact"]: raise RuntimeError("isolated_task_failure_before_interference")
            iso[i]=r
        for n in (4,8,16):
            for target in (4096,16384,32768,131072,None):
                if time.time()+440>=a.deadline: raise RuntimeError("window_expiring")
                label=f"{target or 'six-images'}-n{n}"
                # A distinct run nonce keeps injections from becoming cached
                # duplicates across concurrency levels. Usage reports real sizes.
                runner.run_nonce=uuid4().hex
                row=runner.run_injection_cell(level=n,label=label,target=target)
                row["same_resident_inputs"]=all(r["request_fingerprint"]==iso[i]["request_fingerprint"]
                    for i,r in zip(row["resident_case_indexes"],row["resident_decoders"]))
                row["same_resident_bytes"]=all(r["content_sha256"]==iso[i]["content_sha256"]
                    for i,r in zip(row["resident_case_indexes"],row["resident_decoders"]))
                row["all_residents_active_through_prefill"]=all(
                    r["gaps_first_10s_after_prefill"]["samples"]>0 for r in row["resident_decoders"])
                doc["cells"].append(row);save()
                print(label,row["passed"],row["resident_max_gap_s"],flush=True)
                if not (row["passed"] and row["same_resident_inputs"] and row["all_residents_active_through_prefill"]):
                    raise RuntimeError("mixed_gate_failed")
    except Exception as e:
        doc.update(status="FAIL",reason=getattr(e,"code",None) or (str(e) if type(e) is RuntimeError else type(e).__name__),
                   detail=getattr(e,"detail",None))
        save();return 2
    doc["status"]="PASS";save();return 0


if __name__=="__main__":
    sys.exit(main())
