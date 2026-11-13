"""Replay the predeclared first solo calibration, preserving its stop evidence.

AI-assisted test glue, reuses the archived qualification request constructor.
Only synthetic integer-fixture evidence is persisted. No client keys are used.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness-dir",required=True,type=Path)
    p.add_argument("--model",required=True);p.add_argument("--base-url",required=True)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    runner=q.Runner.__new__(q.Runner);runner.args=SimpleNamespace(model=a.model)
    b,expected=runner.number_sequence_body(0,640,prefix="fixed-resident-qualification")
    client=q.ApiClient(a.base_url,a.model,None,150)
    with a.output.open("x") as f: f.write('{"status":"RUNNING"}\n')
    c=client.stream_chat(b,"untouched-solo-calibration-0")
    row=c.safe_summary(a.model)
    nums=[int(s) for s in re.findall(r"-?\d+",c.content)]
    overlap=min(len(nums),len(expected))
    mismatch=next((i for i in range(overlap) if nums[i]!=expected[i]),None)
    nonnumber=re.sub(r"[0-9,\s-]","",c.content)
    row.update(request_max_tokens=b["max_tokens"],expected_count=len(expected),observed_count=len(nums),
               exact=nums==expected,stop_ok=c.finish_reason=="stop",first_value_mismatch=mismatch,
               common_prefix=overlap if mismatch is None else mismatch,
               first_observed=nums[:3],last_observed=nums[-3:],
               min_value=min(nums,default=None),max_value=max(nums,default=None),
               nonnumeric_chars=len(nonnumber),expected_last=expected[-1],
               expected_first=expected[0],digits_only_fixture=not nonnumber)
    if not nonnumber:
        row["numeric_output"]=c.content
    row["status"]="PASS" if (row["exact"] and row["stop_ok"] and c.done_count==1 and not c.error
                               and row["model_exact"] and c.status==200) else "FAIL"
    a.output.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:row.get(k) for k in ("status","finish_reason","common_prefix","observed_count","completion_tokens","request_max_tokens","first_value_mismatch","last_observed")}),flush=True)
    return 0 if row["status"]=="PASS" else 2


if __name__=="__main__": sys.exit(main())
