"""Exercise the opt-in /health schema while a deterministic real stream runs."""
import argparse
import concurrent.futures
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("harness-dir","output"): p.add_argument("--"+name,type=Path,required=True)
    for name in ("base-url","model"): p.add_argument("--"+name,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    stage=q.normalize_base_url(a.base_url)
    client=q.ApiClient(stage,a.model,None,90);root=stage.removesuffix("/v1")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),q.NoRedirect())
    doc={"status":"RUNNING","samples":[],"streams":[]}
    with a.output.open("x") as f: json.dump(doc,f)
    def save(): a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    def health(reset=False):
        start=time.monotonic()
        with opener.open(root+"/health"+("?reset_peak=1" if reset else ""),timeout=9) as res:
            s=json.load(res)
        mem=s["memory"];live=s["live"]
        assert s["status"]=="ok" and s["warming"] is False and s["model"]==a.model and s["model_ids"]==[a.model]
        assert s["max_batch_size"]==16 and all(type(mem[k]) is int for k in ("active","cache","peak"))
        assert mem["peak"]>=mem["active"]>0 and mem["cache"]>=0
        assert all(live[k]>=0 for k in ("connections","waiting","decode_tokens_per_second","prefill_tokens_per_second"))
        doc["samples"].append(dict(reset=reset,elapsed_s=time.monotonic()-start,**s));save()
        return s
    try:
        idle=health(); assert idle["live"]["connections"]==0
        peak=health(True)["memory"];assert peak["peak"]>=peak["active"]
        r=q.Runner.__new__(q.Runner);r.args=SimpleNamespace(model=a.model)
        # Previously exact short resident; identical body for both arms.
        b,expected=r.number_sequence_body(0,192,prefix="c1r1")
        for poll in (False,True):
            signal=threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                f=pool.submit(client.stream_chat,b,"health-poll" if poll else "health-control",first_output_event=signal)
                if not signal.wait(10): raise RuntimeError("no_stream_output")
                observed=[]
                while poll and not f.done():
                    observed.append(health()["live"]["connections"]);time.sleep(.35)
                c=f.result(timeout=90)
            row=c.safe_summary(a.model);row.update(sequence_exact=r.sequence_exact(c.content,expected),polling=poll)
            doc["streams"].append(row);save()
            assert row["sequence_exact"] and c.finish_reason=="stop" and c.done_count==1 and not c.error
            if poll and 1 not in observed: raise RuntimeError("did_not_observe_one_connection")
        assert doc["streams"][0]["content_sha256"]==doc["streams"][1]["content_sha256"]
        assert health()["live"]["connections"]==0
        doc["status"]="PASS"
    except Exception as exc:
        doc.update(status="FAIL",reason=type(exc).__name__,detail=str(exc)[:120]);save();return 2
    save();print("live health and controlled polling PASS",flush=True);return 0


if __name__=="__main__": sys.exit(main())
