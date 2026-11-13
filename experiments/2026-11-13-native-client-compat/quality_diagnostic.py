"""Isolated synthetic visual and numeric probes. No application prompts or credentials."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from archived_visual import ARCHIVED_SHA
from visual_region_check import search_field_clipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "base-url"):
        p.add_argument("--"+name, required=True)
    for name in ("harness-dir", "fixture-dir", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--deadline", type=int, required=True)
    p.add_argument("--followup", action="store_true", help="Replay count boundaries and test fixed, defect-independent quadrant views")
    p.add_argument("--reference-count-only", action="store_true",
                   help="One production-reference numeric request with its backend's diagnostic token IDs")
    a = p.parse_args()
    os.umask(0o077)
    a.output.mkdir(mode=0o700, exist_ok=False)
    private = a.output/"synthetic-answers"
    private.mkdir(mode=0o700)
    source = a.fixture_dir.parent/"run_benchmark.py"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=ARCHIVED_SHA:
        raise ValueError("unexpected archive")
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location("archived_visual_fixture",source)
    archive=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    from PIL import Image
    client=q.ApiClient(a.base_url,a.model,None,185)
    result={"status":"RUNNING","visual":[],"count":[],"prefix_probes":[]}
    def save():
        (a.output/"results.tmp").write_text(json.dumps(result,indent=2)+"\n")
        (a.output/"results.tmp").replace(a.output/"results.json")
    def time_left():
        if time.time()+250>a.deadline: raise RuntimeError("recovery reserve reached")
    def post(path,body):
        time_left()
        rec=client.json_request("POST",path,body=body)
        if rec.status!=200 or not rec.data:
            raise RuntimeError(f"staging {path} returned {rec.status}, {rec.error}")
        return rec.data
    def uri(image):
        buf=io.BytesIO()
        image.save(buf,format="PNG")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    def visual(label,variant,temp,seed,kind="visual",effort=None):
        msgs,paths,_=archive.image_messages(kind)
        if variant=="header":
            for i,path in enumerate(paths):
                with Image.open(path) as src:
                    msgs[0]["content"][i]["image_url"]["url"]=uri(src.crop((850,0,1440,70)))
        if variant in ("split","quadrants"):
            original=msgs[0]["content"]
            details=[]
            if variant=="quadrants":
                regions=(("upper left",(0,0,720,450)),("upper right",(720,0,1440,450)),
                         ("lower left",(0,450,720,900)),("lower right",(720,450,1440,900)))
            else:
                regions=[("top navigation",(860,0,1440,70)),("market figure",(1110,93,1370,211)),
                         ("right table edge",(1020,440,1440,850))]
            for name,rect in regions:
                details.append({"type":"text","text":f"{name}, baseline then candidate:"})
                for path in paths:
                    with Image.open(path) as src:
                        details.append({"type":"image_url","image_url":{"url":uri(src.crop(rect))}})
            msgs[0]["content"]=original[:2]+details+[original[-1]]
        if variant=="header":
            msgs[0]["content"][-1]["text"]=("Compare these two navigation strips. Describe only visible differences "
                "between the first and second image. Mention which control changed and how, if any.")
        if variant=="scan":
            msgs[0]["content"][-1]["text"]+=(
                " Compare corresponding regions in reading order, top to bottom and left to right. "
                "For each reported change, check that it describes the same control in both images.")
        body=dict(model=a.model,messages=msgs,max_tokens=4096,temperature=temp,top_p=.95,seed=seed,
                  stream=True,stream_options={"include_usage":True},cache_salt=uuid4().hex)
        if effort is not None:
            body["reasoning_effort"]=effort
        time_left()
        c=client.stream_chat(body,label)
        if c.status!=200 or c.error or c.done_count!=1: raise RuntimeError(f"visual transport failure {label}")
        # These are only the hash-pinned synthetic fixtures. Never save reasoning or image data.
        (private/(label+".txt")).write_text(c.content)
        row=c.safe_summary(a.model)
        scored=archive.score(kind,c.content) if variant!="header" else {}
        checks=scored.get("checks",{})
        if kind=="visual" and variant!="header":
            row["search_clipping_in_same_region"]=search_field_clipped(scored.get("parsed") or {})
        row.update(label=label,variant=variant,kind=kind,effort=effort,seed=seed,temperature=temp,checks=checks)
        result["visual"].append(row);save()
        print(label,c.finish_reason,checks,flush=True)
    start=170962510
    end=170964557
    ref="ba1de413c2829d658ec63edd9125c7891687053c52bbaab45b41b2f78da455f0"
    body,expected=q.Runner.number_sequence_body(SimpleNamespace(args=SimpleNamespace(model=a.model)),
        34170,2048,prefix="interference-text-32768-n4")
    if q.body_fingerprint(body)!=ref: raise ValueError("archived count request changed")
    def count(label,shift=0):
        b=q.as_nonstream(body)
        b["return_token_ids"]=True
        b["messages"]=[dict(body["messages"][0])]
        if shift:
            b["messages"][0]["content"]=b["messages"][0]["content"].replace(str(start),str(start+shift)).replace(str(end),str(end+shift))
        reply=post("/chat/completions",b)
        choice=reply["choices"][0]
        ids=choice["token_ids"]; prompt=reply["prompt_token_ids"]
        text=choice["message"].get("content") or ""
        nums=[int(x) for x in re.findall(r"\d+",text)]
        known=list(range(start+shift,end+shift+1))
        row=dict(label=label,start=start+shift,target=end+shift,observed=len(nums),last=nums[-1] if nums else None,
                 exact_prefix=nums==known[:len(nums)],answer_sha256=q.sha256_text(text),finish=choice["finish_reason"],
                 stop_reason=choice.get("stop_reason"),terminal_token=ids[-1],prompt_tokens=len(prompt),
                 output_ids=len(ids),request_limit=b["max_tokens"],usage=reply["usage"])
        result["count"].append(row);save()
        print(label,row["last"],row["terminal_token"],row["finish"],flush=True)
        return prompt,ids,nums
    def boundary(label,context,n=1):
        req=dict(model=a.model,prompt=context,add_special_tokens=False,max_tokens=n,temperature=0,top_p=1,
                 logprobs=5,return_token_ids=True,return_tokens_as_token_ids=True)
        reply=post("/completions",req)
        c=reply["choices"][0]; probs=c.get("logprobs") or {}
        row=dict(label=label,input_ids=len(context),choice_ids=c.get("token_ids"),finish=c["finish_reason"],
                 stop_reason=c.get("stop_reason"),text=c.get("text"),top_first=(probs.get("top_logprobs") or [None])[0])
        result["prefix_probes"].append(row);save()
        return row
    save()
    if a.reference_count_only:
        req=q.as_nonstream(body)
        req["return_token_ids"]=True
        reply=post("/chat/completions",req)
        c=reply["choices"][0]
        answer=c["message"].get("content") or ""
        nums=[int(x) for x in re.findall(r"\d+",answer)]
        runtime=reply.get("tensorfold") or {}
        tokens=runtime.get("token_ids") or []
        result["count"].append(dict(label="original-backend-greedy",observed=len(nums),
            last=nums[-1] if nums else None,exact_prefix=nums==expected[:len(nums)],
            target=end,finish=c["finish_reason"],terminal_token=tokens[-1] if tokens else None,
            output_ids=len(tokens),usage=reply["usage"],answer_sha256=q.sha256_text(answer),
            request_limit=req["max_tokens"]))
        result["status"]="COMPLETE";save()
        return
    # Exact questions come first. The following crop/greedy and suffix arms are diagnostics.
    if a.followup:
        for i in range(3): visual(f"blind-quadrants-{i}","quadrants",.6,0)
        visual("blind-quadrants-identical","quadrants",.6,0,kind="negative")
        visual("full-explicit-high","full",.6,0,effort="high")
        for i in range(2): visual(f"full-reading-order-{i}","scan",.6,0)
    else:
        for i in range(5): visual(f"archived-{i}","full",.6,0)
        visual("full-greedy","full",0,0)
        visual("header-only","header",.6,0)
        visual("regions-assisted","split",.6,0)
    prompt,ids,nums=count("archived-greedy")
    if nums and nums==expected[:len(nums)] and ids[-1]==1:
        prefix=prompt+ids[:-1]
        boundary("identical-prefix-before-stop",prefix)
        # Checkpoint tokenizer id 14 is a comma. Force only the separator in the
        # input; subsequent IDs are selected normally by the unmodified sampler.
        continuation=boundary("same-prefix-force-one-comma",prefix+[14],15)
        next_values=[int(x) for x in re.findall(r"\d+",continuation.get("text") or "")]
        result["forced_next_number_exact"]=bool(next_values and next_values[0]==nums[-1]+1)
        save()
        root=q.ApiClient(a.base_url.removesuffix("/v1"),a.model,None,20)
        def tokenize(text):
            rec=root.json_request("POST","/tokenize",body={"model":a.model,"prompt":text,"add_special_tokens":False})
            if rec.status!=200: raise RuntimeError(f"tokenize returned {rec.status}")
            return rec.data["tokens"]
        last=tokenize(", "+str(nums[-1]))
        if ids[-1-len(last):-1]!=last: raise RuntimeError("numeric token boundary mismatch")
        boundary("one-number-before-stop",prefix[:-len(last)])
        if nums[-1]<end:
            # No roll-out: teacher-force the exact remaining integers before reading the end decision.
            tail=", "+", ".join(str(x) for x in range(nums[-1]+1,end+1))
            boundary("true-target-forced-correct-history",prefix+tokenize(tail))
            # Same generated history; change only the final digit of the instruction's endpoint.
            b=q.as_nonstream(body)
            b.update(return_token_ids=True,max_tokens=1)
            b["messages"]=[{"role":"user","content":body["messages"][0]["content"].replace(str(end),str(end+1))}]
            changed=post("/chat/completions",b)["prompt_token_ids"]
            boundary("same-history-target-plus-one",changed+ids[:-1])
    count("shift-both-endpoints-26",26)
    result["status"]="COMPLETE";save()


if __name__=="__main__":
    main()
