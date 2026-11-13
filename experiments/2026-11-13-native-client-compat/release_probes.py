"""Site-independent qualification probes over the archived local API client.

AI-assisted test code; synthetic data only. No production route defaults.
Pass the same model/prompt set to each engine. Do not call a direct probe an
authenticated application test. Record assertions and timings, not bodies.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def sql_score(text):
    match = re.fullmatch(r"\s*```(?:sql|sqlite)?\s*(.*?)\s*```\s*", text, re.S | re.I)
    query = (match.group(1) if match else text).strip()
    if len(query) > 4000 or not re.match(r"(?is)^(select|with)\b", query):
        return False
    # An isolated in-memory database; allow reading only the fixture and SUM.
    # Keep extension loading disabled, reject multiple statements and bound work.
    for rows,expected in [
        ([("a",10,1),("b",40,0),("a",8,1),("b",9,1),("c",100,0)], [("a",18),("b",9)]),
        ([("z",4,1),("c",7,1),("z",3,1),("b",1,1),("b",99,0)], [("c",7),("z",7),("b",1)]),
    ]:
        db = sqlite3.connect(":memory:")
        try:
            db.execute("create table transfers(sender text, units integer, success integer)")
            db.executemany("insert into transfers values(?,?,?)",rows)
            db.execute("pragma query_only=on")
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 16384)
            db.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 32)
            db.set_authorizer(lambda op,a,b,*_: sqlite3.SQLITE_OK if op==sqlite3.SQLITE_SELECT
                or (op==sqlite3.SQLITE_READ and a=="transfers")
                or (op==sqlite3.SQLITE_FUNCTION and str(b).lower()=="sum") else sqlite3.SQLITE_DENY)
            db.set_progress_handler(lambda: 1,10000)
            if db.execute(query).fetchmany(10)!=expected:
                return False
        except sqlite3.Error:
            return False
        finally:
            db.close()
    return True


def probe(a):
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    client=q.ApiClient(a.base_url,a.model,None,180)
    def body(messages,limit=256):
        return q.base_chat(a.model,messages,limit)
    def stream(payload,name):
        cap=client.stream_chat(payload,name)
        info=cap.safe_summary(a.model)
        ok=(cap.error is None and cap.status==200 and cap.done_count==1 and info["model_exact"]
            and len(cap.response_ids)==1 and info["completion_tokens"]>0)
        return cap,dict(info,wire_ok=ok)
    def json_obj(text):
        try:
            return q.extract_json_object(text)
        except (ValueError,TypeError,q.GateFailure):
            return {}
    def sql(i):
        cap,row=stream(body([{"role":"user","content":"Write one SQLite SELECT query for transfers(sender TEXT, units INTEGER, success INTEGER). For each sender with successful transfers (success=1), return sender and total successful units. Order by total descending, then sender ascending. Return only SQL."}]),f"sql-{i}")
        return dict(row,passed=row["wire_ok"] and cap.finish_reason=="stop" and sql_score(cap.content))
    def contract(i):
        tx="0x"+hashlib.sha256(f"local-receipt-{i}".encode()).hexdigest()
        token="TESTUSD"
        messages=[{"role":"user","content":f"Use lookup_receipt once for transaction {tx}. Then read its Transfer event. Return only JSON with symbol, recipient and decimal_amount; preserve decimal_amount as an exact string without trailing zeros. This is a local synthetic receipt."}]
        b=body(messages)
        b["tools"]=[{"type":"function","function":{"name":"lookup_receipt","description":"Read a local fixture receipt by transaction hash","parameters":{"type":"object","properties":{"tx_hash":{"type":"string"}},"required":["tx_hash"],"additionalProperties":False}}}]
        b["tool_choice"]="auto"
        cap,r=stream(b,f"tool-{i}")
        calls=list(cap.tool_calls.values())
        if not (r["wire_ok"] and cap.finish_reason=="tool_calls" and len(calls)==1
                and calls[0]["name"]=="lookup_receipt" and calls[0]["id"]
                and json_obj(calls[0]["arguments"])=={"tx_hash":tx}):
            return dict(call=r,passed=False)
        call=calls[0]
        recipient="0x"+hashlib.sha256(f"dest-{i}".encode()).hexdigest()[:40]
        units=12000000+i*100000+37500
        receipt={"status":1,"logs":[{"event":"Approval","value":"990000000"},
            {"event":"Transfer","symbol":token,"decimals":6,"to":recipient,"value":str(units)}]}
        messages += [{"role":"assistant","content":None,"tool_calls":[{"id":call["id"],"type":"function",
            "function":{"name":call["name"],"arguments":call["arguments"]}}]},
            {"role":"tool","tool_call_id":call["id"],"content":json.dumps(receipt)}]
        follow=body(messages); follow["tools"]=b["tools"]
        cap,r2=stream(follow,f"tool-follow-{i}")
        want={"symbol":token,"recipient":recipient,"decimal_amount":f"{units//1000000}.{units%1000000:06d}".rstrip("0")}
        return dict(call=r,follow=r2,passed=r2["wire_ok"] and cap.finish_reason=="stop" and json_obj(cap.content)==want)
    def defaults():
        common={"model":a.model,"messages":[{"role":"user","content":"What is nine times seven?"}]}
        ids={}; reasons={}
        variants={"omitted":{}, "low":{"reasoning_effort":"low"}, "high":{"reasoning_effort":"high"},
                  "off":{"reasoning_effort":"none"}, "template-off":{"chat_template_kwargs":{"enable_thinking":False}},
                  "minimal":{"reasoning_effort":"minimal"}, "medium":{"reasoning_effort":"medium"},
                  "xhigh":{"reasoning_effort":"xhigh"}, "max":{"reasoning_effort":"max"}, "ultra":{"reasoning_effort":"ultra"}}
        for name,opt in variants.items():
            rec=client.json_request("POST","/chat/completions",body={
                **common,**opt,"temperature":0,"max_tokens":1,"return_token_ids":True})
            d=rec.data if isinstance(rec.data,dict) else {}
            if rec.status!=200 or not d.get("prompt_token_ids"):
                return dict(passed=False,failed_variant=name,status=rec.status)
            ids[name]=q.sha256_text(json.dumps(d["prompt_token_ids"]))
            reasons[name]=d["choices"][0]["finish_reason"]
        checks=dict(default_matches_low=ids["omitted"]==ids["low"],high_differs=ids["high"]!=ids["low"],
                    off_differs=ids["off"]!=ids["low"],off_matches_template=ids["off"]==ids["template-off"],
                    minimal_matches_low=ids["minimal"]==ids["low"],
                    medium_matches_high=ids["medium"]==ids["high"],xhigh_matches_high=ids["xhigh"]==ids["high"],
                    max_distinct=ids["max"]!=ids["high"],ultra_matches_max=ids["ultra"]==ids["max"],
                    smaller_limit=all(v=="length" for v in reasons.values()))
        return dict(checks=checks,render_hashes=ids,finish=reasons,passed=all(checks.values()))
    def vision(i,large=False):
        ns=SimpleNamespace(args=SimpleNamespace(model=a.model),etherscan=a.etherscan_dir,run_nonce=f"release-{i}")
        b=q.Runner.six_image_body(ns,f"capture-{i}")
        if large:
            # Use a one-token English word calibrated from live usage. Keep the
            # visual question and all six images; filler is content, not padding IDs.
            b["messages"][0]["content"].insert(0,{"type":"text","text":"Audit context:"+(" record"*28000)+"\n"})
        cap,r=stream(b,f"vision-{i}")
        checks=q.score_etherscan(cap.content,images=6,pairs=3)
        if large: checks["long_prompt"]=r["prompt_tokens"]>=28000
        return dict(r,checks=checks,passed=r["wire_ok"] and cap.finish_reason=="stop" and all(checks.values()))
    def remote_refusal():
        b=body([{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://qualification.invalid/image.png"}},
                                          {"type":"text","text":"Describe the image."}]}])
        rec=client.json_request("POST","/chat/completions",body=q.as_nonstream(b))
        # A network/decoder failure can also be a 4xx; static policy proof is
        # separate. Assert useful error text on the live API when available.
        return dict(status=rec.status,error=rec.error,passed=rec.status in (400,422))
    tests=[]
    if a.render: tests.append(("render_defaults",defaults))
    for i in range(1,a.repeats+1):
        tests.extend([(f"sql-{i}",lambda i=i:sql(i)),(f"contract-{i}",lambda i=i:contract(i))])
    tests.append(("remote_media_refusal",remote_refusal))
    if a.multimodal:
        def mixed():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(vision,i) for i in (41,42)]+[pool.submit(sql,43),pool.submit(contract,44)]
                rows=[j.result() for j in jobs]
                return dict(rows=rows,passed=all(r["passed"] for r in rows))
        tests.extend([("simultaneous_images_text_tools",mixed),("long_text_plus_images",lambda:vision(51,True))])
    out={"schema":1,"started_utc":q.utc_now(),"status":"RUNNING","cases":{}}
    with a.output.open("x") as f: json.dump(out,f)
    def save():
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(dir=a.output.parent,mode="w",delete=False) as temp:
            json.dump(out,temp,sort_keys=True,indent=2); temp.write("\n")
        os.replace(temp.name,a.output)
    for name,fn in tests:
        try: row=fn()
        except Exception as exc: row=dict(passed=False,exception=type(exc).__name__)
        out["cases"][name]=row
        print(name,"PASS" if row["passed"] else "FAIL",flush=True)
        if not row["passed"]:
            out.update(status="FAIL",failed=name,finished_utc=q.utc_now());save();return 2
        save()
    out.update(status="PASS",finished_utc=q.utc_now());save();return 0


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness-dir",type=Path,required=True);p.add_argument("--base-url",required=True)
    p.add_argument("--model",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--etherscan-dir",type=Path);p.add_argument("--multimodal",action="store_true")
    p.add_argument("--render",action="store_true");p.add_argument("--repeats",type=int,default=3)
    a=p.parse_args()
    if a.multimodal and not a.etherscan_dir: p.error("multimodal tests require the fixture directory")
    if not 1<=a.repeats<=10: p.error("invalid repeat count")
    sys.exit(probe(a))
