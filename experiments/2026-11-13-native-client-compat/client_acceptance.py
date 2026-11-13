"""Run synthetic text/reasoning, tool continuation and vision through a real client route.

Supply an already-authorized DSH/Tokios or Sembang route and a scoped client
credential file. No service registration or route change. Never print the
credential, image bytes or response bodies. Results alone do not attest which
upstream served them: independently verify the route targets the candidate.
AI-assisted local qualification code.
"""
import argparse
import base64
import ipaddress
import json
import os
import re
import stat
import sys
import urllib.parse
from pathlib import Path


def run(a):
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    u=urllib.parse.urlsplit(a.base_url)
    try: loopback=ipaddress.ip_address(u.hostname or "").is_loopback
    except ValueError: loopback=u.hostname=="localhost"
    if not u.hostname or u.username or u.password or u.query or u.fragment or not (
        u.scheme=="https" or (loopback and u.scheme=="http")):
        raise ValueError("use the authorized HTTPS endpoint or local loopback client ingress")
    if u.hostname.lower()!=a.expected_host.lower():
        raise ValueError("endpoint does not match the separately supplied trusted hostname")
    mode=a.key_file.stat().st_mode
    if a.key_file.is_symlink() or not stat.S_ISREG(mode) or mode & 0o077:
        raise ValueError("expected a directly readable, protected file containing a client key")
    key=q.load_key(a.key_file)
    client=q.ApiClient(a.base_url.rstrip("/"),a.model,key,120)
    def one(name,b):
        c=client.stream_chat(b,name)
        r=c.safe_summary(a.model)
        wire=(c.status==200 and c.error is None and c.done_count==1 and r["model_exact"]
              and r["completion_tokens"]>0 and r["response_request_id_count"]==1)
        return c,r,wire
    def b(messages,cap,thinking=False):
        return q.base_chat(a.model,messages,cap,thinking=thinking)
    results={"route_kind":a.route_kind,"candidate_attribution":"requires independent route verification","cases":{}}
    os.umask(0o077)
    # Reserve a new path; never clobber an earlier acceptance receipt.
    with a.output.open("x") as f: json.dump(dict(results,status="RUNNING"),f)
    def check(name,ok,**fields):
        results["cases"][name]=dict(passed=bool(ok),**fields)
        results["status"]="RUNNING" if ok else "FAIL"
        a.output.write_text(json.dumps(results,indent=2)+"\n")
        print(name,results["cases"][name]["passed"],flush=True)
        if not ok: raise SystemExit(2)
    text=b([{"role":"user","content":"Return exactly APP_TEXT_OK."}],64)
    denied=client.json_request("POST","/chat/completions",body=q.as_nonstream(text),auth=False)
    check("auth_required",denied.status in (401,403),status=denied.status)
    c,r,w=one("client-text",text)
    check("text",w and c.content.strip()=="APP_TEXT_OK" and c.finish_reason=="stop",request=r)
    c,r,w=one("client-reasoning",b([{"role":"user","content":"Briefly reason about 17 times 23; give the number as your final answer."}],512,True))
    check("reasoning",w and bool(c.reasoning.strip()) and "391" in c.content and c.finish_reason=="stop",request=r)
    messages=[{"role":"user","content":"Call receipt_code with id 'fixture-7'. Then reply with only the code returned by the tool."}]
    tools=[{"type":"function","function":{"name":"receipt_code","description":"Read a synthetic local receipt code.",
           "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}}]
    req=b(messages,192);req.update(tools=tools,tool_choice="auto")
    c,r,w=one("client-tool",req); calls=list(c.tool_calls.values())
    args=q.extract_json_object(calls[0]["arguments"]) if len(calls)==1 else {}
    check("tool",w and c.finish_reason=="tool_calls" and len(calls)==1 and calls[0]["name"]=="receipt_code"
          and bool(calls[0]["id"]) and args=={"id":"fixture-7"},request=r)
    call=calls[0]
    messages.extend([{"role":"assistant","content":None,"tool_calls":[{"id":call["id"],"type":"function",
        "function":{"name":call["name"],"arguments":call["arguments"]}}]},
        {"role":"tool","tool_call_id":call["id"],"content":'{"code":"RECEIPT_91_OK"}'}])
    req=b(messages,96);req.update(tools=tools)
    c,r,w=one("client-tool-result",req)
    check("tool_continuation",w and c.content.strip()=="RECEIPT_91_OK" and c.finish_reason=="stop",request=r)
    import hashlib
    image=a.invoice_png.read_bytes()
    if hashlib.sha256(image).hexdigest()!="f23c5dd998662cca8cbf83fe5f6bd9f6c28ea7d1fdc1251c67259506a617a98b":
        raise ValueError("wrong invoice fixture for this oracle")
    uri="data:image/png;base64,"+base64.b64encode(image).decode()
    req=b([{"role":"user","content":[{"type":"image_url","image_url":{"url":uri}},
             {"type":"text","text":"Read the invoice number and total; include both in your answer."}]}],128)
    c,r,w=one("client-vision",req)
    check("image",w and "INV-48213" in c.content.upper() and re.search(r"(?<![\d.])942(?:\.00)?(?![\d.])",c.content)
          and c.finish_reason=="stop",request=r)
    results["status"]="PASS"
    a.output.write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for s in ("key-file","output","invoice-png","harness-dir"):
        p.add_argument("--"+s,required=True,type=Path)
    for s in ("model","base-url","expected-host"):
        p.add_argument("--"+s,required=True)
    p.add_argument("--route-kind",required=True,choices=["tokios","sembang"])
    run(p.parse_args())
