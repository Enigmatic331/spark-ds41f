"""Fixed-resident scheduler matrix; synthetic fixtures, no full response bodies."""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from uuid import uuid4
from replay_conditions import assess_sequence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("harness-dir","output","etherscan-dir"):
        p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("base-url","model"):
        p.add_argument("--"+arg,required=True)
    p.add_argument("--deadline",type=int,required=True)
    p.add_argument("--measure-sized-inputs",action="store_true",
                   help="Calibrate the text lengths and check queue vs scheduled-to-first-token timings")
    p.add_argument("--solo-reference",type=Path,
                   help="Reuse only verified synthetic solo receipts; validate each request hash")
    a=p.parse_args()
    sys.path.insert(0,str(a.harness_dir.resolve()))
    import candidate_qualification as q
    class SizedRunner(q.Runner):
        def long_prefill_body(self, target_tokens, cell_nonce):
            # The 1017 tokenizer encodes each separated ' record' as one token.
            # Calibrate the chat framing on this boot, then verify on every actual response.
            key = "REF_" + q.sha256_text(cell_nonce)[:16].upper()
            # Place a nonce at the start so complete KV-prefix blocks differ between injections.
            size = target_tokens - getattr(self, "prompt_overhead", 64)
            split = size * 3 // 5
            prompt = ("Archive " + cell_nonce + ": " + " record" * split +
                      "\nThe only reference is " + key + ".\n" + " record" * (size - split) +
                      "\nReturn the reference exactly and nothing else.")
            return q.base_chat(self.args.model, [{"role":"user","content":prompt}], 64), key

    cls = SizedRunner if a.measure_sized_inputs else q.Runner
    runner=cls.__new__(cls)
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
        refs = None
        if a.solo_reference is not None:
            refs = json.loads(a.solo_reference.read_text())["isolated"]
            if len(refs) != 16: raise ValueError("expected sixteen archived solo controls")
        for i in range(16):
            if time.time()+300>=a.deadline: raise RuntimeError("window_expiring")
            b,expected=runner.number_sequence_body(i,runner.args.decoder_items,prefix=runner.args.resident_prefix)
            if refs is not None:
                r=refs[i]
                if not (r["exact"] and r["request_fingerprint"]==q.body_fingerprint(b) and
                        r["count"]==len(expected) and r["finish_reason"]=="stop" and r["model_exact"] and
                        r["done_count"]==1 and not r["error"] and r["http_status"]==200):
                    raise ValueError("invalid solo reference")
                iso[i]=dict(r,source="prior 16-slot configuration")
                doc["isolated"].append(iso[i]);save()
                continue
            c=runner.client.stream_chat(b,f"isolated-{i}")
            r=c.safe_summary(a.model);r["exact"]=runner.sequence_exact(c.content,expected)
            r.update(assess_sequence(c.content,expected),request_max_tokens=b["max_tokens"])
            if (not r["exact"] or c.finish_reason!="stop") and re.fullmatch(r"[0-9,\s-]+",c.content):
                # Only numeric synthetic outputs can be written. Keep this
                # outside the public summary; report count/hash/mismatch there.
                (a.output.parent/f"calibration-{i}-numeric.txt").write_text(c.content)
            doc["isolated"].append(r);save()
            runner.assert_stream(c,finishes={"stop"})
            if not r["exact"]: raise RuntimeError("isolated_task_failure_before_interference")
            iso[i]=r
        if a.measure_sized_inputs:
            b,expected=runner.long_prefill_body(4096,uuid4().hex)
            cap=runner.client.stream_chat(b,"size-calibration")
            doc["calibration"]=cap.safe_summary(a.model);save()
            runner.assert_stream(cap,finishes={"stop"})
            if cap.content.strip()!=expected: raise ValueError("size calibration retrieval failed")
            runner.prompt_overhead=64 + int(cap.usage["prompt_tokens"]) - 4096
            if not 0 < runner.prompt_overhead < 256: raise ValueError("unexpected token density")
        for n in ((16,8,4) if a.measure_sized_inputs else (4,8,16)):
            targets=[4096,16384,32768,131072,None]
            for target in targets:
                if time.time()+440>=a.deadline: raise RuntimeError("window_expiring")
                label=f"{target or 'six-images'}-n{n}"
                # A distinct run nonce keeps injections from becoming cached
                # duplicates across concurrency levels. Usage reports real sizes.
                runner.run_nonce=uuid4().hex
                doc["current_cell"]=label;save()
                row=runner.run_injection_cell(level=n,label=label,target=target)
                row["same_resident_inputs"]=all(r["request_fingerprint"]==iso[i]["request_fingerprint"]
                    for i,r in zip(row["resident_case_indexes"],row["resident_decoders"]))
                row["same_resident_bytes"]=all(r["content_sha256"]==iso[i]["content_sha256"]
                    for i,r in zip(row["resident_case_indexes"],row["resident_decoders"]))
                row["all_residents_active_through_prefill"]=all(
                    r["gaps_first_10s_after_prefill"]["samples"]>0 for r in row["resident_decoders"])
                if a.measure_sized_inputs:
                    row["all_residents_active_through_prefill"]=all(
                        r["emitted_after_injected_first_output"] for r in row["resident_decoders"])
                    timing=row["injected"]["server_timing"]
                    row["server_timing_observed"]=all(
                        "vllm."+name in timing for name in ("queue_time_ms","time_to_first_token_ms"))
                    count=row["injected"]["prompt_tokens"]
                    row["size_matched"]=target is None or abs(count-target)<=max(16,target//100)
                    if target is None:
                        targets.append(count)  # text control matched to the measured six-image input
                doc["cells"].append(row);save()
                print(label,row["passed"],row["resident_max_gap_s"],flush=True)
                if not (row["passed"] and row["same_resident_inputs"] and row["all_residents_active_through_prefill"]
                        and row.get("server_timing_observed",True) and row.get("size_matched",True)):
                    raise RuntimeError("mixed_gate_failed")
    except Exception as e:
        doc.update(status="FAIL",reason=getattr(e,"code",None) or (str(e) if type(e) is RuntimeError else type(e).__name__),
                   detail=getattr(e,"detail",None))
        if hasattr(e,"injected_summary"):
            doc["failed_injected_request"]=e.injected_summary
        save();return 2
    doc["status"]="PASS";save();return 0


if __name__=="__main__":
    sys.exit(main())
