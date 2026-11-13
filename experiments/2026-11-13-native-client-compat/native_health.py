"""Opt-in health compatibility for the native TP4 site. No inference-path work.

AI-assisted adaptation to the documented TensorFold status schema. "active",
"cache", and "peak" here describe the head worker's PyTorch CUDA allocator,
not total unified memory or other ranks. Throughput is vLLM's latest interval.
"""
import asyncio
import json
import os
import time
from urllib.parse import parse_qs


class HealthWorker:
    def site_allocator_snapshot(self, reset_peak=False):
        if self.rank != 0:
            return None
        import torch
        device=self.device
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device)
        active=torch.cuda.memory_allocated(device)
        return dict(active=active,cache=torch.cuda.memory_reserved(device)-active,
                    peak=torch.cuda.max_memory_allocated(device))


def live_snapshot(client):
    wait=decode=prefill=0
    seen=0
    for outer in client.logger_manager.stat_loggers:
        group=getattr(outer,"per_engine_stat_loggers",None)
        for log in (group.values() if group is not None else (outer,)):
            if not hasattr(log,"last_scheduler_stats"):
                continue
            samples=getattr(log,"last_scheduler_stats_dict",None)
            for stats in (samples.values() if samples is not None else (log.last_scheduler_stats,)):
                wait+=stats.num_waiting_reqs+stats.num_skipped_waiting_reqs
            decode+=log.last_generation_throughput
            prefill+=log.last_prompt_throughput
            seen+=1
    if not seen:
        raise ValueError("no scheduler telemetry")
    return dict(connections=client.get_num_unfinished_requests(),waiting=wait,
                decode_tokens_per_second=round(decode,1),prefill_tokens_per_second=round(prefill,1))


class HealthMiddleware:
    def __init__(self,app):
        self.app=app
        self.lock=asyncio.Lock()
        self.sample=None
        self.expires=0.

    async def __call__(self,scope,receive,send):
        if scope["type"]!="http" or scope["method"]!="GET" or scope["path"]!="/health":
            return await self.app(scope,receive,send)
        # Let the normal middleware/health handler determine admission, engine
        # readiness and HTTP status. Never replace an upstream failure with OK.
        messages=[]
        async def keep(message): messages.append(message)
        await self.app(scope,receive,keep)
        if not messages or messages[0].get("status")!=200:
            for m in messages: await send(m)
            return
        state=scope["app"].state
        client=state.engine_client
        status=200
        try:
            body={"status":"ok"}
            # Match the baseline's limited public health disclosure if auth is on.
            if not (state.args.api_key or os.environ.get("VLLM_API_KEY")):
                names=state.args.served_model_name
                if not isinstance(names,list) or not names or not client: raise ValueError("missing model")
                reset=parse_qs(scope.get("query_string",b"").decode()).get("reset_peak")==["1"]
                async with self.lock:
                    if reset or self.sample is None or self.expires < time.monotonic():
                        result=await client.collective_rpc("site_allocator_snapshot",timeout=5,args=(reset,))
                        head,=(entry for entry in result if entry is not None)
                        if set(head)!={"active","cache","peak"} or any(type(x) is not int or x<0 for x in head.values()):
                            raise ValueError("missing allocator telemetry")
                        self.sample=head
                        self.expires=time.monotonic()+2
                body.update(model=names[0],model_ids=names,warming=False,
                            max_batch_size=client.vllm_config.scheduler_config.max_num_seqs,
                            live=live_snapshot(client),memory=self.sample)
        except Exception:
            status=503
            body={"status":"unavailable"}
        encoded=json.dumps(body,separators=(",",":")).encode()
        headers=[(k,v) for k,v in messages[0].get("headers",[]) if k.lower() not in
                 {b"content-length",b"content-type",b"content-encoding"}]
        headers.extend([(b"content-type",b"application/json"),(b"content-length",str(len(encoded)).encode())])
        await send({"type":"http.response.start","status":status,"headers":headers})
        await send({"type":"http.response.body","body":encoded})
