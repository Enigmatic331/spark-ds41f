import asyncio
import json
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from native_health import HealthMiddleware


class TestHealth(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls=[];self.http_status=200;self.bad_memory=False
        async def collect(name,timeout,args):
            self.calls.append((name,timeout,args))
            return ([None]*4 if self.bad_memory else
                    [None,{"active":10,"cache":3,"peak":11},None,None])
        st=NS(num_waiting_reqs=1,num_skipped_waiting_reqs=1)
        log=NS(last_scheduler_stats=st,last_prompt_throughput=10,last_generation_throughput=20)
        self.client=NS(get_num_unfinished_requests=lambda:5,
                       logger_manager=NS(stat_loggers=[NS(per_engine_stat_loggers={0:log}),NS()]),
                       collective_rpc=collect,vllm_config=NS(scheduler_config=NS(max_num_seqs=16)))
        self.state=NS(engine_client=self.client,args=NS(api_key=None,served_model_name=["model"]))
        async def app(scope,receive,send):
            await send(dict(type="http.response.start",status=self.http_status,
                            headers=[(b"x-example",b"a"),(b"content-length",b"0")]))
            await send(dict(type="http.response.body",body=b""))
        self.m=HealthMiddleware(app)

    async def get(self,path="/health",query=b""):
        messages=[]
        async def send(m): messages.append(m)
        await self.m(dict(type="http",method="GET",path=path,query_string=query,app=NS(state=self.state)),
                     None,send)
        return messages[0]["status"],dict(messages[0]["headers"]),messages[1]["body"]

    async def test_samples_once_and_reset_is_forwarded(self):
        with patch.dict(os.environ,VLLM_API_KEY=""):
            code,headers,data=await self.get()
            self.assertEqual(code,200)
            self.assertEqual(int(headers[b"content-length"]),len(data))
            self.assertEqual(headers[b"x-example"],b"a")
            doc=json.loads(data)
            self.assertEqual(doc["live"],dict(connections=5,waiting=2,decode_tokens_per_second=20,
                                               prefill_tokens_per_second=10))
            self.assertEqual(doc["memory"],dict(active=10,cache=3,peak=11))
            await self.get()
            await self.get(query=b"reset_peak=1")
        self.assertEqual([v[2] for v in self.calls],[(False,),(True,)])

    async def test_original_error_or_other_route_unchanged(self):
        self.http_status=503
        self.assertEqual((await self.get())[0],503)
        self.http_status=200
        self.assertEqual((await self.get("/v1/models"))[2],b"")
        self.assertFalse(self.calls)

    async def test_auth_disclosure(self):
        with patch.dict(os.environ,VLLM_API_KEY="test-only"):
            self.assertEqual(json.loads((await self.get())[2]),{"status":"ok"})
        self.assertFalse(self.calls)

    async def test_unavailable_is_not_fabricated(self):
        self.bad_memory=True
        with patch.dict(os.environ,VLLM_API_KEY=""):
            self.assertEqual((await self.get())[0],503)


if __name__=="__main__": unittest.main()
