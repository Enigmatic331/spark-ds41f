"""Execute the actual patched URL entry methods without loading GPU dependencies."""
import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit


def run(path):
    source=ast.parse(path.read_text())
    methods={"load_from_url","load_from_url_async","_assert_url_in_allowed_media_domains"}
    cls,=(c for c in source.body if isinstance(c,ast.ClassDef)
          and methods <= {getattr(n,"name",None) for n in c.body})
    shell=ast.parse("from __future__ import annotations\nclass Probe: pass")
    shell.body[1].body=[n for n in cls.body if getattr(n,"name",None) in methods]
    ns={"os":os,"asyncio":asyncio,"parse_url":urlsplit,"global_thread_pool":None}
    exec(compile(ast.fix_missing_locations(shell),str(path),"exec"),ns)
    probe=ns["Probe"]();probe.allowed_media_domains=["approved.invalid"]
    # No network or caches are supplied. Any fallthrough before the expected
    # policy error thus fails rather than contacting anything.
    probe._load_data_url=lambda value,_: value
    with ThreadPoolExecutor(1) as pool, patch.dict(os.environ,SPARK_DS41F_MEDIA_DATA_ONLY="1"):
        ns["global_thread_pool"]=pool
        for method in (probe.load_from_url, lambda *args: asyncio.run(probe.load_from_url_async(*args))):
            for scheme in ("http","https"):
                try: method(scheme+"://approved.invalid/fixture",None)
                except ValueError as exc: assert "Remote media URLs are disabled" in str(exc)
                else: raise AssertionError("unexpected remote access")
            assert method("data:image/png;base64,fixture",None).startswith("data:")
    with patch.dict(os.environ,SPARK_DS41F_MEDIA_DATA_ONLY="0"):
        probe._assert_url_in_allowed_media_domains(urlsplit("https://approved.invalid/fixture"))
        try: probe._assert_url_in_allowed_media_domains(urlsplit("https://other.invalid/fixture"))
        except ValueError: pass
        else: raise AssertionError("upstream allowlist regression")
    print("PASS: sync/async refusal before I/O, data dispatch, unchanged opt-out allowlist")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("connector",type=Path);run(p.parse_args().connector)
