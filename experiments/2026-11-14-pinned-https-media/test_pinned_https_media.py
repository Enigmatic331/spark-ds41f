#!/usr/bin/env python3
"""Offline tests for the process-local pinned HTTPS media policy."""

from __future__ import annotations

import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.request import url2pathname

from urllib3.util import parse_url


TRUSTED = (
    "https://raw.githubusercontent.com/christopherowen/spark-ds41f/"
    "998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/"
    "2026-09-25-vision/fixtures/shapes.png"
)
CARD = TRUSTED.replace("shapes.png", "card.png")
CANONICAL_SHA256 = "fdd722e172d644d077dc21fb35bcab2be2f70c814ddbf47c24d287af3724a80e"


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, int | None]] = []

    def get_bytes(self, url, *, timeout, allow_redirects, max_bytes):
        self.calls.append((url, allow_redirects, max_bytes))
        return b"trusted-image"

    async def async_get_bytes(self, url, *, timeout, allow_redirects, max_bytes):
        self.calls.append((url, allow_redirects, max_bytes))
        return b"trusted-image"


class FakeIO:
    @staticmethod
    def get_max_bytes() -> int:
        return 4096

    @staticmethod
    def load_bytes(value: bytes) -> bytes:
        return value


def canonical_probe(path: Path):
    source = ast.parse(path.read_text())
    methods = {
        "load_from_url",
        "load_from_url_async",
        "_assert_url_in_allowed_media_domains",
        "_load_file_url",
    }
    cls, = (
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and methods <= {getattr(item, "name", None) for item in node.body}
    )
    shell = ast.parse("from __future__ import annotations\nclass Probe: pass")
    shell.body[1].body = [
        item for item in cls.body if getattr(item, "name", None) in methods
    ]
    namespace = {
        "os": os,
        "asyncio": asyncio,
        "parse_url": parse_url,
        "url2pathname": url2pathname,
        "Path": Path,
        "global_thread_pool": None,
        "envs": SimpleNamespace(VLLM_MEDIA_URL_ALLOW_REDIRECTS=False),
        "_wrap_media_fetch_error": lambda _url, exc: exc,
    }
    exec(compile(ast.fix_missing_locations(shell), str(path), "exec"), namespace)
    return namespace


def import_overlay(path: Path, connector_class):
    connector_module = ModuleType("vllm.multimodal.media.connector")
    connector_module.MediaConnector = connector_class
    fake_modules = {
        "vllm": ModuleType("vllm"),
        "vllm.multimodal": ModuleType("vllm.multimodal"),
        "vllm.multimodal.media": ModuleType("vllm.multimodal.media"),
        "vllm.multimodal.media.connector": connector_module,
    }
    with patch.dict(sys.modules, fake_modules):
        spec = importlib.util.spec_from_file_location("_spark_pinned_https_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def make_probe(probe_class):
    probe = probe_class()
    probe.allowed_media_domains = ["raw.githubusercontent.com"]
    probe.allowed_local_media_path = None
    probe.connection = FakeConnection()
    probe._get_cached_bytes = lambda _url: None
    probe._put_cached_bytes = lambda _url, _data: None
    probe._load_data_url = lambda value, _io: value
    return probe


def rejected_sync(probe, url: str) -> None:
    before = len(probe.connection.calls)
    try:
        probe.load_from_url(url, FakeIO())
    except ValueError:
        pass
    else:
        raise AssertionError(f"unexpectedly accepted {url!r}")
    assert len(probe.connection.calls) == before, "rejection occurred after network"


async def rejected_async(probe, url: str) -> None:
    before = len(probe.connection.calls)
    try:
        await probe.load_from_url_async(url, FakeIO())
    except ValueError:
        pass
    else:
        raise AssertionError(f"unexpectedly accepted async {url!r}")
    assert len(probe.connection.calls) == before, "async rejection occurred after network"


def run(connector_path: Path, overlay_path: Path) -> None:
    canonical_before = connector_path.read_bytes()
    assert hashlib.sha256(canonical_before).hexdigest() == CANONICAL_SHA256
    namespace = canonical_probe(connector_path)
    probe_class = namespace["Probe"]
    environment = {
        "SPARK_DS41F_MEDIA_DATA_ONLY": "1",
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS": "0",
        "SPARK_DS41F_MEDIA_HTTPS_ALLOWLIST_JSON": json.dumps([TRUSTED, CARD]),
    }
    with patch.dict(os.environ, environment, clear=False):
        module = import_overlay(overlay_path, probe_class)
    assert module.TRUSTED_URLS == frozenset((TRUSTED, CARD))
    assert module.TRUSTED_HOSTS == ("raw.githubusercontent.com",)
    patched_method = probe_class._assert_url_in_allowed_media_domains
    patched_file_method = probe_class._load_file_url
    module._install_policy()
    assert probe_class._assert_url_in_allowed_media_domains is patched_method
    assert probe_class._load_file_url is patched_file_method
    probe = make_probe(probe_class)
    rejected = [
        TRUSTED.replace("https://", "http://", 1),
        TRUSTED.replace("raw.githubusercontent.com", "github.com", 1),
        TRUSTED.replace("raw.githubusercontent.com", "evil.raw.githubusercontent.com", 1),
        TRUSTED.replace("https://", "https://user@", 1),
        TRUSTED.replace("raw.githubusercontent.com", "raw.githubusercontent.com:443", 1),
        TRUSTED.replace("raw.githubusercontent.com", "raw.githubusercontent.com:444", 1),
        TRUSTED + "?download=1",
        TRUSTED + "#fragment",
        TRUSTED.replace("shapes.png", "other.png"),
        TRUSTED.replace("998b21a0b9ecdc5c61698be89e7318bef417116e", "main"),
        TRUSTED.replace("shapes.png", "shapes%2epng"),
    ]
    with ThreadPoolExecutor(1) as pool, patch.dict(os.environ, environment, clear=False):
        namespace["global_thread_pool"] = pool
        assert probe.load_from_url(TRUSTED, FakeIO()) == b"trusted-image"
        assert probe.load_from_url(CARD, FakeIO()) == b"trusted-image"
        assert asyncio.run(probe.load_from_url_async(CARD, FakeIO())) == b"trusted-image"
        assert probe.load_from_url("data:image/png;base64,fixture", FakeIO()).startswith("data:")
        for url in rejected:
            rejected_sync(probe, url)
            asyncio.run(rejected_async(probe, url))
        assert all(call[1] is False for call in probe.connection.calls)
        assert all(call[2] == 4096 for call in probe.connection.calls)

        sentinel = "file:///nonexistent-spark-ds41f-policy-sentinel.png"
        try:
            probe.load_from_url(sentinel, FakeIO())
        except ValueError as exc:
            assert "File media URLs are disabled" in str(exc)
        else:
            raise AssertionError("file URL unexpectedly accepted")
        try:
            asyncio.run(probe.load_from_url_async(sentinel, FakeIO()))
        except ValueError as exc:
            assert "File media URLs are disabled" in str(exc)
        else:
            raise AssertionError("async file URL unexpectedly accepted")

        probe.allowed_media_domains = []
        rejected_sync(probe, TRUSTED)
        probe.allowed_media_domains = ["raw.githubusercontent.com", "example.com"]
        rejected_sync(probe, TRUSTED)
        probe.allowed_media_domains = ["raw.githubusercontent.com"]

    with patch.dict(
        os.environ,
        {**environment, "VLLM_MEDIA_URL_ALLOW_REDIRECTS": "1"},
        clear=False,
    ):
        rejected_sync(probe, TRUSTED)

    # Disabling private site mode preserves the canonical allowlist behavior.
    with patch.dict(os.environ, {"SPARK_DS41F_MEDIA_DATA_ONLY": "0"}, clear=False):
        probe.allowed_media_domains = ["approved.invalid"]
        probe._assert_url_in_allowed_media_domains(parse_url("https://approved.invalid/a"))
        try:
            probe._assert_url_in_allowed_media_domains(parse_url("https://other.invalid/a"))
        except ValueError:
            pass
        else:
            raise AssertionError("canonical opt-out allowlist behavior changed")
        try:
            probe.load_from_url(
                "file:///nonexistent-spark-ds41f-policy-sentinel.png", FakeIO()
            )
        except RuntimeError as exc:
            assert "allowed-local-media-path" in str(exc)
        else:
            raise AssertionError("canonical opt-out file behavior changed")

    bad_allowlists = [
        "not-json",
        "[]",
        json.dumps([TRUSTED, TRUSTED]),
        json.dumps([TRUSTED.replace("https://", "http://", 1)]),
        json.dumps([TRUSTED + "?query=1"]),
        json.dumps([TRUSTED.replace("https://", "https://user@", 1)]),
        json.dumps([TRUSTED.replace("raw.githubusercontent.com", "raw.githubusercontent.com:444", 1)]),
    ]
    for value in bad_allowlists:
        with patch.dict(os.environ, {module.ALLOWLIST_ENV: value}, clear=False):
            try:
                module._load_allowlist()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"bad allowlist accepted: {value}")

    called = []

    async def app(scope, receive, send):
        called.append((scope, receive, send))

    middleware = module.PinnedHttpsMediaPolicyMiddleware(app)
    asyncio.run(middleware({"type": "http"}, object(), object()))
    assert len(called) == 1
    assert connector_path.read_bytes() == canonical_before
    print(
        "PASS: process-local operator exact-URL HTTPS allowlist (two fixtures); "
        "canonical file unchanged; "
        "domain/port/auth/query/path/commit guarded; redirects fail closed; "
        "sync+async rejection before I/O; data/file/opt-out behavior preserved"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connector", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    args = parser.parse_args()
    run(args.connector, args.overlay)
