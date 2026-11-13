#!/usr/bin/env python3
"""Fetch the one pinned fixture with verified TLS and no redirect handling."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import ssl


HOST = "raw.githubusercontent.com"
DIRECTORY = (
    "/christopherowen/spark-ds41f/"
    "998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/"
    "2026-09-25-vision/fixtures/"
)
FIXTURES = {
    "shapes.png": "5f2eddceb0ca8205e0cbef34c5bd5ea5f317c317e523304b6df3a9635c7e80b5",
    "card.png": "f0638ec4318deb5c373d4fd588f57280a3a5a1d189d5696a88c4c45ad80e0a7d",
}
MAX_BYTES = 4 * 1024 * 1024


def fetch(name: str, expected_sha256: str, fixture_root: Path | None, timeout: float):
    connection = http.client.HTTPSConnection(
        HOST, 443, timeout=timeout, context=ssl.create_default_context()
    )
    try:
        connection.request(
            "GET",
            DIRECTORY + name,
            headers={"Host": HOST, "User-Agent": "spark-ds41f-qualification/1"},
        )
        response = connection.getresponse()
        tls_version = connection.sock.version() if connection.sock else None
        cipher = connection.sock.cipher()[0] if connection.sock else None
        if response.status != 200:
            # No redirect handler exists: every 3xx and other non-200 response fails.
            raise RuntimeError(f"expected HTTP 200, got {response.status}")
        payload = response.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise RuntimeError(f"fixture exceeded {MAX_BYTES} bytes")
    finally:
        connection.close()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"remote SHA256 {digest} != expected {expected_sha256}")
    if fixture_root is not None:
        local = (fixture_root / name).read_bytes()
        local_digest = hashlib.sha256(local).hexdigest()
        if local_digest != expected_sha256 or local != payload:
            raise RuntimeError("remote bytes do not match the repository fixture")
    return {
        "name": name,
        "http_status": 200,
        "bytes": len(payload),
        "sha256": digest,
        "tls_version": tls_version,
        "cipher": cipher,
        "redirects_followed": 0,
        "repository_fixture_equal": fixture_root is not None,
    }


def run(fixture_root: Path | None, timeout: float) -> None:
    receipts = [
        fetch(name, digest, fixture_root, timeout)
        for name, digest in FIXTURES.items()
    ]
    print(
        json.dumps(
            {
                "status": "PASS",
                "host": HOST,
                "port": 443,
                "fixtures": receipts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    run(args.fixture_root, args.timeout)
