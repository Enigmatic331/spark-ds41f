#!/usr/bin/env python3
"""Exercise the pinned URL and fail-closed negatives through a staged OpenAI API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request


TRUSTED = (
    "https://raw.githubusercontent.com/christopherowen/spark-ds41f/"
    "998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/"
    "2026-09-25-vision/fixtures/shapes.png"
)
TRUSTED_FIXTURES = {
    "shapes": TRUSTED,
    "card": TRUSTED.replace("shapes.png", "card.png"),
}
NEGATIVES = {
    # Loopback only if the policy regresses; it cannot create external egress.
    "untrusted_host": "https://127.0.0.1:9/not-fetched.png",
    # Same host/path but no longer the exact URL; rejected before HTTP dispatch.
    "redirect_shaped_query": TRUSTED + "?redirect=https%3A%2F%2F127.0.0.1%3A9%2F",
    "file_url": "file:///etc/passwd",
}


def payload(model: str, image_url: str) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Briefly describe the visible shapes and colors in this image.",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 256,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()


def post(
    url: str, key: str | None, body: bytes, timeout: float
) -> tuple[int, bytes, float]:
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), time.monotonic() - start
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI base, e.g. http://host:port/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    key = None if args.no_auth else os.environ.get(args.api_key_env)
    if not args.no_auth and not key:
        parser.error(f"environment variable {args.api_key_env!r} is unset")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    result = {"trusted": {}, "negative": {}}
    for name, image_url in TRUSTED_FIXTURES.items():
        status, raw, elapsed = post(
            endpoint, key, payload(args.model, image_url), args.timeout
        )
        if status != 200:
            raise RuntimeError(f"trusted {name} URL returned HTTP {status}")
        parsed = json.loads(raw)
        message = parsed["choices"][0]["message"]
        answer = (message.get("content") or "") + (
            message.get("reasoning_content") or ""
        )
        if not answer:
            raise RuntimeError(f"trusted {name} URL completed without content or reasoning")
        result["trusted"][name] = {
            "status": status,
            "seconds": round(elapsed, 6),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "request_id": parsed.get("id"),
        }
    for name, image_url in NEGATIVES.items():
        status, raw, elapsed = post(endpoint, key, payload(args.model, image_url), args.timeout)
        if status not in (400, 422):
            raise RuntimeError(f"{name} returned HTTP {status}, expected 400/422")
        result["negative"][name] = {
            "status": status,
            "seconds": round(elapsed, 6),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
        }
    result["status"] = "PASS"
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
