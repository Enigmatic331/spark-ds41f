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
    "shapes": {
        "url": TRUSTED,
        "question": (
            "Describe the shapes and their colours in this image, and read the text "
            "in it exactly."
        ),
        "expected": (("red",), ("circle",), ("blue",), ("square", "rectangle"), ("SPARK3",), ("42",)),
    },
    "card": {
        "url": TRUSTED.replace("shapes.png", "card.png"),
        "question": "What is the large title in this image, and what number is shown for Code aggregate?",
        "expected": (("DeepSeek",), ("173",)),
    },
}
NEGATIVES = {
    # Loopback only if the policy regresses; it cannot create external egress.
    "untrusted_host": "https://127.0.0.1:9/not-fetched.png",
    # Same host/path but no longer the exact URL; rejected before HTTP dispatch.
    "redirect_shaped_query": TRUSTED + "?redirect=https%3A%2F%2F127.0.0.1%3A9%2F",
    "file_url": "file:///nonexistent-spark-ds41f-policy-sentinel.png",
}


def payload(model: str, image_url: str, question: str) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
            "seed": 17,
            "chat_template_kwargs": {"reasoning_effort": "low"},
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
    for name, case in TRUSTED_FIXTURES.items():
        status, raw, elapsed = post(
            endpoint,
            key,
            payload(args.model, case["url"], case["question"]),
            args.timeout,
        )
        if status != 200:
            raise RuntimeError(f"trusted {name} URL returned HTTP {status}")
        parsed = json.loads(raw)
        choice = parsed["choices"][0]
        message = choice["message"]
        answer = message.get("content") or ""
        missing = [
            " or ".join(options)
            for options in case["expected"]
            if not any(option.lower() in answer.lower() for option in options)
        ]
        if not answer or missing:
            raise RuntimeError(
                f"trusted {name} final content failed known answer: missing {missing}"
            )
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(
                f"trusted {name} did not terminate normally: {choice.get('finish_reason')!r}"
            )
        result["trusted"][name] = {
            "status": status,
            "seconds": round(elapsed, 6),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "final_content_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "finish_reason": choice.get("finish_reason"),
            "known_answer": True,
            "request_id": parsed.get("id"),
        }
    for name, image_url in NEGATIVES.items():
        status, raw, elapsed = post(
            endpoint,
            key,
            payload(args.model, image_url, "Describe this image briefly."),
            args.timeout,
        )
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
