"""Private, process-local exact HTTPS media allowlist.

Imported as an additional vLLM ``--middleware`` on the rank-0 API process. The
module patches only ``MediaConnector._assert_url_in_allowed_media_domains`` in
memory; the canonical connector file remains byte-for-byte unchanged.

The operator supplies a JSON list of exact reviewed URLs. The list is parsed and
frozen at module import, so it cannot expand during the process lifetime.

Provenance: newly implemented, AI-assisted site integration over the imported
Apache-2.0 vLLM MediaConnector interface.
"""

from __future__ import annotations

import json
import os
from typing import Any

from urllib3.util import parse_url

from vllm.multimodal.media.connector import MediaConnector


ALLOWLIST_ENV = "SPARK_DS41F_MEDIA_HTTPS_ALLOWLIST_JSON"
_PATCH_MARKER = "spark_ds41f_pinned_https_v1"


def _load_allowlist() -> tuple[frozenset[str], tuple[str, ...]]:
    raw = os.environ.get(ALLOWLIST_ENV)
    if raw is None:
        raise RuntimeError(f"{ALLOWLIST_ENV} must contain a JSON list of exact URLs")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ALLOWLIST_ENV} is not valid JSON") from exc
    if (
        not isinstance(values, list)
        or not 1 <= len(values) <= 16
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise RuntimeError(
            f"{ALLOWLIST_ENV} must be a unique non-empty JSON string list (max 16)"
        )
    hosts: set[str] = set()
    for value in values:
        spec = parse_url(value)
        if (
            spec.url != value
            or spec.scheme != "https"
            or spec.hostname is None
            or spec.port not in (None, 443)
            or spec.auth is not None
            or spec.query is not None
            or spec.fragment is not None
        ):
            raise RuntimeError(
                f"{ALLOWLIST_ENV} entries must be canonical exact HTTPS URLs on port 443 "
                "without userinfo, query, or fragment"
            )
        hosts.add(spec.hostname)
    return frozenset(values), tuple(sorted(hosts))


TRUSTED_URLS, TRUSTED_HOSTS = _load_allowlist()


def _install_policy() -> None:
    current = MediaConnector._assert_url_in_allowed_media_domains
    if getattr(current, "_spark_ds41f_policy", None) == _PATCH_MARKER:
        return
    if getattr(current, "_spark_ds41f_policy", None) is not None:
        raise RuntimeError("refusing to replace an unknown MediaConnector policy overlay")

    def guarded(self: MediaConnector, url_spec: Any) -> None:
        if os.environ.get("SPARK_DS41F_MEDIA_DATA_ONLY") != "1":
            current(self, url_spec)
            return
        trusted = (
            url_spec.url in TRUSTED_URLS
            and url_spec.scheme == "https"
            and url_spec.hostname in TRUSTED_HOSTS
            and url_spec.port in (None, 443)
            and url_spec.auth is None
            and url_spec.query is None
            and url_spec.fragment is None
            and tuple(sorted(self.allowed_media_domains)) == TRUSTED_HOSTS
        )
        if not trusted:
            raise ValueError(
                "Remote media URLs are disabled except for the operator's exact "
                "HTTPS allowlist; send a base64 data URL."
            )
        if os.environ.get("VLLM_MEDIA_URL_ALLOW_REDIRECTS") != "0":
            raise ValueError(
                "The exact HTTPS media allowlist requires redirects disabled."
            )

    guarded._spark_ds41f_policy = _PATCH_MARKER  # type: ignore[attr-defined]
    guarded._spark_ds41f_original = current  # type: ignore[attr-defined]
    MediaConnector._assert_url_in_allowed_media_domains = guarded


_install_policy()


class PinnedHttpsMediaPolicyMiddleware:
    """Pass-through ASGI middleware whose import installs the connector guard."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)


__all__ = [
    "ALLOWLIST_ENV",
    "PinnedHttpsMediaPolicyMiddleware",
    "TRUSTED_HOSTS",
    "TRUSTED_URLS",
]
