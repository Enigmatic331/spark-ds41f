"""Private, opt-in grounded-image answer policy for the native DSv4.1 trial.

This is a site overlay, not an upstream vLLM module.  It is loaded only as an
explicit ASGI middleware.  The middleware does not inspect or rewrite HTTP
bodies; its initialization installs one process-local wrapper at the renderer
boundary.  The wrapper stages a new message list for model rendering while
leaving the validated request and its caller-authored messages untouched.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any


POLICY_ENV = "SPARK_DS41F_GROUNDED_MM_POLICY"
POLICY_MODE = "v1"
POLICY_TEXT = (
    "Within the caller's requested scope and format, for image-based answers, "
    "carry the relevant visible evidence you identified into the final "
    "response. When an image contains an annotation that explains a finding, "
    "include its text with that finding. For comparisons, tie each change to "
    "its specific region. Report only what the images support, do not add "
    "unrequested details, and follow the requested format."
)

_INSTALL_MARKER = "_spark_ds41f_grounded_mm_policy_v1"
_TEXT_PART_TYPES = frozenset(("text", "input_text", "output_text"))
_log = logging.getLogger(__name__)


def _is_default_image_url(part: Mapping[str, Any]) -> bool:
    """Accept only an ordinary, uncached image URL with default detail."""
    if part.get("uuid") is not None:
        return False

    image_url = part.get("image_url")
    nested_detail: Any = None
    if isinstance(image_url, str):
        url = image_url
    elif isinstance(image_url, Mapping):
        url = image_url.get("url")
        nested_detail = image_url.get("detail")
    else:
        return False

    detail = part.get("detail", nested_detail)
    return isinstance(url, str) and bool(url.strip()) and detail in (None, "auto")


def _has_only_default_images_and_text(content: Any) -> bool:
    """Return true for a normal OpenAI image/text content list only."""
    if not isinstance(content, list):
        return False

    image_count = 0
    for part in content:
        if isinstance(part, str):
            continue
        if not isinstance(part, Mapping):
            return False

        part_type = part.get("type")
        if part_type in _TEXT_PART_TYPES:
            if not isinstance(part.get("text", ""), str):
                return False
            continue

        # The untyped form is accepted by vLLM as a compatibility spelling,
        # but it still must contain only a normal image_url payload.
        if part_type == "image_url" or (
            part_type is None and "image_url" in part
        ):
            if not _is_default_image_url(part):
                return False
            image_count += 1
            continue

        # Fail closed for embeds, PIL objects, cached UUIDs, audio/video,
        # prompt embeds, input_image aliases, and unknown future modalities.
        return False

    return image_count > 0


def _has_reused_prompt_ids(request: Any) -> bool:
    params = getattr(request, "kv_transfer_params", None)
    return isinstance(params, Mapping) and bool(params.get("prompt_token_ids"))


def _should_stage_policy(request: Any, messages: Sequence[Any]) -> bool:
    """Apply only to a fresh, unconstrained final user image turn."""
    if getattr(request, "add_generation_prompt", None) is not True:
        return False
    if getattr(request, "continue_final_message", False):
        return False
    if getattr(request, "response_format", None) is not None:
        return False
    if getattr(request, "structured_outputs", None) is not None:
        return False
    if _has_reused_prompt_ids(request) or not messages:
        return False

    final = messages[-1]
    if not isinstance(final, Mapping) or final.get("role") != "user":
        return False
    return _has_only_default_images_and_text(final.get("content"))


def stage_grounded_messages(request: Any, messages: list[Any]) -> list[Any]:
    """Return an internal render-only message list, or the original object.

    The policy is a leading system message so it cannot be mistaken for
    caller-authored user text.  Every caller message follows it unchanged,
    preserving later system/developer instructions and the exact final user
    request.  Nested caller objects are deliberately shared, not mutated.
    """
    if not _should_stage_policy(request, messages):
        return messages
    if (
        messages
        and isinstance(messages[0], Mapping)
        and messages[0].get("role") == "system"
        and messages[0].get("content") == POLICY_TEXT
    ):
        return messages
    return [{"role": "system", "content": POLICY_TEXT}, *messages]


def install_renderer_policy(*, mode: str | None = None) -> bool:
    """Install the process-local renderer wrapper once.

    Returns ``True`` when this call installs the wrapper and ``False`` when an
    identical wrapper is already active.  Any mode other than the exact
    versioned opt-in fails closed.
    """
    selected = os.environ.get(POLICY_ENV) if mode is None else mode
    if selected != POLICY_MODE:
        raise RuntimeError(
            f"{POLICY_ENV} must equal {POLICY_MODE!r} when the private "
            "grounded multimodal middleware is loaded"
        )

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.renderers.online_renderer import OnlineRenderer

    current = OnlineRenderer.preprocess_chat
    if getattr(current, _INSTALL_MARKER, False):
        return False

    @wraps(current)
    async def preprocess_chat_with_grounded_policy(
        renderer: Any,
        request: Any,
        messages: list[Any],
        *args: Any,
        **kwargs: Any,
    ):
        staged = messages
        if isinstance(request, ChatCompletionRequest):
            staged = stage_grounded_messages(request, messages)
        return await current(renderer, request, staged, *args, **kwargs)

    setattr(preprocess_chat_with_grounded_policy, _INSTALL_MARKER, True)
    setattr(preprocess_chat_with_grounded_policy, "_unwrapped", current)
    OnlineRenderer.preprocess_chat = preprocess_chat_with_grounded_policy
    _log.warning(
        "Enabled private grounded multimodal answer policy %s at the "
        "render-only message boundary",
        POLICY_MODE,
    )
    return True


class GroundedMultimodalPolicyMiddleware:
    """Pass-through ASGI middleware whose initialization installs the hook."""

    def __init__(self, app: Any) -> None:
        self.app = app
        install_renderer_policy()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)
