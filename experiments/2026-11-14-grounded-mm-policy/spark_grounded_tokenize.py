"""Site adapter to give vLLM's /tokenize route the same image-turn policy as chat."""
from functools import wraps
import os

from spark_grounded_mm_policy import POLICY_ENV, POLICY_MODE, stage_grounded_messages


def install():
    from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
    from vllm.renderers.online_renderer import OnlineRenderer

    if os.environ.get(POLICY_ENV) != POLICY_MODE:
        raise RuntimeError("The tokenize adapter requires the active image-answer policy.")
    current = OnlineRenderer.preprocess_chat
    if getattr(current, "_spark_grounded_tokenize", False):
        return

    @wraps(current)
    async def tokenize_with_policy(renderer, request, messages, *args, **kwargs):
        if isinstance(request, TokenizeChatRequest):
            messages = stage_grounded_messages(request, messages)
        return await current(renderer, request, messages, *args, **kwargs)

    tokenize_with_policy._spark_grounded_tokenize = True
    OnlineRenderer.preprocess_chat = tokenize_with_policy


class GroundedTokenizeMiddleware:
    def __init__(self, app):
        self.app = app
        install()

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)
