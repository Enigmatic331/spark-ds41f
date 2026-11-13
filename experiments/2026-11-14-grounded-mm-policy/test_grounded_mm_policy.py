#!/usr/bin/env python3
"""Offline sanity tests for the private grounded multimodal policy overlay."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import spark_grounded_mm_policy as policy


def request(**updates):
    values = {
        "add_generation_prompt": True,
        "continue_final_message": False,
        "response_format": None,
        "structured_outputs": None,
        "kv_transfer_params": None,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "temperature": 0.6,
        "top_p": 0.95,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def image_part(*, detail=None, uuid=None):
    image = {"url": "data:image/png;base64,AA=="}
    if detail is not None:
        image["detail"] = detail
    part = {"type": "image_url", "image_url": image}
    if uuid is not None:
        part["uuid"] = uuid
    return part


def image_messages(text="Compare these screenshots."):
    return [
        {
            "role": "developer",
            "content": "Use the caller's requested scope.",
        },
        {
            "role": "user",
            "content": [image_part(), {"type": "text", "text": text}],
        },
    ]


@contextmanager
def fake_vllm_modules():
    class FakeChatCompletionRequest(SimpleNamespace):
        pass

    class FakeOnlineRenderer:
        async def preprocess_chat(self, request, messages, *args, **kwargs):
            self.received_request = request
            self.received_messages = messages
            self.received_args = args
            self.received_kwargs = kwargs
            return messages

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.chat_completion": types.ModuleType(
            "vllm.entrypoints.openai.chat_completion"
        ),
        "vllm.entrypoints.openai.chat_completion.protocol": types.ModuleType(
            "vllm.entrypoints.openai.chat_completion.protocol"
        ),
        "vllm.renderers": types.ModuleType("vllm.renderers"),
        "vllm.renderers.online_renderer": types.ModuleType(
            "vllm.renderers.online_renderer"
        ),
    }
    modules[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionRequest = FakeChatCompletionRequest
    modules["vllm.renderers.online_renderer"].OnlineRenderer = FakeOnlineRenderer
    with patch.dict(sys.modules, modules):
        yield FakeChatCompletionRequest, FakeOnlineRenderer


class GroundedPolicyTests(unittest.TestCase):
    def test_positive_stages_leading_system_without_mutation(self):
        req = request()
        messages = image_messages()
        before = copy.deepcopy(messages)
        staged = policy.stage_grounded_messages(req, messages)

        self.assertIsNot(staged, messages)
        self.assertEqual(staged[0], {"role": "system", "content": policy.POLICY_TEXT})
        self.assertEqual(staged[1:], messages)
        self.assertIs(staged[1], messages[0])
        self.assertIs(staged[2], messages[1])
        self.assertEqual(messages, before)
        self.assertEqual(req.chat_template_kwargs, {"reasoning_effort": "low"})
        self.assertEqual(req.tools[0]["function"]["name"], "lookup")
        self.assertEqual(req.temperature, 0.6)
        self.assertEqual(req.top_p, 0.95)

    def test_caller_instructions_follow_internal_policy(self):
        messages = image_messages("Transcribe exactly; return only the text.")
        staged = policy.stage_grounded_messages(request(), messages)
        self.assertEqual([item["role"] for item in staged], ["system", "developer", "user"])
        self.assertEqual(staged[-1], messages[-1])
        self.assertNotIn(policy.POLICY_TEXT, json.dumps(messages))

    def test_idempotent_for_already_staged_copy(self):
        messages = image_messages()
        staged = policy.stage_grounded_messages(request(), messages)
        self.assertIs(policy.stage_grounded_messages(request(), staged), staged)

    def test_fail_closed_matrix(self):
        base = image_messages()
        cases = []
        cases.append((request(add_generation_prompt=False), base))
        cases.append((request(continue_final_message=True), base))
        cases.append((request(response_format={"type": "json_schema"}), base))
        cases.append((request(structured_outputs={"json": {}}), base))
        cases.append((request(kv_transfer_params={"prompt_token_ids": [1]}), base))
        cases.append((request(), [{"role": "user", "content": "text only"}]))
        cases.append((request(), [base[-1], {"role": "user", "content": "follow-up"}]))
        cases.append((request(), [{"role": "assistant", "content": [image_part()]}]))
        cases.append((request(), [{"role": "user", "content": [image_part(detail="high")]}]))
        cases.append((request(), [{"role": "user", "content": [image_part(uuid="cached")]}]))
        cases.append((request(), [{"role": "user", "content": [{"type": "image_embeds", "image_embeds": []}]}]))
        cases.append((request(), [{"role": "user", "content": [image_part(), {"type": "audio_url", "audio_url": {"url": "x"}}]}]))
        cases.append((request(), [{"role": "user", "content": [image_part(), {"type": "mystery"}]}]))

        for req, messages in cases:
            with self.subTest(req=req, messages=messages):
                self.assertIs(policy.stage_grounded_messages(req, messages), messages)

    def test_untyped_default_image_compatibility(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"image_url": "data:image/png;base64,AA=="},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        self.assertIsNot(policy.stage_grounded_messages(request(), messages), messages)

    def test_renderer_wrapper_uses_original_request_and_local_messages(self):
        with fake_vllm_modules() as (Request, Renderer):
            self.assertTrue(policy.install_renderer_policy(mode="v1"))
            self.assertFalse(policy.install_renderer_policy(mode="v1"))

            req = Request(**request().__dict__)
            messages = image_messages()
            before = copy.deepcopy(messages)
            renderer = Renderer()
            result = asyncio.run(
                renderer.preprocess_chat(req, messages, "template", marker=7)
            )

            self.assertIs(renderer.received_request, req)
            self.assertEqual(renderer.received_messages[0]["role"], "system")
            self.assertEqual(renderer.received_messages[1:], messages)
            self.assertEqual(messages, before)
            self.assertEqual(renderer.received_args, ("template",))
            self.assertEqual(renderer.received_kwargs, {"marker": 7})
            self.assertEqual(result, renderer.received_messages)

    def test_middleware_requires_exact_opt_in(self):
        async def app(scope, receive, send):
            return None

        with patch.dict(os.environ, {policy.POLICY_ENV: "off"}, clear=False):
            with self.assertRaises(RuntimeError):
                policy.GroundedMultimodalPolicyMiddleware(app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
