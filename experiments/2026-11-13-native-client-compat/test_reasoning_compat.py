"""CPU-only regression checks for the two actual vLLM response serializers.

Extract only the reviewed serializer methods with AST; importing the complete
vLLM stack would introduce GPU and optional dependency requirements. These tests
cover serialization semantics, not full API or actual-client qualification.

New AI-assisted test code. The serializer methods under test belong to the
upstream vLLM classes and retain their source provenance.
"""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
from pathlib import Path

import pydantic
from pydantic import BaseModel, Field, model_serializer


TARGETS = (
    ("vllm/entrypoints/openai/chat_completion/protocol.py", "ChatMessage"),
    ("vllm/entrypoints/generate/base/protocol.py", "DeltaMessage"),
)


def serializer_model(source: Path, class_name: str, *, remove_alias: bool = False):
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = copy.deepcopy(next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_serialize"
    ))
    if remove_alias:
        # Negative control: reproduce the former behaviour by removing only the
        # statement that adds the legacy output key.
        original_size = len(method.body)
        method.body = [
            stmt for stmt in method.body
            if not (isinstance(stmt, ast.If) and any(
                isinstance(node, ast.Constant) and node.value == "reasoning_content"
                for node in ast.walk(stmt)
            ))
        ]
        assert len(method.body) == original_size - 1, "expected exactly one alias block"
    skeleton = ast.parse('''
class Response(BaseModel):
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)
''')
    skeleton.body[0].body.append(method)
    ast.fix_missing_locations(skeleton)
    namespace = {"BaseModel": BaseModel, "Field": Field, "model_serializer": model_serializer}
    exec(compile(skeleton, str(source), "exec"), namespace)
    return namespace["Response"]


def assert_alias(data: dict) -> None:
    if data.get("reasoning") is not None:
        assert "reasoning_content" in data, "legacy reasoning field missing"
        assert data["reasoning_content"] == data["reasoning"]
    else:
        assert "reasoning_content" not in data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    args = parser.parse_args()
    cases = 0
    negative_controls = 0
    for relative, name in TARGETS:
        model = serializer_model(args.vllm_source / relative, name)
        for kwargs in ({}, {"reasoning": None}, {"reasoning": ""}, {"reasoning": "Thought."}):
            for none, unset, alias, round_trip in itertools.product((False, True), repeat=4):
                options = dict(exclude_none=none, exclude_unset=unset,
                               by_alias=alias, round_trip=round_trip)
                response = model(content="Answer", **kwargs)
                data = response.model_dump(**options)
                wire = json.loads(response.model_dump_json(**options))
                assert data == wire
                assert_alias(data)
                assert data["content"] == "Answer"
                assert "tool_calls" not in data
                cases += 1

        # Explicit field filtering and nested serialization are significant to
        # clients that dump response envelopes or select content/tool fields.
        call = {"id": "call_fixture", "type": "function",
                "function": {"name": "fixture", "arguments": "{}"}}
        response = model(content="Answer", reasoning="Thought.", tool_calls=[call])
        for options in ({}, {"exclude": {"reasoning"}}, {"include": {"content", "tool_calls"}}):
            data = response.model_dump(**options)
            assert data == json.loads(response.model_dump_json(**options))
            assert_alias(data)
            assert data["tool_calls"] == [call]
            assert data["content"] == "Answer"
            cases += 1

        envelope = pydantic.create_model("Envelope", message=(model, ...))
        data = json.loads(envelope(message=response).model_dump_json())
        assert_alias(data["message"])
        assert data["message"]["tool_calls"] == [call]
        cases += 1

        unpatched = serializer_model(args.vllm_source / relative, name, remove_alias=True)
        try:
            assert_alias(unpatched(reasoning="Thought.").model_dump())
        except AssertionError:
            negative_controls += 1
        else:
            raise AssertionError("test did not detect the missing compatibility alias")

    print(json.dumps({"status": "PASS", "serialization_cases": cases,
                      "negative_controls_detected": negative_controls,
                      "pydantic_version": pydantic.__version__, "gpu_imports": False}))


if __name__ == "__main__":
    main()
