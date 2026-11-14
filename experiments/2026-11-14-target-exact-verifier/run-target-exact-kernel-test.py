#!/usr/bin/env python3
"""Dependency-free GPU checks for the target-exact commit kernel."""

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


def reference(
    target: torch.Tensor,
    drafts: torch.Tensor,
    offsets: list[int],
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = torch.empty((len(offsets) - 1, width), dtype=torch.int64)
    counts = torch.empty(len(offsets) - 1, dtype=torch.int32)
    target = target.cpu()
    drafts = drafts.cpu()
    for request, (start, end) in enumerate(zip(offsets, offsets[1:])):
        num_drafts = end - start - 1
        accepted = 0
        for step in range(num_drafts):
            draft = int(drafts[start + step + 1])
            expected = int(target[start + step])
            if draft >= 0 and draft == expected:
                sampled[request, step] = draft
                accepted += 1
            else:
                sampled[request, step] = expected
                break
        if accepted == num_drafts:
            sampled[request, num_drafts] = target[start + num_drafts]
        counts[request] = accepted + 1
    return sampled, counts


def check_commit_kernel(draft_dtype: torch.dtype) -> None:
    drafts_per_request = [5, 3, 0, 5, 1, 4]
    accepted_per_request = [5, 1, 0, 2, 0, 3]
    offsets = [0]
    for count in drafts_per_request:
        offsets.append(offsets[-1] + count + 1)

    target = torch.arange(offsets[-1], dtype=torch.int64, device="cuda") + 100
    drafts = torch.full(
        (offsets[-1],), -777, dtype=draft_dtype, device="cuda"
    )
    for request, (count, accepted) in enumerate(
        zip(drafts_per_request, accepted_per_request)
    ):
        start = offsets[request]
        for step in range(count):
            if step < accepted:
                drafts[start + step + 1] = target[start + step]
            elif step == accepted:
                drafts[start + step + 1] = (
                    -1 if request % 2 else target[start + step] + 1
                )
            else:
                drafts[start + step + 1] = target[start + step]

    cu_num_logits = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    sampled, counts = tilelang_sampling.target_exact_rejection_sample(
        target, drafts, cu_num_logits, num_speculative_steps=5
    )
    torch.cuda.synchronize()
    expected, expected_counts = reference(target, drafts, offsets, width=6)
    assert torch.equal(counts.cpu(), expected_counts)
    for request, count in enumerate(expected_counts.tolist()):
        assert torch.equal(
            sampled[request, :count].cpu(), expected[request, :count]
        )


def check_declarations() -> None:
    owner = RejectionSampler.__new__(RejectionSampler)
    owner.sampler = SimpleNamespace(
        use_fp64_gumbel=False,
        logprobs_mode="raw_logprobs",
    )
    owner.num_speculative_steps = 9
    owner.num_draft_steps = 5
    owner.watermark_key = None
    owner.use_block_verification = False
    owner.use_target_exact_verification = True
    owner.synthetic_conditional_rates = None

    registry = JitWarmupRegistry(None)
    with registry.activate():
        owner.register_warmups(
            logits=[(torch.bfloat16, 129408)],
            processed_logits=[(torch.float32, 129280)],
            vocab_size=129280,
            has_draft_logits=False,
            dead_rows=True,
        )
    calls = prepare.declared_calls(registry)
    declared = {
        args
        for factory, args, _ in calls
        if factory is tilelang_sampling.target_exact_rejection_kernel
    }
    assert declared == {
        (6, "int64", "int32"),
        (6, "int64", "int64"),
        (10, "int64", "int32"),
        (10, "int64", "int64"),
    }


def main() -> None:
    assert torch.cuda.is_available()
    check_commit_kernel(torch.int32)
    check_commit_kernel(torch.int64)
    check_declarations()
    print("PASS target-exact commit kernel: int32/int64 drafts and warmup declarations")


if __name__ == "__main__":
    main()
