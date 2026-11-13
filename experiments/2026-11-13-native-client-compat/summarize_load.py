"""Summarize the fixed-resident trial with measured token counts and no bodies."""
import argparse
import json
import math
import statistics
from pathlib import Path


def percentiles(values):
    values = sorted(values)
    if not values:
        return {}
    summary = {}
    for p in (50, 95, 99):
        pos = (len(values)-1) * p / 100
        frac = pos - int(pos)
        summary[f"p{p}"] = round(values[int(pos)]*(1-frac) + values[math.ceil(pos)]*frac, 4)
    return summary


def summarize(cell):
    rows = cell["resident_decoders"]
    injection = cell["injected"]
    timing = injection["server_timing"]
    rates = [r["client_decode_tps"] for r in rows]
    result = {
        "cell":cell["label"], "resident_count":len(rows), "input_tokens":injection["prompt_tokens"],
        "cached_tokens":injection["cached_tokens"], "injected_checks":cell["injected_checks"],
        "text_size_matched":cell.get("size_matched"), "injected_queue_ms":timing.get("vllm.queue_time_ms"),
        "scheduled_to_first_ms":timing.get("vllm.time_to_first_token_ms"),
        "injected_client_ttft_s":injection["ttft_s"], "max_resident_gap_s":cell["resident_max_gap_s"],
        "residents_emitting_after_injected_first":sum(r["emitted_after_injected_first_output"] for r in rows),
        "all_resident_answers_exact":all(r["sequence_exact"] for r in rows),
        "all_resident_output_hashes_match_solo":cell["same_resident_bytes"],
        "resident_overall_decode_tps":percentiles(rates),
        "resident_rate_jain_index":sum(rates)**2/(len(rates)*sum(r*r for r in rates)),
        "resident_ttft_s":percentiles([r["ttft_s"] for r in rows]),
        "resident_time_to_completion_s":percentiles([r["e2e_s"] for r in rows]),
        "resident_median_visible_digits_per_second":{},
        "gap_gate_s":cell["max_decoder_gap_gate_s"], "resident_gate_failures":cell["failed_assertions"]
    }
    for period in ("before", "during", "recovery"):
        digits = [r["visible_digit_rates"][period]["digits_per_second"] for r in rows]
        valid = [x for x in digits if x is not None]
        result["resident_median_visible_digits_per_second"][period] = statistics.median(valid) if valid else None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data = json.loads(args.receipt.read_text())
    out = {"run_status":data["status"], "reason":data.get("reason"), "current_cell":data.get("current_cell"),
           "scope":"Single wave per cell; percentile samples are residents in that wave.",
           "cells":[summarize(cell) for cell in data["cells"]]}
    print(json.dumps(out, indent=2, allow_nan=False))
