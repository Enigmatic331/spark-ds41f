"""Check the synthetic archive's output fields against recorded generated IDs.

Site-authored, AI-assisted diagnostic. Reads records locally and prints counts
only. Uses the archived checkpoint vocabulary with its byte-level decoder.
"""

import argparse
import hashlib
import json
from pathlib import Path


def inspect(tokenizer, receipt):
    raw = tokenizer.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b":
        raise ValueError("unexpected checkpoint vocabulary")
    table = json.loads(raw)
    vocab = {i: token for token, i in table["model"]["vocab"].items()}
    special = {t["content"]: t["id"] for t in table["added_tokens"]}
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    raw_bytes = {chr(i): i for i in direct}
    missing = [i for i in range(256) if i not in direct]
    raw_bytes.update({chr(256 + n): i for n, i in enumerate(missing)})

    def decode(ids):
        return bytes(raw_bytes[ch] for i in ids for ch in vocab[i]).decode()

    data = json.loads(receipt.read_bytes())
    if data.get("synthetic_fixture") is not True or data.get("image_sha256") != [
        "b2c7109fa155eecd1c96c78bc78a3d233ecae45f1b3dae1d16fb62412de6dc64"
    ]:
        raise ValueError("expected the archived synthetic debug fixture")
    for row in data["rows"]:
        ids = row["output_token_ids"]
        stop = special["</think>"]
        if ids.count(stop) != 1 or ids[-1] != 1 or row["finish_reason"] != "stop":
            raise ValueError("unexpected generated boundaries")
        boundary = ids.index(stop)
        reasoning = decode(ids[:boundary])
        content = decode(ids[boundary + 1 : -1])
        if hashlib.sha256(reasoning.encode()).hexdigest() != row["reasoning_sha256"]:
            raise ValueError("reasoning hash mismatch")
        if content != row["synthetic_visible_content"]:
            raise ValueError("visible content differs from generated answer")
        yield {
            "repeat": row["repeat"],
            "reasoning_mentions": reasoning.lower().count("value column clipped"),
            "content_mentions": content.lower().count("value column clipped"),
            "archived_pass": row["original_archived_score_passed"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.receipts:
        print(json.dumps(list(inspect(args.tokenizer, path))))
