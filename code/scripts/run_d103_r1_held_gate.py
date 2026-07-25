#!/usr/bin/env python3
"""Evaluate a complete D103-R1 source-held receipt package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_held_falsifier import (  # noqa: E402
    canonical_json_sha256,
    evaluate_complete_gate,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-sha256", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipts_path = args.receipts_json.resolve()
    receipts = _read_json(receipts_path)
    expected_keys = {
        "receiver_ids",
        "class_ids",
        "performance_rows",
        "day_stability_rows",
        "d102_provenance",
        "tx_probe_rows",
        "quantization_receipt",
        "resource_receipt",
        "access_receipt",
    }
    if set(receipts) != expected_keys:
        raise ValueError("held receipt package key closure drift")
    result = evaluate_complete_gate(
        receiver_ids=receipts["receiver_ids"],
        class_ids=receipts["class_ids"],
        performance_rows=receipts["performance_rows"],
        day_stability_rows=receipts["day_stability_rows"],
        d102_provenance=receipts["d102_provenance"],
        tx_probe_rows=receipts["tx_probe_rows"],
        quantization_receipt=receipts["quantization_receipt"],
        resource_receipt=receipts["resource_receipt"],
        access_receipt=receipts["access_receipt"],
    )
    result["input_receipts_sha256"] = hashlib.sha256(receipts_path.read_bytes()).hexdigest()
    result["analysis_sha256"] = canonical_json_sha256(result)
    encoded = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    output_path = args.output_json.resolve()
    _write_new(output_path, encoded)
    _write_new(
        args.output_sha256.resolve(),
        (hashlib.sha256(encoded).hexdigest() + "  " + output_path.name + "\n").encode("ascii"),
    )
    print(result["status"])
    # A complete scientific rejection is a valid analyzed result, not a
    # technical process failure. Invalid/incomplete receipts raise above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
