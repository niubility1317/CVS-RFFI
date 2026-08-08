#!/usr/bin/env python
"""Bind pre-label capture receipts to truth-free Phase3 local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase3_care_poe import (
    bind_verified_physical_evidence,
    canonical_json,
    event_key,
    physical_binding_root,
    read_jsonl,
    sha256_json,
    write_jsonl,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-evidence", required=True)
    parser.add_argument("--binding-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_evidence)
    binding_path = Path(args.binding_jsonl)
    output_path = Path(args.output_jsonl)
    receipt_path = Path(args.receipt_out)
    for target in (output_path, receipt_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    evidence = read_jsonl(input_path)
    bindings = read_jsonl(binding_path)
    rebound = bind_verified_physical_evidence(evidence, bindings)
    write_jsonl(output_path, rebound)
    receipt = {
        "schema": "cvs.phase3.physical_evidence_binding_receipt.v1",
        "evidence_level": "VERIFIED_PHYSICAL_BINDING_INTERFACE_NO_PERFORMANCE_RESULT",
        "truth_or_role_opened": False,
        "input_evidence_rows": len(evidence),
        "binding_rows": len(bindings),
        "output_rows": len(rebound),
        "event_count": len({event_key(record) for record in rebound}),
        "node_count": len({record["node_id"] for record in rebound}),
        "binding_receipt_count": len({record["physical_binding_receipt_id"] for record in rebound}),
        "input_evidence_sha256": _sha256_file(input_path),
        "binding_jsonl_sha256": _sha256_file(binding_path),
        "physical_binding_root": physical_binding_root(bindings),
        "output_jsonl_sha256": _sha256_file(output_path),
        "output_evidence_root": sha256_json(rebound),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(receipt) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
