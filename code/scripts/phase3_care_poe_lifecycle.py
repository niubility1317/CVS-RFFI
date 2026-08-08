#!/usr/bin/env python
"""Execute the fail-closed anonymous-to-fresh-K G0 lifecycle fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase3_care_poe import (
    authorize_registration,
    build_fresh_k_bridge,
    canonical_json,
    create_anonymous_entity,
    read_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--credential-template", required=True)
    parser.add_argument("--fresh-support", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    candidates = [
        row for row in predictions
        if row.get("arm") == "D" and row.get("node_budget") == 5 and row.get("decision") == "unknown"
    ]
    if len(candidates) != 1:
        raise RuntimeError("fixture must produce exactly one D:N5 unknown decision")
    anonymous = create_anonymous_entity(candidates[0])
    credential = json.loads(Path(args.credential_template).read_text(encoding="utf-8"))
    credential["anonymous_entity_id"] = anonymous["anonymous_entity_id"]
    authorization = authorize_registration(anonymous, credential, now_ms=2_000.0)
    bridge = build_fresh_k_bridge(authorization, read_jsonl(args.fresh_support), k=args.k)
    receipt = {
        "evidence_level": "TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT",
        "anonymous": anonymous,
        "authorization": authorization,
        "fresh_k_bridge": bridge,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

