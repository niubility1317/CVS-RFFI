#!/usr/bin/env python
"""Normalize optimizer post-run summaries into optimizer_batch_summary_v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimizer_workflow_lib import load_json_compat, standardize_summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path, help="Existing post-run summary JSON to normalize.")
    parser.add_argument("--output", type=Path, help="Destination JSON path. Defaults next to input.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = load_json_compat(args.summary_json)
    payload = standardize_summary(root, args.summary_json)
    output = args.output or args.summary_json.with_name("optimizer_batch_summary_v1.json")
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "schema": payload["schema"],
                "batch": payload["batch"],
                "lane": payload["lane"],
                "candidate_count": payload["candidate_count"],
                "batch_best_views": sorted(payload["batch_bests"].keys()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
