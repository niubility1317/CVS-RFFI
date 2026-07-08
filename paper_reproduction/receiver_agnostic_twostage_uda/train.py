from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import write_json
from paper_reproduction.receiver_agnostic_twostage_uda.protocol import (
    build_receiver_ratio_plan,
    validate_paper_faithful_config,
)


def build_dry_run_payload(config: dict) -> dict:
    checked = validate_paper_faithful_config(config)
    return {
        "artifact_type": "dry_run_only",
        "schema_version": 2,
        "paper": "Receiver-Agnostic Radio Frequency Fingerprinting Based on Two-stage Unsupervised Domain Adaptation and Fine-tuning",
        "scope": checked["claim_boundary"],
        "dataset": checked["dataset"],
        "formal_training_status": "blocked",
        "result_claim_status": "no_formal_metrics",
        "synthetic_smoke_allowed": True,
        "requires_real_manysig_for_fig7_table_i_fig8": True,
        "receiver_ratio_plan": build_receiver_ratio_plan(checked),
        "paper_unspecified_fields": checked["paper_unspecified_fields"],
        "preprocessing": checked["preprocessing"],
        "claim_blocks": [
            "not CVS Stage2-C",
            "not satellite/LEO deployment evidence",
            "not open-set or new-class registration",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful dry-run entrypoint for Bao et al. two-stage UDA RFFI.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the reproduction matrix.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for dry-run payload.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if not args.dry_run:
        raise SystemExit("formal training is intentionally gated; run --dry-run first and fill paper-unspecified fields")
    payload = build_dry_run_payload(config)
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
