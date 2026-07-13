from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_reproduction.common.config import contains_unresolved_placeholder, contains_unspecified, load_json_config
from paper_reproduction.CSIL.protocol import PAPER_TITLE, validate_paper_faithful_config


def build_dry_run_payload(config: dict) -> dict:
    checked = validate_paper_faithful_config(config)
    return {
        "method_id": "csil_class_incremental_iot",
        "paper": PAPER_TITLE,
        "algorithm": "CSIL zero-bias cosine classifier with channel separation, KD, and EWC",
        "claim_boundary": checked["claim_boundary"],
        "not_cvs_stage2": True,
        "not_satellite_deployment_evidence": True,
        "dataset": "ADS-B",
        "adsb_protocol": {
            "receiver": "USRP B210",
            "carrier_mhz": 1090,
            "sample_rate_mhz": 8,
            "complex_samples_per_message": 1024,
            "class_filter": "top 100 transponders with at least 500 samples",
            "feature_tensor": "32x32x3 residual channels",
            "train_validation_split": "60/40",
        },
        "stage_plan": checked["stage_plan"],
        "paper_reported_hyperparameters": checked["paper_reported_hyperparameters"],
        "paper_unspecified_fields": checked["paper_unspecified_fields"],
        "evidence_targets": {
            "Fig.7": "DoC by incremental stage",
            "Fig.8": "new/old/overall accuracy by stage",
            "Fig.9": "ablation new/old/overall accuracy by stage",
            "TableI": "fingerprint conflict diagnostic before/after IL",
            "TableII": "CSIL ablation table",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CSIL ADS-B paper-faithful reproduction entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.formal and contains_unspecified(config):
        raise ValueError("formal CSIL config still contains paper-unspecified")
    if args.formal and contains_unresolved_placeholder(config):
        raise ValueError("formal CSIL config still contains unresolved placeholder")
    if args.dry_run:
        print(json.dumps(build_dry_run_payload(config), ensure_ascii=False, sort_keys=True))
        return 0
    raise SystemExit("full ADS-B training requires a local features artifact; use --dry-run for protocol validation")


if __name__ == "__main__":
    raise SystemExit(main())

