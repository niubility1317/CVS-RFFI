"""Build one real source-only D127 Phase1 quantized candidate bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from cvsrffi.stage2_d127_phase1_release import (
    CANDIDATE_IDS,
    build_d127_phase1_single_candidate_from_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one frozen D127 Phase1 A/B/C quantized asset bundle from source-only inputs."
    )
    parser.add_argument("--candidate-id", required=True, choices=CANDIDATE_IDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-lock", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--selected-iq-archive", required=True)
    parser.add_argument("--selected-iq-archive-sha256", required=True)
    parser.add_argument("--selected-iq-receipt", required=True)
    parser.add_argument("--selected-iq-receipt-sha256", required=True)
    parser.add_argument("--ls-label-join-archive", required=True)
    parser.add_argument("--ls-label-join-archive-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_d127_phase1_single_candidate_from_source(
        candidate_id=args.candidate_id,
        output_dir=args.output_dir,
        method_lock_path=args.method_lock,
        method_lock_sha256=args.method_lock_sha256,
        selected_iq_archive=args.selected_iq_archive,
        selected_iq_archive_sha256=args.selected_iq_archive_sha256,
        selected_iq_receipt=args.selected_iq_receipt,
        selected_iq_receipt_sha256=args.selected_iq_receipt_sha256,
        ls_label_join_archive=args.ls_label_join_archive,
        ls_label_join_archive_sha256=args.ls_label_join_archive_sha256,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry.
    raise SystemExit(main())
