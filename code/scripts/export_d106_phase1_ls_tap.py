#!/usr/bin/env python3
"""Build, export, or validate the frozen D106 Phase1 ``L_s`` strict tap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d106_phase1_tap import (  # noqa: E402
    build_d106_train_held_disjoint_receipt,
    extract_d106_ls_received_iq,
    export_d106_phase1_ls_tap,
    load_d106_phase1_ls_tap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen D106 builder-only L_s IQ join and strict tap"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    disjoint = commands.add_parser(
        "build-disjoint-receipt",
        help="independently recompute L_s+U_s versus source-held ID disjointness",
    )
    disjoint.add_argument("--source-split-manifest", type=Path, required=True)
    disjoint.add_argument("--source-split-manifest-sha256", required=True)
    disjoint.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser(
        "extract", help="validate 8400x3 storage and seal only selected 588 L_s IQ rows"
    )
    extract.add_argument("--source-split-manifest", type=Path, required=True)
    extract.add_argument("--source-split-manifest-sha256", required=True)
    extract.add_argument("--disjoint-receipt", type=Path, required=True)
    extract.add_argument("--disjoint-receipt-sha256", required=True)
    extract.add_argument("--source-train-cache-set", type=Path, required=True)
    extract.add_argument("--selection-salt-receipt", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)

    export = commands.add_parser(
        "export", help="join exact L_s IQ and publish the non-IQ strict tap"
    )
    export.add_argument("--selected-iq-archive", type=Path, required=True)
    export.add_argument("--selected-iq-archive-sha256", required=True)
    export.add_argument("--selected-iq-receipt", type=Path, required=True)
    export.add_argument("--selected-iq-receipt-sha256", required=True)
    export.add_argument("--storage-validator-receipt", type=Path, required=True)
    export.add_argument("--storage-validator-receipt-sha256", required=True)
    export.add_argument("--ls-archive", type=Path, required=True)
    export.add_argument("--ls-archive-sha256", required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--checkpoint-sha256", required=True)
    export.add_argument("--runtime-manifest", type=Path, required=True)
    export.add_argument("--runtime-sha256", required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--device", default="cpu")

    validate = commands.add_parser(
        "validate", help="strictly load a sealed D106 tap and derive z_id"
    )
    validate.add_argument("--archive", type=Path, required=True)
    validate.add_argument("--archive-sha256", required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-disjoint-receipt":
        result = build_d106_train_held_disjoint_receipt(
            source_split_manifest=args.source_split_manifest,
            source_split_manifest_sha256=args.source_split_manifest_sha256,
            output_path=args.output,
        )
    elif args.command == "extract":
        result = extract_d106_ls_received_iq(
            source_split_manifest=args.source_split_manifest,
            source_split_manifest_sha256=args.source_split_manifest_sha256,
            disjoint_receipt=args.disjoint_receipt,
            disjoint_receipt_sha256=args.disjoint_receipt_sha256,
            source_train_cache_set=args.source_train_cache_set,
            selection_salt_receipt=args.selection_salt_receipt,
            output_dir=args.output_dir,
        )
    elif args.command == "export":
        result = export_d106_phase1_ls_tap(
            selected_iq_archive=args.selected_iq_archive,
            selected_iq_archive_sha256=args.selected_iq_archive_sha256,
            selected_iq_receipt=args.selected_iq_receipt,
            selected_iq_receipt_sha256=args.selected_iq_receipt_sha256,
            storage_validator_receipt=args.storage_validator_receipt,
            storage_validator_receipt_sha256=(
                args.storage_validator_receipt_sha256
            ),
            ls_archive=args.ls_archive,
            ls_archive_sha256=args.ls_archive_sha256,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            runtime_manifest=args.runtime_manifest,
            runtime_sha256=args.runtime_sha256,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        loaded = load_d106_phase1_ls_tap(
            args.archive,
            args.receipt,
            expected_archive_sha256=args.archive_sha256,
            expected_receipt_sha256=args.receipt_sha256,
        )
        result = {
            "status": "D106_LS_STRICT_TAP_VALID",
            "row_count": int(len(loaded.z_id)),
            "z_id_derived_from_pre_relu": True,
            "received_iq_persisted": False,
        }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
