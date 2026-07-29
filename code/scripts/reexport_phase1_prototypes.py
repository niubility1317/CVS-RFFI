#!/usr/bin/env python3
"""Re-export Phase1 prototypes from one completed source-val checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from SSDG import train_ssdg


class Phase1ReexportError(RuntimeError):
    """Raised when checkpoint-only prototype publication is incomplete."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def reexport(args: argparse.Namespace) -> dict:
    checkpoint_path = Path(args.checkpoint).resolve()
    wisig_path = Path(args.wisig_pkl).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise Phase1ReexportError("source checkpoint is absent or empty")
    if not wisig_path.is_file():
        raise Phase1ReexportError("WiSig pickle is missing")
    if output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite an existing reexport output directory"
        )
    output_dir.mkdir(parents=True)
    device = train_ssdg.resolve_device(args.device)
    train_ssdg._prepare_cuda_memory_audit(device)
    checkpoint = train_ssdg.load_checkpoint(
        str(checkpoint_path),
        device,
    )
    saved_args = dict(checkpoint.get("args") or {})
    data_args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", str(output_dir)]
    )
    for key, value in saved_args.items():
        setattr(data_args, key, value)
    data_args.wisig_pkl = str(wisig_path)
    data_args.device = str(device)
    data_args.num_workers = int(args.num_workers)
    data_args.phase2_export_prototypes = True
    data_args.phase2_export_checkpoint = str(checkpoint_path)
    data_args.phase2_export_path = str(
        output_dir / "phase2_zid_prototypes.pt"
    )
    train_ssdg.set_seed(int(getattr(data_args, "seed", 0)))
    data_ctx = train_ssdg._build_ssdg_wisig_data(data_args, device)
    model_args = train_ssdg.merge_checkpoint_args(
        checkpoint,
        data_args,
        input_len=int(data_ctx["input_len"]),
        num_domains=int(data_ctx["num_domains"]),
    )
    model_args = train_ssdg._apply_model_cli_args(
        model_args,
        data_args,
    )
    model = train_ssdg.build_baseline_model(model_args, device)
    train_ssdg._load_phase1_checkpoint_strict(
        model,
        checkpoint,
        checkpoint_path,
    )
    model.eval()
    package = train_ssdg._maybe_export_phase2_prototypes_ssdg(
        data_args,
        model,
        data_ctx,
        device,
        default_checkpoint=checkpoint_path,
    )
    if not isinstance(package, dict):
        raise Phase1ReexportError("prototype exporter returned no package")
    prototype_path = output_dir / "phase2_zid_prototypes.pt"
    prototype_json_path = output_dir / "phase2_zid_prototypes.json"
    if any(
        not path.is_file() or path.stat().st_size <= 0
        for path in (prototype_path, prototype_json_path)
    ):
        raise Phase1ReexportError(
            "prototype exporter left absent or empty artifacts"
        )
    json.loads(prototype_json_path.read_text(encoding="utf-8-sig"))
    receipt = {
        "schema": "cvs.phase1.prototype_reexport_receipt.v1",
        "status": "COMPLETE",
        "exit_code": 0,
        "row_key": str(args.row_key),
        "source_run_id": str(args.source_run_id),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256_path(checkpoint_path),
        "exporter_git_commit": str(args.exporter_git_commit),
        "prototype_paths": {
            "prototype_path": str(prototype_path),
            "prototype_json_path": str(prototype_json_path),
        },
        "prototype_hashes": {
            "prototype_path": _sha256_path(prototype_path),
            "prototype_json_path": _sha256_path(
                prototype_json_path
            ),
        },
    }
    receipt_path = output_dir / "phase1_reexport_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            receipt,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--row-key", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--exporter-git-commit", required=True)
    return parser


def main() -> int:
    receipt = reexport(_parser().parse_args())
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
