#!/usr/bin/env python3
"""Build independent CSIL and MoPC-HR base states from the full source set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256  # noqa: E402
from model_dual_cvsincnet import backbone_forward_compat  # noqa: E402
from paper_reproduction.common.wisig_runtime import (  # noqa: E402
    collate_wisig,
    load_wisig_compact_pkl,
)
from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import (  # noqa: E402
    build_csil_base_state,
    build_mopc_base_state,
)
from paper_reproduction.cvs_aligned.evaluate import _make_source_dataset  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    if _sha256(checkpoint_path) != BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint SHA drift")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exact, load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=256, device=device
    )
    backbone = exact.id_backbone.to(device)
    feature_key = str(getattr(exact, "id_feature_key", "feat_joint"))

    def feature_fn(model: torch.nn.Module, rows: torch.Tensor):
        auxiliary = backbone_forward_compat(
            model,
            rows,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        feature = auxiliary.get(feature_key)
        if not torch.is_tensor(feature):
            feature = auxiliary.get("feat_joint")
        logits = auxiliary.get("logits")
        if not torch.is_tensor(feature) or not torch.is_tensor(logits):
            raise ValueError("ADV3B02 identity feature surface drift")
        return feature.float(), logits.float()

    old_labels = [value for value in str(args.old_class_labels).split(",") if value]
    receivers = [
        value for value in str(args.source_receiver_labels).split(",") if value
    ]
    if len(old_labels) != 6 or len(receivers) != 7:
        raise ValueError("official base old-class/source-receiver lock drift")
    payload = load_wisig_compact_pkl(Path(args.manysig_pkl))
    config = {
        "target_old_tx_labels": old_labels,
        "source_receiver_labels": receivers,
        "source_days": [0, 1],
        "equalized": int(args.equalized),
        "source_train_samples_per_combo": int(args.samples_per_combo),
        "seed": int(args.seed),
    }
    dataset = _make_source_dataset(config, payload)
    loader = DataLoader(
        dataset,
        batch_size=int(args.load_batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_wisig,
    )
    tx_lookup = {
        str(label): index for index, label in enumerate(payload.get("tx_list", []))
    }
    compact = {
        tx_lookup[label]: index for index, label in enumerate(old_labels)
    }
    iq_rows = []
    label_rows = []
    for batch in loader:
        iq_rows.append(batch["iq"].float())
        label_rows.append(
            torch.tensor(
                [compact[int(value)] for value in batch["label"].tolist()],
                dtype=torch.long,
            )
        )
    source_x = torch.cat(iq_rows).to(device)
    source_y = torch.cat(label_rows).to(device)
    expected = (
        len(old_labels) * len(receivers) * 2 * int(args.samples_per_combo)
    )
    if len(source_y) != expected or expected != 8400:
        raise ValueError(
            f"official base requires exactly 8400 rows, got {len(source_y)}"
        )
    counts = torch.bincount(source_y, minlength=len(old_labels))
    if counts.tolist() != [1400] * len(old_labels):
        raise ValueError(f"source class coverage drift: {counts.tolist()}")

    csil = build_csil_base_state(
        backbone,
        source_x,
        source_y,
        feature_fn=feature_fn,
        old_count=len(old_labels),
        seed=int(args.seed),
    )
    mopc = build_mopc_base_state(
        backbone,
        source_x,
        source_y,
        feature_fn=feature_fn,
        old_count=len(old_labels),
        total_capacity=int(args.total_capacity),
        seed=int(args.seed),
    )
    state = {
        "schema": "cvs.adv3b02.official_repo_base_state.v2",
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "old_class_labels": old_labels,
        "source_receiver_labels": receivers,
        "source_days": [0, 1],
        "source_train_samples_per_combo": int(args.samples_per_combo),
        "base_sample_count": int(len(source_y)),
        "base_class_counts": counts.cpu().tolist(),
        "total_capacity": int(args.total_capacity),
        "csil": csil,
        "mopc_hr": mopc,
        "checkpoint_load_audit": load_audit,
        "official_repo_commits": {
            "csil": "8ce8637daf4dc60eeb1c56bff64c050c5b2353e9",
            "mopc_hr": "ae6554316ad1a2175920e330133a2f103408bf78",
        },
        "raw_exemplars_stored": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    receipt = {
        "schema": state["schema"],
        "status": "PASS",
        "output": str(output),
        "output_sha256": _sha256(output),
        "base_sample_count": state["base_sample_count"],
        "base_class_counts": state["base_class_counts"],
        "source_receiver_count": len(receivers),
        "source_day_count": 2,
        "total_capacity": int(args.total_capacity),
        "csil_optimizer_steps": int(csil["optimizer_steps"]),
        "mopc_optimizer_steps": int(mopc["optimizer_steps"]),
        "checkpoint_load_audit": load_audit,
    }
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manysig-pkl", type=Path, required=True)
    parser.add_argument("--old-class-labels", required=True)
    parser.add_argument("--source-receiver-labels", required=True)
    parser.add_argument("--samples-per-combo", type=int, default=100)
    parser.add_argument("--equalized", type=int, default=1)
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--load-batch-size", type=int, default=256)
    parser.add_argument("--total-capacity", type=int, default=26)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
