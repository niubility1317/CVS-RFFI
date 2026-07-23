#!/usr/bin/env python3
"""Precompute paper-method base fingerprints, prototypes, and CSIL Fisher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
from paper_reproduction.cvs_aligned.adv3b02_paper_full_ci import (  # noqa: E402
    _class_means,
    zero_bias_logits,
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
    backbone = exact.id_backbone.to(device).eval()
    feature_key = str(getattr(exact, "id_feature_key", "feat_joint"))

    def feature_fn(rows: torch.Tensor):
        auxiliary = backbone_forward_compat(
            backbone,
            rows,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        feature = auxiliary.get(feature_key)
        if not torch.is_tensor(feature):
            feature = auxiliary.get("feat_joint")
        if not torch.is_tensor(feature):
            raise ValueError("ADV3B02 base feature key drift")
        return feature.float()

    old_labels = [value for value in str(args.old_class_labels).split(",") if value]
    source_receivers = [
        value for value in str(args.source_receiver_labels).split(",") if value
    ]
    if len(old_labels) != 6 or not source_receivers:
        raise ValueError("base class/source receiver lock drift")
    dataset_payload = load_wisig_compact_pkl(Path(args.manysig_pkl))
    config = {
        "target_old_tx_labels": old_labels,
        "source_receiver_labels": source_receivers,
        "source_days": [0, 1],
        "equalized": int(args.equalized),
        "source_train_samples_per_combo": int(args.samples_per_combo),
        "seed": int(args.seed),
    }
    dataset = _make_source_dataset(config, dataset_payload)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_wisig,
    )
    tx_lookup = {
        str(label): index for index, label in enumerate(dataset_payload.get("tx_list", []))
    }
    source_ids = [tx_lookup[label] for label in old_labels]
    compact = {value: index for index, value in enumerate(source_ids)}

    feature_rows = []
    label_rows = []
    with torch.no_grad():
        for batch in loader:
            rows = batch["iq"].to(device)
            labels = torch.tensor(
                [compact[int(value)] for value in batch["label"].tolist()],
                dtype=torch.long,
                device=device,
            )
            feature_rows.append(feature_fn(rows))
            label_rows.append(labels)
    features = torch.cat(feature_rows)
    labels = torch.cat(label_rows)
    fingerprints = _class_means(features, labels, len(old_labels)).detach()

    fisher_sum = {
        name: torch.zeros_like(parameter)
        for name, parameter in backbone.named_parameters()
    }
    fisher_batches = 0
    for batch in loader:
        rows = batch["iq"].to(device)
        labels = torch.tensor(
            [compact[int(value)] for value in batch["label"].tolist()],
            dtype=torch.long,
            device=device,
        )
        backbone.zero_grad(set_to_none=True)
        loss = F.cross_entropy(zero_bias_logits(feature_fn(rows), fingerprints), labels)
        loss.backward()
        for name, parameter in backbone.named_parameters():
            if parameter.grad is not None:
                fisher_sum[name].add_(parameter.grad.detach().pow(2))
        fisher_batches += 1
    if fisher_batches <= 0:
        raise ValueError("base source loader produced no Fisher batches")
    fisher = {
        name: torch.exp((value / fisher_batches).clamp(max=20.0)).detach().cpu()
        if bool(torch.count_nonzero(value))
        else value.detach().cpu()
        for name, value in fisher_sum.items()
    }
    payload = {
        "schema": "cvs.adv3b02.paper_full_base_state.v1",
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "old_class_labels": old_labels,
        "source_receiver_labels": source_receivers,
        "source_days": [0, 1],
        "source_train_samples_per_combo": int(args.samples_per_combo),
        "base_sample_count": int(len(labels)),
        "old_fingerprints": fingerprints.detach().cpu(),
        "old_prototypes": fingerprints.detach().cpu(),
        "fisher": fisher,
        "checkpoint_load_audit": load_audit,
        "raw_exemplars_stored": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    receipt = {
        "schema": payload["schema"],
        "status": "PASS",
        "output": str(output),
        "output_sha256": _sha256(output),
        "base_sample_count": payload["base_sample_count"],
        "old_class_count": len(old_labels),
        "feature_dim": int(fingerprints.shape[1]),
        "fisher_tensor_count": len(fisher),
        "raw_exemplars_stored": False,
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
    parser.add_argument("--source-train-samples-per-combo", dest="samples_per_combo", type=int, default=100)
    parser.add_argument("--equalized", type=int, default=1)
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
