#!/usr/bin/env python3
"""Generate and independently score FastTrust ``V_select-as-U`` artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG import train_ssdg
from cvsrffi.phase1_pseudolabel_quality import (
    build_truth_blind_records,
    score_truth_blind_records,
)
from cvsrffi.muse_ssdg import route_fasttrust_rc4


def _checkpoint_args(checkpoint, output_dir: Path, device: str):
    defaults = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", str(output_dir), "--device", str(device)]
    )
    stored = checkpoint.get("args")
    if not isinstance(stored, dict):
        raise ValueError("checkpoint args are required for V_select reconstruction")
    for key, value in stored.items():
        setattr(defaults, str(key), value)
    defaults.output_dir = str(output_dir)
    defaults.device = str(device)
    return defaults


def _build_models(checkpoint, args, data_ctx, device):
    baseline_args = checkpoint.get("baseline_args")
    if not isinstance(baseline_args, dict):
        raise ValueError("checkpoint baseline_args are required")
    model_args = SimpleNamespace(**baseline_args)
    model = train_ssdg.build_baseline_model(model_args, device)
    train_ssdg._load_phase1_checkpoint_strict(model, checkpoint, "V_select checkpoint")
    model.eval()

    ema_state = checkpoint.get("ema_model")
    if ema_state is None:
        raise ValueError("checkpoint ema_model is required for RC4 quality audit")
    ema_model = deepcopy(model).to(device)
    ema_model.load_state_dict(ema_state, strict=True)
    ema_model.eval()

    if bool(args.rc4_use_anchor):
        anchor_path = Path(str(args.teacher_ckpt))
        if not anchor_path.is_absolute():
            candidate = Path(str(args.baseline_ckpt))
            anchor_path = candidate if candidate.is_absolute() else anchor_path
        anchor_checkpoint = train_ssdg.load_checkpoint(str(anchor_path), device)
        anchor_args = train_ssdg.merge_checkpoint_args(
            anchor_checkpoint,
            argparse.Namespace(),
            input_len=int(data_ctx["input_len"]),
            num_domains=int(data_ctx["num_domains"]),
        )
        anchor_args.id_feature_key = str(getattr(args, "id_feature_key", "feat_joint"))
        anchor_model = train_ssdg.build_baseline_model(anchor_args, device)
        anchor_model.load_state_dict(anchor_checkpoint["model"], strict=False)
        anchor_model.eval()
    else:
        anchor_model = ema_model
    return model, ema_model, anchor_model


def _route_names(route):
    names = []
    for index in range(int(route.hard.numel())):
        if bool(route.hard[index]):
            names.append("H")
        elif bool(route.partial[index]):
            names.append("P")
        elif bool(route.negative[index]):
            names.append("N")
        else:
            names.append("R")
    return names


def _values(extra, key, count):
    value = extra.get(key)
    if torch.is_tensor(value):
        result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        result = list(value)
    else:
        result = [value] * count
    if len(result) != count:
        raise ValueError(f"metadata {key} length does not match batch")
    return result


def _metadata_tensor(metadata, key: str, count: int, device):
    """Read an already-unpacked truth-hidden metadata field as a label tensor."""

    return torch.as_tensor(_values(metadata, key, count), device=device, dtype=torch.long)


def _truth_hidden_inputs(batch, device):
    x_value, domain_payload = train_ssdg._move_muse_unlabeled_batch(batch, device)
    domains, metadata = domain_payload
    if not torch.is_tensor(domains):
        domains = torch.as_tensor(domains, device=device)
    return x_value, domains.to(device=device, dtype=torch.long), metadata


def generate(args) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    artifact_path = Path(args.artifact_out).resolve()
    if artifact_path.exists():
        raise FileExistsError("refusing to overwrite V_select artifact")
    checkpoint = train_ssdg.load_checkpoint(str(checkpoint_path), torch.device(args.device))
    work_args = _checkpoint_args(checkpoint, artifact_path.parent, args.device)
    train_ssdg.set_seed(int(work_args.seed))
    device = train_ssdg.resolve_device(work_args.device)
    data_ctx = train_ssdg._build_ssdg_wisig_data(work_args, device)
    _, ema_model, anchor_model = _build_models(checkpoint, work_args, data_ctx, device)
    calibration = checkpoint.get("rc4_calibration")
    if calibration is None:
        raise ValueError("checkpoint rc4_calibration is required")

    records = []
    truth_hidden_dataset = train_ssdg._MUSEUnlabeledDatasetView(
        data_ctx["val_loader"].dataset
    )
    audit_loader = train_ssdg.make_loader(
        truth_hidden_dataset,
        train_ssdg._resolve_unlabeled_batch_size(work_args),
        False,
        int(work_args.num_workers),
        device,
        False,
        int(work_args.prefetch_factor),
    )
    with torch.no_grad():
        for batch in audit_loader:
            x_select, domains, extra = _truth_hidden_inputs(batch, device)
            count = int(x_select.shape[0])
            receivers = _metadata_tensor(extra, "rx_i", count, device)
            weak_2 = train_ssdg._strong_augment(
                x_select, max(1e-5, float(work_args.strong_noise_std) * 0.25)
            )
            combined = torch.cat([x_select, weak_2], dim=0)
            combined_domains = torch.cat([domains, domains], dim=0)
            ema_output = ema_model(
                combined,
                y_tx=None,
                grl_lambda=0.0,
                return_aux=True,
                domain_labels=combined_domains,
            )
            ema_1, ema_2 = train_ssdg._split_muse_output(
                ema_output, count, int(combined.shape[0])
            )
            anchor_output = anchor_model(
                x_select,
                y_tx=None,
                grl_lambda=0.0,
                return_aux=True,
                domain_labels=domains,
            )
            route = route_fasttrust_rc4(
                anchor_output["tx_logits"],
                ema_1["tx_logits"],
                ema_2["tx_logits"],
                domains=domains,
                receivers=receivers,
                z_norm=ema_1["z_id"].float().norm(dim=-1),
                calibration=calibration,
                hard_max_fraction=float(work_args.sat_anchor_hard_max_fraction),
                hard_effective_budget=float(work_args.rc4_hard_effective_budget),
                candidate_max_classes=int(work_args.muse_candidate_max_classes),
                partial_effective_budget=float(work_args.rc4_partial_effective_budget),
                negative_effective_budget=float(work_args.rc4_negative_effective_budget),
                total_identity_effective_budget=float(work_args.rc4_total_identity_effective_budget),
                use_calibrated_partial_threshold=bool(work_args.rc4_use_calibrated_partial_threshold),
                enable_hard=bool(work_args.rc4_enable_hard),
                enable_partial=bool(work_args.rc4_enable_partial),
                enable_negative=bool(work_args.rc4_enable_negative),
                class_receiver_cap=bool(work_args.rc4_class_receiver_cap),
                class_receiver_effective_budget=float(work_args.rc4_class_receiver_effective_budget),
                use_calibrated_risk=bool(work_args.rc4_use_correctness_calibration),
            )
            base_indices = _values(extra, "base_index", count)
            sample_ids = [f"v_select:{int(value)}" for value in base_indices]
            batch_records = build_truth_blind_records(
                physical_sample_ids=sample_ids,
                receivers=_values(extra, "rx_i", count),
                days=_values(extra, "day_i", count),
                routes=_route_names(route),
                pseudo_labels=route.pseudo.detach().cpu().tolist(),
                candidate_masks=route.candidate_mask,
                fused_probabilities=route.fused_probability,
                p_correct=route.p_correct.detach().cpu().tolist(),
                p_set_safe=route.p_set_safe.detach().cpu().tolist(),
                sample_weights=route.weights.detach().cpu().tolist(),
            )
            records.extend(batch_records)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"artifact_rows": len(records), "truth_access": False}, sort_keys=True))
    return 0


def extract_truth(args) -> int:
    """Export source-side truth in a process separate from artifact generation."""

    checkpoint_path = Path(args.checkpoint).resolve()
    truth_path = Path(args.truth_out).resolve()
    if truth_path.exists():
        raise FileExistsError("refusing to overwrite V_select truth sidecar")
    checkpoint = train_ssdg.load_checkpoint(str(checkpoint_path), torch.device(args.device))
    work_args = _checkpoint_args(checkpoint, truth_path.parent, args.device)
    device = train_ssdg.resolve_device(work_args.device)
    data_ctx = train_ssdg._build_ssdg_wisig_data(work_args, device)
    truth = {}
    for batch in data_ctx["val_loader"]:
        _, labels, extra = train_ssdg.move_batch(batch, device)
        count = int(labels.numel())
        base_indices = _values(extra, "base_index", count)
        sample_ids = [f"v_select:{int(value)}" for value in base_indices]
        for sample_id, label in zip(sample_ids, labels.detach().cpu().tolist()):
            truth[sample_id] = int(label)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"truth_rows": len(truth)}, sort_keys=True))
    return 0


def score(args) -> int:
    artifact_path = Path(args.artifact).resolve()
    truth_path = Path(args.truth).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite V_select quality score")
    records = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines() if line]
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    summary = score_truth_blind_records(records, truth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--artifact-out", required=True)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.set_defaults(func=generate)
    truth_parser = subparsers.add_parser("extract-truth")
    truth_parser.add_argument("--checkpoint", required=True)
    truth_parser.add_argument("--truth-out", required=True)
    truth_parser.add_argument("--device", default="cuda")
    truth_parser.set_defaults(func=extract_truth)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--artifact", required=True)
    score_parser.add_argument("--truth", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.set_defaults(func=score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
