from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.nmfdu_training import NMFDUStageController
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl
from post_stage_common import load_checkpoint


_NMFDU_STATE_PREFIX = "id_backbone.nmfdu_gate."


def parse_index_csv(value: object) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not indices:
        raise ValueError("source scope requires at least one receiver/day index")
    return indices


def build_real_source_batch(
    dataset: Mapping[str, Any],
    *,
    input_len: int,
    train_rxs: Sequence[int],
    train_days: Sequence[int],
    equalized: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Select only declared Phase1 source train receiver/day rows."""

    source = WiSigCompactDataset(
        dict(dataset),
        out_len=int(input_len),
        crop_mode="center",
        normalize=True,
        center=False,
        equalized=int(equalized),
        rx_keep=tuple(int(value) for value in train_rxs),
        day_keep=tuple(int(value) for value in train_days),
        domain="rx_day",
        max_samples_per_combo=max(1, int(batch_size)),
        sample_strategy="front",
        seed=0,
    )
    count = min(int(batch_size), len(source))
    if count < 1:
        raise ValueError("declared Phase1 source scope contains no physical samples")
    rows = [source[index] for index in range(count)]
    x = torch.stack([row[0] for row in rows])
    y = torch.tensor([int(row[1]) for row in rows], dtype=torch.long)
    metadata = []
    for row in rows:
        item = dict(row[3])
        item["source_role"] = "phase1_source_labeled_smoke"
        metadata.append(item)
    return x, y, metadata


def validate_legacy_transfer(
    missing_keys: Sequence[str], unexpected_keys: Sequence[str]
) -> None:
    if unexpected_keys:
        raise ValueError(f"legacy transfer produced unexpected keys: {list(unexpected_keys)}")
    outside = [key for key in missing_keys if not str(key).startswith(_NMFDU_STATE_PREFIX)]
    if outside:
        raise ValueError(f"legacy transfer missing state outside NMFDU: {outside}")
    if not missing_keys:
        raise ValueError("NMFDU model did not expose any new state")


def _build_nmfdu_model(
    checkpoint: Mapping[str, Any],
    legacy_model: torch.nn.Module,
    *,
    input_len: int,
    num_domains: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    from SSDG import train_ssdg

    parser = train_ssdg.build_arg_parser()
    parsed = parser.parse_args(["--output_dir", ".tmp_nmfdu_real_checkpoint_smoke"])
    for key, value in dict(checkpoint.get("args") or {}).items():
        setattr(parsed, key, value)
    parsed.device = str(device)
    merged = train_ssdg.merge_checkpoint_args(
        checkpoint,
        parsed,
        input_len=int(input_len),
        num_domains=int(num_domains),
    )
    merged = train_ssdg._apply_model_cli_args(merged, parsed)
    merged.physical_gate_variant = "nmfdu_v1"
    model = train_ssdg.build_baseline_model(merged, device)
    incompatible = model.load_state_dict(legacy_model.state_dict(), strict=False)
    validate_legacy_transfer(incompatible.missing_keys, incompatible.unexpected_keys)
    return model, list(incompatible.missing_keys)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    wisig_path = Path(args.wisig_pkl).resolve()
    if not checkpoint_path.is_file() or not wisig_path.is_file():
        raise FileNotFoundError("checkpoint and wisig_pkl must both exist")
    device = torch.device(str(args.device))
    checkpoint = load_checkpoint(str(checkpoint_path), torch.device("cpu"))
    checkpoint_args = dict(checkpoint.get("args") or {})
    input_len = int(checkpoint_args.get("wisig_out_len", 256))
    legacy_model, legacy_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=input_len,
        device=device,
    )
    dataset = load_wisig_compact_pkl(str(wisig_path))
    x, y, metadata = build_real_source_batch(
        dataset,
        input_len=input_len,
        train_rxs=parse_index_csv(checkpoint_args.get("wisig_train_rxs", "0")),
        train_days=parse_index_csv(checkpoint_args.get("wisig_train_days", "0")),
        equalized=int(checkpoint_args.get("wisig_equalized", 1)),
        batch_size=int(args.batch_size),
    )
    x = x.to(device)
    y = y.to(device)

    legacy_model.eval()
    with torch.no_grad():
        legacy_output = legacy_model(
            x,
            y_tx=None,
            grl_lambda=1.0,
            return_aux=True,
        )
    if not torch.isfinite(legacy_output["tx_logits"]).all():
        raise RuntimeError("legacy strict checkpoint produced non-finite logits")

    model, initialized_keys = _build_nmfdu_model(
        checkpoint,
        legacy_model,
        input_len=input_len,
        num_domains=int(legacy_audit["num_domains_from_state"]),
        device=device,
    )
    controller = NMFDUStageController((80, 120, 200))
    stage = controller.apply(model, 121)
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(
        x,
        y_tx=y,
        grl_lambda=1.0,
        return_aux=True,
        update_nmfdu_support=True,
        nmfdu_support_mask=torch.ones_like(y, dtype=torch.bool),
        return_physical_gate_diag=True,
    )
    loss = F.cross_entropy(output["tx_logits"], y)
    loss.backward()
    identity_output = output.get("aux_id", output)
    gate = identity_output["physical_gate_diag"]
    per_sample = gate["per_sample"]
    finite_keys = ("weights", "null_weight", "q_sample", "I", "D", "S", "U")
    if not all(torch.isfinite(per_sample[key]).all() for key in finite_keys):
        raise RuntimeError("NMFDU produced non-finite physical diagnostics")
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(_NMFDU_STATE_PREFIX) and parameter.grad is not None
    ]
    nonzero_gradients = sum(
        int(bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0.0)
        for gradient in gradients
    )
    if nonzero_gradients < 1:
        raise RuntimeError("NMFDU real-source backward produced no finite nonzero gradient")

    return {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "wisig_pkl": str(wisig_path),
        "input_len": input_len,
        "rows": int(y.numel()),
        "source_sample_metadata": metadata,
        "query_truth_read": False,
        "phase2_support_or_query_read": False,
        "labels_or_roles_passed_to_physical_evidence": False,
        "legacy_checkpoint_load": legacy_audit,
        "legacy_prediction_shape": list(legacy_output["tx_logits"].shape),
        "nmfdu_prediction_shape": list(output["tx_logits"].shape),
        "nmfdu_stage": int(stage.index),
        "nmfdu_initialized_key_count": len(initialized_keys),
        "nmfdu_initialized_key_prefix": _NMFDU_STATE_PREFIX,
        "branch_names": list(gate["branch_names"]),
        "q_sample_mean": float(per_sample["q_sample"].mean().detach().cpu()),
        "null_mean": float(per_sample["null_weight"].mean().detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "nmfdu_nonzero_gradient_tensors": nonzero_gradients,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADV3B02 NMFDU real-checkpoint no-query smoke")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_smoke(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
