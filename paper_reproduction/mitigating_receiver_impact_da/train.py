from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import write_json
from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
from paper_reproduction.mitigating_receiver_impact_da.protocol import (
    build_paper_task_plan,
    validate_paper_faithful_config,
)


def build_dry_run_payload(config: dict) -> dict:
    checked = validate_paper_faithful_config(config)
    return {
        "method_id": "mitigating_receiver_impact_da",
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "citation": "Liu Yang, Qiang Li, Xiaoyang Ren, Yi Fang, and Shafei Wang, IEEE Internet of Things Journal, 2024",
        "algorithm": "GAD adversarial training with DV-KL domain alignment and adaptive pseudo-labeling",
        "scope": checked["claim_boundary"],
        "dataset": checked["dataset"],
        "capture_days": checked["capture_days"],
        "source_target_tasks": checked["source_target_tasks"],
        "target_labels_scope": "evaluation_only",
        "paper_task_plan": build_paper_task_plan(checked),
        "paper_reported_hyperparameters": checked["paper_reported_hyperparameters"],
        "paper_unspecified_fields": checked["paper_unspecified_fields"],
        "paper_evidence_targets": {
            "Table II": "task/display-method plan only",
            "Table III": "not reproduced in dry-run",
            "Table IV": "not reproduced in dry-run",
            "Fig.5-7": "not reproduced in dry-run",
        },
        "claim_blocks": [
            "not CVS Stage2-A/B/C",
            "not satellite/LEO deployment evidence",
            "not open-set or new-class registration",
            "target labels are evaluation-only in paper-faithful DA",
        ],
    }


def _batch_tensor(batch: Any, key: str, fallback_index: int) -> torch.Tensor:
    if isinstance(batch, dict):
        value = batch[key]
    else:
        value = batch[fallback_index]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _next_cycling(iterator: Iterable[Any], current_iterator: Any, *, name: str) -> tuple[Any, Any]:
    try:
        return next(current_iterator), current_iterator
    except StopIteration:
        current_iterator = iter(iterator)
    try:
        return next(current_iterator), current_iterator
    except StopIteration as exc:
        raise ValueError(f"{name} batches cannot be empty") from exc


def _state_payload(state: PseudoLabelState) -> dict[str, Any]:
    return {
        "pseudo_counts": [float(v) for v in state.pseudo_counts.tolist()],
        "predicted_counts": [float(v) for v in state.predicted_counts.tolist()],
        "total_seen": int(state.total_seen),
    }


def run_gada_training_loop(
    model: ReceiverImpactGADNet,
    source_batches: Iterable[Any],
    target_batches: Iterable[Any],
    *,
    optimizer_t: Any,
    optimizer_ec: Any,
    epochs: int,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    estimate_steps: int = 7,
    base_tau: float = 0.7,
    mu: float = 0.5,
    kl_weight: float = 0.005,
    class_prior: torch.Tensor | None = None,
    max_batches_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Execute the Algorithm 1 GAD loop over caller-provided source/target batches.

    This is intentionally data-loader agnostic: paper-faithful WiSig task construction
    remains outside this helper, while this function owns the update ordering,
    pseudo-label state, metrics, and checkpoint surface.
    """
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if max_batches_per_epoch is not None and int(max_batches_per_epoch) <= 0:
        raise ValueError("max_batches_per_epoch must be positive when provided")

    resolved_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(resolved_device)
    state = PseudoLabelState(num_classes=int(model.num_tx))
    target_iterator = iter(target_batches)
    history: list[dict[str, float | int]] = []
    total_batches = 0

    for epoch_index in range(int(epochs)):
        epoch_batches = 0
        epoch_loss = 0.0
        epoch_selected = 0
        source_iterator = iter(source_batches)
        for source_batch in source_iterator:
            if max_batches_per_epoch is not None and epoch_batches >= int(max_batches_per_epoch):
                break
            target_batch, target_iterator = _next_cycling(target_batches, target_iterator, name="target")
            source_x = _batch_tensor(source_batch, "iq", 0).to(resolved_device)
            source_y = _batch_tensor(source_batch, "label", 1).long().to(resolved_device)
            target_x = _batch_tensor(target_batch, "iq", 0).to(resolved_device)
            result = gada_batch_step(
                model,
                source_x,
                source_y,
                target_x,
                state=state,
                optimizer_t=optimizer_t,
                optimizer_ec=optimizer_ec,
                estimate_steps=estimate_steps,
                base_tau=base_tau,
                mu=mu,
                kl_weight=kl_weight,
                class_prior=class_prior,
            )
            epoch_batches += 1
            total_batches += 1
            epoch_loss += float(result["loss"].item())
            epoch_selected += int(result["target_selected"].item())

        if epoch_batches == 0:
            raise ValueError("source batches cannot be empty")
        history.append(
            {
                "epoch": epoch_index + 1,
                "batches": epoch_batches,
                "loss_mean": epoch_loss / float(epoch_batches),
                "target_selected": epoch_selected,
                "target_seen_total": int(state.total_seen),
            }
        )

    payload: dict[str, Any] = {
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "algorithm": "Algorithm 1 GAD training loop",
        "epochs": int(epochs),
        "batches": total_batches,
        "estimate_steps": int(estimate_steps),
        "base_tau": float(base_tau),
        "mu": float(mu),
        "kl_weight": float(kl_weight),
        "history": history,
        "state": _state_payload(state),
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **payload,
                "epoch": int(epochs),
                "model_state_dict": model.state_dict(),
            },
            path,
        )
        payload["checkpoint_path"] = str(path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful dry-run entrypoint for the IoTJ 2024 receiver-impact DA RFFI paper.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the reproduction matrix.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for dry-run payload.")
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("formal WiSig training CLI is intentionally gated; use --dry-run or call run_gada_training_loop with explicit loaders")

    config = load_json_config(args.config)
    payload = build_dry_run_payload(config)
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
