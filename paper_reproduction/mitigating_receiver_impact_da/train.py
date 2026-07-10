from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import set_seed, tx_accuracy, write_json
from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
from paper_reproduction.mitigating_receiver_impact_da.data import build_manysig_task_loaders, load_wisig_compact_pkl
from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
from paper_reproduction.mitigating_receiver_impact_da.protocol import (
    PAPER_TASKS,
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


def _optional_batch_tensor(batch: Any, key: str, fallback_index: int) -> torch.Tensor | None:
    try:
        if isinstance(batch, dict):
            if key not in batch:
                return None
            value = batch[key]
        else:
            if not hasattr(batch, "__len__") or len(batch) <= fallback_index:
                return None
            value = batch[fallback_index]
    except (KeyError, IndexError, TypeError):
        return None
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _batch_base_indices(batch: Any) -> torch.Tensor | None:
    if not isinstance(batch, dict) or "meta" not in batch:
        return None
    indices: list[int] = []
    for meta in batch["meta"]:
        if not isinstance(meta, dict) or "base_index" not in meta:
            return None
        indices.append(int(meta["base_index"]))
    return torch.tensor(indices, dtype=torch.long)


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


def _tensor_int_list(value: torch.Tensor) -> list[int]:
    return [int(v) for v in value.detach().cpu().flatten().tolist()]


def _tensor_float_list(value: torch.Tensor) -> list[float]:
    return [float(v) for v in value.detach().cpu().flatten().tolist()]


def _safe_ratio_lists(numerator: torch.Tensor, denominator: torch.Tensor) -> list[float | None]:
    out: list[float | None] = []
    for num, den in zip(numerator.detach().cpu().flatten().tolist(), denominator.detach().cpu().flatten().tolist()):
        den_f = float(den)
        out.append(None if den_f <= 0.0 else float(num) / den_f)
    return out


def _target_state_shape(target_batches: Iterable[Any]) -> tuple[int | None, int | None]:
    try:
        dataset = getattr(target_batches, "dataset", None)
        target_size = None if dataset is None else int(len(dataset))
    except TypeError:
        target_size = None
    try:
        target_batch_count = int(len(target_batches))  # type: ignore[arg-type]
    except TypeError:
        target_batch_count = None
    return target_size, target_batch_count


def _iterate_paired_batches(
    source_batches: Iterable[Any],
    target_batches: Iterable[Any],
    *,
    mode: str,
    target_iterator: Any,
) -> tuple[Iterable[tuple[Any, Any]], Any]:
    normalized = str(mode).strip().lower()
    if normalized == "zip_min":
        return zip(iter(source_batches), iter(target_batches)), target_iterator
    if normalized == "cycle_target":
        def _generator() -> Iterable[tuple[Any, Any]]:
            nonlocal target_iterator
            for source_batch in iter(source_batches):
                target_batch, target_iterator = _next_cycling(target_batches, target_iterator, name="target")
                yield source_batch, target_batch

        return _generator(), target_iterator
    raise ValueError("batch_pairing must be one of: cycle_target, zip_min")


def _evaluate_target_metrics(
    model: ReceiverImpactGADNet,
    loader: Iterable[Any],
    *,
    device: torch.device | str,
    include_loss: bool = False,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    batches = 0
    confusion: torch.Tensor | None = None
    with torch.no_grad():
        for batch in loader:
            x = _batch_tensor(batch, "iq", 0).to(device)
            y = _batch_tensor(batch, "label", 1).long().to(device)
            logits = model.classify(x)
            if include_loss:
                loss_sum += float(F.cross_entropy(logits, y).detach().cpu())
            pred = logits.argmax(dim=1)
            if confusion is None:
                num_classes = int(getattr(model, "num_tx", int(logits.shape[1])))
                confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
            batch_correct = pred == y
            correct += int(batch_correct.sum().item())
            total += int(y.numel())
            batches += 1
            for true_label, pred_label in zip(y.detach().cpu().flatten().tolist(), pred.detach().cpu().flatten().tolist()):
                true_i = int(true_label)
                pred_i = int(pred_label)
                if 0 <= true_i < confusion.shape[0] and 0 <= pred_i < confusion.shape[1]:
                    confusion[true_i, pred_i] += 1
    if batches == 0:
        raise ValueError("target evaluation batches cannot be empty")
    if confusion is None:
        raise ValueError("target evaluation produced no predictions")
    total_by_class = confusion.sum(dim=1)
    pred_hist = confusion.sum(dim=0)
    correct_by_class = confusion.diag()
    payload: dict[str, Any] = {
        "target_accuracy": 0.0 if total <= 0 else correct / float(total),
        "target_total": int(total),
        "target_batches": int(batches),
        "target_true_hist": _tensor_int_list(total_by_class),
        "target_pred_hist": _tensor_int_list(pred_hist),
        "target_correct_by_class": _tensor_int_list(correct_by_class),
        "target_total_by_class": _tensor_int_list(total_by_class),
        "target_accuracy_by_class": _safe_ratio_lists(correct_by_class, total_by_class),
        "target_confusion_matrix": [[int(v) for v in row] for row in confusion.detach().cpu().tolist()],
    }
    if include_loss:
        payload["target_loss"] = loss_sum / float(batches)
    return payload


def _evaluate_target_loss_accuracy(model: ReceiverImpactGADNet, loader: Iterable[Any], *, device: torch.device | str) -> dict[str, Any]:
    return _evaluate_target_metrics(model, loader, device=device, include_loss=True)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    mode: str,
    step_size: int,
    gamma: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    normalized = str(mode).strip().lower()
    if normalized == "none":
        return None
    if normalized != "step":
        raise ValueError("lr_scheduler_mode must be one of: none, step")
    if int(step_size) <= 0:
        raise ValueError("lr_step_size must be positive when StepLR is enabled")
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("lr_gamma must be in (0,1] when StepLR is enabled")
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))


def _scheduler_step(scheduler: Any | None) -> None:
    if scheduler is not None:
        scheduler.step()


def _optimizer_lr(optimizer: Any) -> float | None:
    param_groups = getattr(optimizer, "param_groups", None)
    if not param_groups:
        return None
    return float(param_groups[0].get("lr", 0.0))


def run_gada_training_loop(
    model: ReceiverImpactGADNet,
    source_batches: Iterable[Any],
    target_batches: Iterable[Any],
    *,
    optimizer_t: Any,
    optimizer_ec: Any,
    scheduler_t: Any | None = None,
    scheduler_ec: Any | None = None,
    epochs: int,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    estimate_steps: int = 7,
    base_tau: float = 0.7,
    mu: float = 0.5,
    kl_weight: float = 0.005,
    class_prior: torch.Tensor | None = None,
    class_weight_smoothing: float = 0.0,
    class_weight_clip_min: float | None = None,
    class_weight_clip_max: float | None = None,
    class_weight_mean_normalize: bool = False,
    kl_estimator_mode: str = "dvkl",
    mine_ma_rate: float = 0.01,
    mine_update_scale: float = 0.5,
    pseudo_threshold_mode: str = "paper",
    pseudo_score_mode: str = "probability",
    pseudo_threshold_floor: float = 0.0,
    pseudo_quota_mode: str = "none",
    pseudo_quota_per_class: int | None = None,
    class_weight_timing: str = "previous",
    pseudo_state_scope: str = "epoch",
    batch_pairing: str = "zip_min",
    adapt_start_epoch: int = 0,
    label_smoothing: float = 0.0,
    target_eval_batches: Iterable[Any] | None = None,
    target_model_selection: str = "final",
    weighted_ce_reduction: str = "paper_sample_mean",
    use_domain_alignment: bool = True,
    use_pseudo: bool = True,
    use_class_weight: bool = True,
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
    state_scope = str(pseudo_state_scope).strip().lower()
    if state_scope not in {"global", "epoch"}:
        raise ValueError("pseudo_state_scope must be one of: global, epoch")
    target_size, target_batches_count = _target_state_shape(target_batches)
    state = PseudoLabelState(
        num_classes=int(model.num_tx),
        target_size=target_size if state_scope == "epoch" else None,
        target_batches=target_batches_count if state_scope == "epoch" else None,
    )
    target_iterator = iter(target_batches)
    history: list[dict[str, float | int]] = []
    target_eval_history: list[dict[str, float | int]] = []
    total_batches = 0
    best_target_loss = float("inf")
    best_target_epoch: int | None = None
    best_target_state: dict[str, torch.Tensor] | None = None
    normalized_selection = str(target_model_selection).strip().lower()
    if normalized_selection not in {"final", "target_loss_best"}:
        raise ValueError("target_model_selection must be one of: final, target_loss_best")
    normalized_quota = str(pseudo_quota_mode).strip().lower()
    if normalized_quota not in {"none", "balanced_topk"}:
        raise ValueError("pseudo_quota_mode must be one of: none, balanced_topk")
    if normalized_quota == "balanced_topk" and (pseudo_quota_per_class is None or int(pseudo_quota_per_class) <= 0):
        raise ValueError("pseudo_quota_per_class must be positive when balanced_topk is enabled")

    for epoch_index in range(int(epochs)):
        epoch_number = epoch_index + 1
        if state_scope == "epoch":
            state.reset_epoch()
        epoch_batches = 0
        epoch_loss = 0.0
        epoch_loss_source = 0.0
        epoch_loss_target = 0.0
        epoch_loss_kl = 0.0
        epoch_estimate_zeta = 0.0
        epoch_conf = 0.0
        epoch_selected = 0
        epoch_selected_correct = 0
        epoch_audit_total = 0
        epoch_pred_correct = 0
        epoch_weight_min = float("inf")
        epoch_weight_max = float("-inf")
        num_classes = int(model.num_tx)
        epoch_pred_hist = torch.zeros(num_classes, dtype=torch.long)
        epoch_true_hist = torch.zeros(num_classes, dtype=torch.long)
        epoch_pred_correct_by_true = torch.zeros(num_classes, dtype=torch.long)
        epoch_selected_hist = torch.zeros(num_classes, dtype=torch.long)
        epoch_selected_true_hist = torch.zeros(num_classes, dtype=torch.long)
        epoch_selected_correct_by_true = torch.zeros(num_classes, dtype=torch.long)
        epoch_selected_correct_by_pred = torch.zeros(num_classes, dtype=torch.long)
        epoch_weight_vector_sum = torch.zeros(num_classes, dtype=torch.float64)
        epoch_threshold_vector_sum = torch.zeros(num_classes, dtype=torch.float64)
        epoch_weight_vector_last = torch.ones(num_classes, dtype=torch.float64)
        epoch_threshold_vector_last = torch.full((num_classes,), float(base_tau), dtype=torch.float64)
        paired_batches, target_iterator = _iterate_paired_batches(
            source_batches,
            target_batches,
            mode=batch_pairing,
            target_iterator=target_iterator,
        )
        for source_batch, target_batch in paired_batches:
            if max_batches_per_epoch is not None and epoch_batches >= int(max_batches_per_epoch):
                break
            source_x = _batch_tensor(source_batch, "iq", 0).to(resolved_device)
            source_y = _batch_tensor(source_batch, "label", 1).long().to(resolved_device)
            target_x = _batch_tensor(target_batch, "iq", 0).to(resolved_device)
            target_y_audit = _optional_batch_tensor(target_batch, "label", 1)
            if target_y_audit is not None:
                target_y_audit = target_y_audit.long().to(resolved_device)
            target_indices = _batch_base_indices(target_batch) if state_scope == "epoch" else None
            if epoch_number <= int(adapt_start_epoch):
                model.train()
                optimizer_ec.zero_grad()
                logits = model.classify(source_x)
                source_loss = F.cross_entropy(logits, source_y, label_smoothing=float(label_smoothing))
                source_loss.backward()
                optimizer_ec.step()
                result = {
                    "loss": source_loss.detach(),
                    "loss_weighted_ce": source_loss.detach(),
                    "loss_source": source_loss.detach(),
                    "loss_target": source_loss.detach() * 0.0,
                    "loss_kl": source_loss.detach() * 0.0,
                    "target_selected": torch.tensor(0, device=resolved_device),
                    "target_conf_mean": torch.tensor(0.0, device=resolved_device),
                    "class_weight_min": torch.tensor(1.0, device=resolved_device),
                    "class_weight_max": torch.tensor(1.0, device=resolved_device),
                    "estimate_steps": 0,
                    "estimate_loss": torch.tensor(0.0, device=resolved_device),
                    "estimate_zeta": torch.tensor(0.0, device=resolved_device),
                }
            else:
                result = gada_batch_step(
                    model,
                    source_x,
                    source_y,
                    target_x,
                    target_y_audit=target_y_audit,
                    state=state,
                    optimizer_t=optimizer_t,
                    optimizer_ec=optimizer_ec,
                    estimate_steps=estimate_steps,
                    base_tau=base_tau,
                    mu=mu,
                    kl_weight=kl_weight,
                    class_prior=class_prior,
                    class_weight_smoothing=class_weight_smoothing,
                    class_weight_clip_min=class_weight_clip_min,
                    class_weight_clip_max=class_weight_clip_max,
                    class_weight_mean_normalize=class_weight_mean_normalize,
                    kl_estimator_mode=kl_estimator_mode,
                    mine_ma_rate=mine_ma_rate,
                    mine_update_scale=mine_update_scale,
                    pseudo_threshold_mode=pseudo_threshold_mode,
                    pseudo_score_mode=pseudo_score_mode,
                    pseudo_threshold_floor=pseudo_threshold_floor,
                    pseudo_quota_mode=normalized_quota,
                    pseudo_quota_per_class=pseudo_quota_per_class,
                    class_weight_timing=class_weight_timing,
                    weighted_ce_reduction=weighted_ce_reduction,
                    use_domain_alignment=use_domain_alignment,
                    use_pseudo=use_pseudo,
                    use_class_weight=use_class_weight,
                    target_indices=target_indices,
                )
            _scheduler_step(scheduler_ec)
            _scheduler_step(scheduler_t)
            epoch_batches += 1
            total_batches += 1
            epoch_loss += float(result["loss"].item())
            epoch_loss_source += float(result["loss_source"].item())
            epoch_loss_target += float(result["loss_target"].item())
            epoch_loss_kl += float(result["loss_kl"].item())
            epoch_estimate_zeta += float(result["estimate_zeta"].item())
            epoch_conf += float(result["target_conf_mean"].item())
            epoch_selected += int(result["target_selected"].item())
            if "target_audit_total" in result:
                epoch_selected_correct += int(result["target_selected_correct"].item())
                epoch_audit_total += int(result["target_audit_total"].item())
                epoch_pred_correct += int(result["target_pred_correct"].item())
            if "target_pred_hist" in result:
                epoch_pred_hist += result["target_pred_hist"].detach().cpu().long()
            if "target_pseudo_selected_hist" in result:
                epoch_selected_hist += result["target_pseudo_selected_hist"].detach().cpu().long()
            if "target_true_hist" in result:
                epoch_true_hist += result["target_true_hist"].detach().cpu().long()
            if "target_pred_correct_by_true_class" in result:
                epoch_pred_correct_by_true += result["target_pred_correct_by_true_class"].detach().cpu().long()
            if "target_pseudo_selected_true_hist" in result:
                epoch_selected_true_hist += result["target_pseudo_selected_true_hist"].detach().cpu().long()
            if "target_pseudo_selected_correct_by_true_class" in result:
                epoch_selected_correct_by_true += result["target_pseudo_selected_correct_by_true_class"].detach().cpu().long()
            if "target_pseudo_selected_correct_by_pred_class" in result:
                epoch_selected_correct_by_pred += result["target_pseudo_selected_correct_by_pred_class"].detach().cpu().long()
            if "class_weight_vector" in result:
                weight_vector = result["class_weight_vector"].detach().cpu().double()
                epoch_weight_vector_sum += weight_vector
                epoch_weight_vector_last = weight_vector
            if "pseudo_threshold_vector" in result:
                threshold_vector = result["pseudo_threshold_vector"].detach().cpu().double()
                epoch_threshold_vector_sum += threshold_vector
                epoch_threshold_vector_last = threshold_vector
            epoch_weight_min = min(epoch_weight_min, float(result["class_weight_min"].item()))
            epoch_weight_max = max(epoch_weight_max, float(result["class_weight_max"].item()))

        if epoch_batches == 0:
            raise ValueError("source batches cannot be empty")
        if not torch.isfinite(torch.tensor(epoch_weight_min)):
            epoch_weight_min = 1.0
        if not torch.isfinite(torch.tensor(epoch_weight_max)):
            epoch_weight_max = 1.0
        history.append(
            {
                "epoch": epoch_number,
                "batches": epoch_batches,
                "loss_mean": epoch_loss / float(epoch_batches),
                "loss_source_mean": epoch_loss_source / float(epoch_batches),
                "loss_target_mean": epoch_loss_target / float(epoch_batches),
                "loss_kl_mean": epoch_loss_kl / float(epoch_batches),
                "estimate_zeta_mean": epoch_estimate_zeta / float(epoch_batches),
                "target_conf_mean": epoch_conf / float(epoch_batches),
                "class_weight_min": epoch_weight_min,
                "class_weight_max": epoch_weight_max,
                "class_weight_mean_by_class": _tensor_float_list(epoch_weight_vector_sum / float(epoch_batches)),
                "class_weight_last_by_class": _tensor_float_list(epoch_weight_vector_last),
                "pseudo_threshold_mean_by_class": _tensor_float_list(epoch_threshold_vector_sum / float(epoch_batches)),
                "pseudo_threshold_last_by_class": _tensor_float_list(epoch_threshold_vector_last),
                "target_selected": epoch_selected,
                "target_seen_total": int(state.total_seen),
                "target_pred_hist": _tensor_int_list(epoch_pred_hist),
                "target_true_hist": _tensor_int_list(epoch_true_hist),
                "target_pred_correct_by_true_class": _tensor_int_list(epoch_pred_correct_by_true),
                "target_pred_acc_by_true_class": _safe_ratio_lists(epoch_pred_correct_by_true, epoch_true_hist),
                "target_pseudo_selected_hist": _tensor_int_list(epoch_selected_hist),
                "target_pseudo_selected_true_hist": _tensor_int_list(epoch_selected_true_hist),
                "target_pseudo_selected_correct_by_true_class": _tensor_int_list(epoch_selected_correct_by_true),
                "target_pseudo_selected_correct_by_pred_class": _tensor_int_list(epoch_selected_correct_by_pred),
                "target_pseudo_selected_acc_by_true_class": _safe_ratio_lists(epoch_selected_correct_by_true, epoch_selected_true_hist),
                "target_pseudo_selected_acc_by_pred_class": _safe_ratio_lists(epoch_selected_correct_by_pred, epoch_selected_hist),
                "target_pseudo_selected_acc": None if epoch_selected <= 0 else epoch_selected_correct / float(epoch_selected),
                "target_pred_acc": None if epoch_audit_total <= 0 else epoch_pred_correct / float(epoch_audit_total),
                "lr_ec": _optimizer_lr(optimizer_ec),
                "lr_t": _optimizer_lr(optimizer_t),
            }
        )
        if target_eval_batches is not None:
            eval_row = _evaluate_target_loss_accuracy(model, target_eval_batches, device=resolved_device)
            eval_row["epoch"] = epoch_number
            target_eval_history.append(eval_row)
            if normalized_selection == "target_loss_best" and float(eval_row["target_loss"]) < best_target_loss:
                best_target_loss = float(eval_row["target_loss"])
                best_target_epoch = epoch_number
                best_target_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if normalized_selection == "target_loss_best" and best_target_state is not None:
        model.load_state_dict(best_target_state)

    payload: dict[str, Any] = {
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "algorithm": "Algorithm 1 GAD training loop",
        "model_profile": str(getattr(model, "model_profile", "unknown")),
        "epochs": int(epochs),
        "batches": total_batches,
        "estimate_steps": int(estimate_steps),
        "base_tau": float(base_tau),
        "mu": float(mu),
        "kl_weight": float(kl_weight),
        "class_weight_smoothing": float(class_weight_smoothing),
        "class_weight_clip_min": None if class_weight_clip_min is None else float(class_weight_clip_min),
        "class_weight_clip_max": None if class_weight_clip_max is None else float(class_weight_clip_max),
        "class_weight_mean_normalize": bool(class_weight_mean_normalize),
        "kl_estimator_mode": str(kl_estimator_mode),
        "mine_ma_rate": float(mine_ma_rate),
        "mine_update_scale": float(mine_update_scale),
        "pseudo_threshold_mode": str(pseudo_threshold_mode),
        "pseudo_score_mode": str(pseudo_score_mode),
        "pseudo_threshold_floor": float(pseudo_threshold_floor),
        "pseudo_quota_mode": normalized_quota,
        "pseudo_quota_per_class": None if pseudo_quota_per_class is None else int(pseudo_quota_per_class),
        "class_weight_timing": str(class_weight_timing),
        "pseudo_state_scope": str(pseudo_state_scope),
        "batch_pairing": str(batch_pairing),
        "adapt_start_epoch": int(adapt_start_epoch),
        "label_smoothing": float(label_smoothing),
        "lr_scheduler_active": bool(scheduler_t is not None or scheduler_ec is not None),
        "target_model_selection": str(target_model_selection),
        "weighted_ce_reduction": str(weighted_ce_reduction),
        "use_domain_alignment": bool(use_domain_alignment),
        "use_pseudo": bool(use_pseudo),
        "use_class_weight": bool(use_class_weight),
        "target_eval_history": target_eval_history,
        "best_target_loss_epoch": best_target_epoch,
        "best_target_loss": None if best_target_epoch is None else best_target_loss,
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


def _limited_batches(loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    if max_batches is None:
        return loader
    limit = int(max_batches)

    def _generator() -> Iterable[Any]:
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            yield batch

    return _generator()


def _evaluate_target_accuracy(model: ReceiverImpactGADNet, loader: Iterable[Any], *, device: torch.device | str) -> float:
    return float(_evaluate_target_metrics(model, loader, device=device, include_loss=False)["target_accuracy"])


def _source_class_prior_from_dataset(dataset: Any, *, num_classes: int) -> torch.Tensor:
    """Estimate Eq. (9) p_prior(k) from the labeled source set."""
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    index = getattr(dataset, "index", None)
    if index is not None:
        index_complete = True
        for item in index:
            label = getattr(item, "tx_i", None)
            if label is None and isinstance(item, dict):
                label = item.get("tx_i")
            if label is None:
                index_complete = False
                break
            label_i = int(label)
            if 0 <= label_i < int(num_classes):
                counts[label_i] += 1.0
        if index_complete and counts.sum() > 0:
            return counts / counts.sum()
        counts.zero_()

    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise ValueError("source dataset must expose labels through index or __getitem__")
    for item_index in range(len(dataset)):
        label_tensor = _batch_tensor(dataset[item_index], "label", 1).long().flatten()
        for label in label_tensor.tolist():
            label_i = int(label)
            if 0 <= label_i < int(num_classes):
                counts[label_i] += 1.0
    if counts.sum() <= 0:
        raise ValueError("source class prior cannot be estimated from an empty or unlabeled source dataset")
    return counts / counts.sum()


def _resolve_class_prior(source_loader: Any, *, num_classes: int, mode: str) -> torch.Tensor | None:
    normalized = str(mode).strip().lower()
    if normalized == "source":
        dataset = getattr(source_loader, "dataset", source_loader)
        return _source_class_prior_from_dataset(dataset, num_classes=num_classes)
    if normalized == "uniform":
        return torch.full((int(num_classes),), 1.0 / float(num_classes), dtype=torch.float32)
    if normalized in {"none", "disabled"}:
        return None
    raise ValueError("class_prior_mode must be one of: source, uniform, none")


def _audit_target_predictions(
    model: Any,
    loader: Iterable[Any],
    *,
    device: torch.device | str,
    tau_values: tuple[float, ...] = (0.7, 0.95),
) -> dict[str, Any]:
    model.eval()
    tau_stats = {
        float(tau): {
            "selected": 0,
            "selected_correct": 0,
        }
        for tau in tau_values
    }
    correct = 0
    total = 0
    conf_sum = 0.0
    true_hist: torch.Tensor | None = None
    pred_hist: torch.Tensor | None = None
    correct_by_true: torch.Tensor | None = None
    with torch.no_grad():
        for batch in loader:
            x = _batch_tensor(batch, "iq", 0).to(device)
            y = _batch_tensor(batch, "label", 1).long().to(device)
            probs = torch.softmax(model.classify(x), dim=1)
            confidence, pred = probs.max(dim=1)
            if true_hist is None:
                num_classes = int(probs.shape[1])
                true_hist = torch.zeros(num_classes, dtype=torch.long)
                pred_hist = torch.zeros(num_classes, dtype=torch.long)
                correct_by_true = torch.zeros(num_classes, dtype=torch.long)
            batch_correct = pred == y
            correct += int(batch_correct.sum().item())
            total += int(y.numel())
            conf_sum += float(confidence.sum().item())
            true_hist += torch.bincount(y.detach().cpu(), minlength=int(probs.shape[1]))
            pred_hist += torch.bincount(pred.detach().cpu(), minlength=int(probs.shape[1]))
            correct_by_true += torch.bincount(y[batch_correct].detach().cpu(), minlength=int(probs.shape[1]))
            for tau, stats in tau_stats.items():
                selected = confidence > float(tau)
                stats["selected"] += int(selected.sum().item())
                stats["selected_correct"] += int((batch_correct & selected).sum().item())
                stats.setdefault("selected_pred_hist", torch.zeros(int(probs.shape[1]), dtype=torch.long))
                stats.setdefault("selected_true_hist", torch.zeros(int(probs.shape[1]), dtype=torch.long))
                stats.setdefault("selected_correct_by_true_class", torch.zeros(int(probs.shape[1]), dtype=torch.long))
                stats["selected_pred_hist"] += torch.bincount(pred[selected].detach().cpu(), minlength=int(probs.shape[1]))
                stats["selected_true_hist"] += torch.bincount(y[selected].detach().cpu(), minlength=int(probs.shape[1]))
                stats["selected_correct_by_true_class"] += torch.bincount(
                    y[selected & batch_correct].detach().cpu(),
                    minlength=int(probs.shape[1]),
                )
    tau_sweep = []
    for tau, stats in sorted(tau_stats.items()):
        selected = int(stats["selected"])
        selected_true_hist = stats.get("selected_true_hist")
        selected_correct_by_true = stats.get("selected_correct_by_true_class")
        tau_sweep.append(
            {
                "tau": float(tau),
                "selected": selected,
                "coverage": 0.0 if total <= 0 else selected / float(total),
                "selected_acc": None if selected <= 0 else int(stats["selected_correct"]) / float(selected),
                "selected_pred_hist": [] if "selected_pred_hist" not in stats else _tensor_int_list(stats["selected_pred_hist"]),
                "selected_true_hist": [] if selected_true_hist is None else _tensor_int_list(selected_true_hist),
                "selected_correct_by_true_class": []
                if selected_correct_by_true is None
                else _tensor_int_list(selected_correct_by_true),
                "selected_acc_by_true_class": []
                if selected_true_hist is None or selected_correct_by_true is None
                else _safe_ratio_lists(selected_correct_by_true, selected_true_hist),
            }
        )
    if true_hist is None or pred_hist is None or correct_by_true is None:
        raise ValueError("target audit batches cannot be empty")
    return {
        "total": int(total),
        "target_pred_acc": None if total <= 0 else correct / float(total),
        "target_conf_mean": None if total <= 0 else conf_sum / float(total),
        "target_true_hist": _tensor_int_list(true_hist),
        "target_pred_hist": _tensor_int_list(pred_hist),
        "target_correct_by_true_class": _tensor_int_list(correct_by_true),
        "target_acc_by_true_class": _safe_ratio_lists(correct_by_true, true_hist),
        "tau_sweep": tau_sweep,
    }


def _train_source_only(
    model: ReceiverImpactGADNet,
    source_loader: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    epochs: int,
    device: torch.device | str,
    max_batches_per_epoch: int | None,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    history: list[dict[str, float | int]] = []
    for epoch_index in range(int(epochs)):
        model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        batches = 0
        for batch in _limited_batches(source_loader, max_batches_per_epoch):
            x = _batch_tensor(batch, "iq", 0).to(device)
            y = _batch_tensor(batch, "label", 1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(x)
            loss = F.cross_entropy(logits, y, label_smoothing=float(label_smoothing))
            loss.backward()
            optimizer.step()
            _scheduler_step(scheduler)
            loss_sum += float(loss.detach().cpu())
            acc_sum += tx_accuracy(logits.detach().cpu(), y.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("source batches cannot be empty")
        history.append(
            {
                "epoch": epoch_index + 1,
                "batches": batches,
                "loss_mean": loss_sum / float(batches),
                "source_batch_acc_mean": acc_sum / float(batches),
                "lr": _optimizer_lr(optimizer),
            }
        )
    return {"history": history}


def _task_slug(task: str) -> str:
    return str(task).replace("->", "_to_").replace("/", "_")


def _classify_reproduction_claim(
    *,
    official_compat: bool,
    official_compat_safe_pseudo: bool,
    class_prior_mode: str,
    class_weight_smoothing: float,
    class_weight_clip_min: float | None,
    class_weight_clip_max: float | None,
    class_weight_mean_normalize: bool,
    kl_estimator_mode: str,
    pseudo_threshold_mode: str,
    pseudo_score_mode: str,
    pseudo_threshold_floor: float,
    pseudo_quota_mode: str,
    class_weight_timing: str,
    pseudo_state_scope: str,
    batch_pairing: str,
    adapt_start_epoch: int,
    label_smoothing: float,
    target_model_selection: str,
    weighted_ce_reduction: str,
    lr_scheduler_mode: str,
    model_profile: str,
    learning_rate: float,
    estimate_steps: int,
    base_tau: float,
    mu: float,
    kl_weight: float,
    use_domain_alignment: bool,
    use_pseudo: bool,
    use_class_weight: bool,
    max_samples_per_combo: int | None,
    max_batches_per_epoch: int | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if str(target_model_selection).strip().lower() != "final":
        return {
            "reproduction_profile": "oracle_diagnostic",
            "claim_status": "diagnostic_only",
            "claim_reasons": ["target-label checkpoint/model selection"],
        }
    if max_samples_per_combo is not None or max_batches_per_epoch is not None:
        reasons.append("bounded smoke/truncated data path")
    ablation_active = not (bool(use_domain_alignment) and bool(use_pseudo) and bool(use_class_weight))
    diagnostic_profile = "diagnostic_extension"
    if ablation_active:
        enabled = [
            name
            for name, value in (
                ("domain_alignment", use_domain_alignment),
                ("pseudo_labeling", use_pseudo),
                ("class_weight", use_class_weight),
            )
            if bool(value)
        ]
        reasons.append("Table III component ablation: " + "+".join(enabled or ["source_only"]))
        diagnostic_profile = "paper_ablation_diagnostic"
    if str(model_profile).strip().lower() != "standard_resnet18":
        reasons.append("author-linked template architecture parameters are inferred, not published")
        if not ablation_active:
            diagnostic_profile = "architecture_hypothesis_diagnostic"
    if float(pseudo_threshold_floor) != 0.0:
        reasons.append("pseudo threshold floor is not in the paper")
    if str(pseudo_quota_mode).strip().lower() != "none":
        reasons.append("pseudo quota is not in the paper or released trainer")
    if float(class_weight_smoothing) != 0.0:
        reasons.append("class-weight smoothing is not in the paper")
    if class_weight_clip_min is not None or class_weight_clip_max is not None:
        reasons.append("class-weight clipping is not in the paper")
    if bool(class_weight_mean_normalize):
        reasons.append("class-weight mean normalization is not in the paper")
    reported_scalars = {
        "learning_rate": (float(learning_rate), 0.0006),
        "estimate_steps": (int(estimate_steps), 7),
        "base_tau": (float(base_tau), 0.7),
        "mu": (float(mu), 0.5),
        "kl_weight": (float(kl_weight), 0.005),
    }
    reasons.extend(
        f"reported paper scalar mismatch: {name}"
        for name, (actual_value, expected_value) in reported_scalars.items()
        if actual_value != expected_value
    )

    if bool(official_compat):
        expected = {
            "official_compat_safe_pseudo": False,
            "class_prior_mode": "source",
            "kl_estimator_mode": "mine_ma",
            "pseudo_threshold_mode": "official",
            "pseudo_score_mode": "logit",
            "class_weight_timing": "current",
            "pseudo_state_scope": "epoch",
            "batch_pairing": "zip_min",
            "weighted_ce_reduction": "pytorch_weighted_mean",
        }
        actual = {
            "official_compat_safe_pseudo": bool(official_compat_safe_pseudo),
            "class_prior_mode": str(class_prior_mode),
            "kl_estimator_mode": str(kl_estimator_mode),
            "pseudo_threshold_mode": str(pseudo_threshold_mode),
            "pseudo_score_mode": str(pseudo_score_mode),
            "class_weight_timing": str(class_weight_timing),
            "pseudo_state_scope": str(pseudo_state_scope),
            "batch_pairing": str(batch_pairing),
            "weighted_ce_reduction": str(weighted_ce_reduction),
        }
        reasons.extend(f"released trainer mismatch: {key}" for key, value in expected.items() if actual[key] != value)
        if reasons:
            return {
                "reproduction_profile": diagnostic_profile,
                "claim_status": "diagnostic_only",
                "claim_reasons": reasons,
            }
        return {
            "reproduction_profile": "released_trainer_semantics_bounded",
            "claim_status": "bounded_released_trainer_semantics",
            "claim_reasons": ["public trainer omits model, data, and experiment config"],
        }

    expected = {
        "class_prior_mode": "uniform",
        "kl_estimator_mode": "dvkl",
        "pseudo_threshold_mode": "paper",
        "pseudo_score_mode": "probability",
        "class_weight_timing": "previous",
        "pseudo_state_scope": "epoch",
        "batch_pairing": "zip_min",
        "adapt_start_epoch": 0,
        "label_smoothing": 0.0,
        "weighted_ce_reduction": "paper_sample_mean",
        "lr_scheduler_mode": "none",
    }
    actual = {
        "class_prior_mode": str(class_prior_mode),
        "kl_estimator_mode": str(kl_estimator_mode),
        "pseudo_threshold_mode": str(pseudo_threshold_mode),
        "pseudo_score_mode": str(pseudo_score_mode),
        "class_weight_timing": str(class_weight_timing),
        "pseudo_state_scope": str(pseudo_state_scope),
        "batch_pairing": str(batch_pairing),
        "adapt_start_epoch": int(adapt_start_epoch),
        "label_smoothing": float(label_smoothing),
        "weighted_ce_reduction": str(weighted_ce_reduction),
        "lr_scheduler_mode": str(lr_scheduler_mode),
    }
    reasons.extend(f"strict paper mismatch: {key}" for key, value in expected.items() if actual[key] != value)
    if reasons:
        return {
            "reproduction_profile": diagnostic_profile,
            "claim_status": "diagnostic_only",
            "claim_reasons": reasons,
        }
    return {
        "reproduction_profile": "paper_equations_bounded",
        "claim_status": "bounded_paper_reproduction",
        "claim_reasons": ["paper omits architecture details, optimizer, batch size, epochs, split, and stopping rule"],
    }


def run_table2_reproduction(
    compact_or_path: dict[str, Any] | str | Path,
    *,
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    output_dir: Path | str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 0.0006,
    lr_scheduler_mode: str = "none",
    lr_step_size: int = 2500,
    lr_gamma: float = 0.6,
    max_samples_per_combo: int | None = None,
    max_batches_per_epoch: int | None = None,
    source_pretrain_epochs: int | None = None,
    estimate_steps: int = 7,
    base_tau: float = 0.7,
    mu: float = 0.5,
    kl_weight: float = 0.005,
    class_prior_mode: str = "uniform",
    class_weight_smoothing: float = 0.0,
    class_weight_clip_min: float | None = None,
    class_weight_clip_max: float | None = None,
    class_weight_mean_normalize: bool = False,
    kl_estimator_mode: str = "dvkl",
    mine_ma_rate: float = 0.01,
    mine_update_scale: float = 0.5,
    pseudo_threshold_mode: str = "paper",
    pseudo_score_mode: str = "probability",
    pseudo_threshold_floor: float = 0.0,
    pseudo_quota_mode: str = "none",
    pseudo_quota_per_class: int | None = None,
    class_weight_timing: str = "previous",
    pseudo_state_scope: str = "epoch",
    batch_pairing: str = "zip_min",
    adapt_start_epoch: int = 0,
    label_smoothing: float = 0.0,
    target_model_selection: str = "final",
    weighted_ce_reduction: str = "paper_sample_mean",
    official_compat: bool = False,
    official_compat_safe_pseudo: bool = False,
    model_profile: str = "standard_resnet18",
    use_domain_alignment: bool = True,
    use_pseudo: bool = True,
    use_class_weight: bool = True,
    seed: int = 0,
    device: torch.device | str | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    set_seed(int(seed))
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if official_compat:
        if class_prior_mode == "uniform":
            class_prior_mode = "source"
        kl_estimator_mode = "mine_ma"
        if official_compat_safe_pseudo:
            pseudo_threshold_mode = "paper"
            pseudo_score_mode = "probability"
        else:
            pseudo_threshold_mode = "official"
            pseudo_score_mode = "logit"
        class_weight_timing = "current"
        pseudo_state_scope = "epoch"
        batch_pairing = "zip_min"
        weighted_ce_reduction = "pytorch_weighted_mean"
        if source_pretrain_epochs is None:
            source_pretrain_epochs = 0
    reproduction_claim = _classify_reproduction_claim(
        official_compat=official_compat,
        official_compat_safe_pseudo=official_compat_safe_pseudo,
        class_prior_mode=class_prior_mode,
        class_weight_smoothing=class_weight_smoothing,
        class_weight_clip_min=class_weight_clip_min,
        class_weight_clip_max=class_weight_clip_max,
        class_weight_mean_normalize=class_weight_mean_normalize,
        kl_estimator_mode=kl_estimator_mode,
        pseudo_threshold_mode=pseudo_threshold_mode,
        pseudo_score_mode=pseudo_score_mode,
        pseudo_threshold_floor=pseudo_threshold_floor,
        pseudo_quota_mode=pseudo_quota_mode,
        class_weight_timing=class_weight_timing,
        pseudo_state_scope=pseudo_state_scope,
        batch_pairing=batch_pairing,
        adapt_start_epoch=adapt_start_epoch,
        label_smoothing=label_smoothing,
        target_model_selection=target_model_selection,
        weighted_ce_reduction=weighted_ce_reduction,
        lr_scheduler_mode=lr_scheduler_mode,
        model_profile=model_profile,
        learning_rate=learning_rate,
        estimate_steps=estimate_steps,
        base_tau=base_tau,
        mu=mu,
        kl_weight=kl_weight,
        use_domain_alignment=use_domain_alignment,
        use_pseudo=use_pseudo,
        use_class_weight=use_class_weight,
        max_samples_per_combo=max_samples_per_combo,
        max_batches_per_epoch=max_batches_per_epoch,
    )
    requested_tasks = list(PAPER_TASKS if tasks is None else tasks)
    requested_methods = [str(method).lower() for method in (methods or ["source_only", "proposed"])]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    compact = load_wisig_compact_pkl(str(compact_or_path)) if isinstance(compact_or_path, (str, Path)) else compact_or_path
    rows: list[dict[str, Any]] = []

    for task in requested_tasks:
        loaders = build_manysig_task_loaders(
            compact,
            task=task,
            batch_size=batch_size,
            max_samples_per_combo=max_samples_per_combo,
            seed=seed,
            num_workers=num_workers,
        )
        source_class_prior = _resolve_class_prior(
            loaders["source"],
            num_classes=6,
            mode=class_prior_mode,
        )
        for method in requested_methods:
            set_seed(int(seed))
            if method not in {"source_only", "proposed"}:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "status": "not_implemented",
                        "target_labels_scope": "evaluation_only",
                    }
                )
                continue
            model = ReceiverImpactGADNet(num_tx=6, model_profile=model_profile).to(resolved_device)
            checkpoint_path = output_path / f"{_task_slug(task)}_{method}.pt"
            if method == "source_only":
                optimizer = torch.optim.Adam(
                    list(model.feature_extractor.parameters()) + list(model.classifier.parameters()),
                    lr=float(learning_rate),
                )
                scheduler = _build_lr_scheduler(
                    optimizer,
                    mode=lr_scheduler_mode,
                    step_size=lr_step_size,
                    gamma=lr_gamma,
                )
                train_result = _train_source_only(
                    model,
                    loaders["source"],
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    device=resolved_device,
                    max_batches_per_epoch=max_batches_per_epoch,
                    label_smoothing=label_smoothing,
                )
                torch.save(
                    {
                        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
                        "method": method,
                        "task": task,
                        "model_profile": str(model_profile),
                        "model_state_dict": model.inference_state_dict(),
                        "history": train_result["history"],
                    },
                    checkpoint_path,
                )
            else:
                resolved_pretrain_epochs = int(epochs if source_pretrain_epochs is None else source_pretrain_epochs)
                if resolved_pretrain_epochs < 0:
                    raise ValueError("source_pretrain_epochs must be non-negative")
                source_pretrain_result: dict[str, Any] | None = None
                if resolved_pretrain_epochs > 0:
                    source_optimizer = torch.optim.Adam(
                        list(model.feature_extractor.parameters()) + list(model.classifier.parameters()),
                        lr=float(learning_rate),
                    )
                    source_scheduler = _build_lr_scheduler(
                        source_optimizer,
                        mode=lr_scheduler_mode,
                        step_size=lr_step_size,
                        gamma=lr_gamma,
                    )
                    source_pretrain_result = _train_source_only(
                        model,
                        loaders["source"],
                        optimizer=source_optimizer,
                        scheduler=source_scheduler,
                        epochs=resolved_pretrain_epochs,
                        device=resolved_device,
                        max_batches_per_epoch=max_batches_per_epoch,
                        label_smoothing=label_smoothing,
                    )
                source_pretrain_target_audit = _audit_target_predictions(
                    model,
                    loaders["target_eval"],
                    device=resolved_device,
                    tau_values=tuple(sorted({0.7, float(base_tau)})),
                )
                optimizer_t = torch.optim.Adam(model.estimate_network.parameters(), lr=float(learning_rate))
                optimizer_ec = torch.optim.Adam(
                    list(model.feature_extractor.parameters()) + list(model.classifier.parameters()),
                    lr=float(learning_rate),
                )
                scheduler_t = _build_lr_scheduler(
                    optimizer_t,
                    mode=lr_scheduler_mode,
                    step_size=lr_step_size,
                    gamma=lr_gamma,
                )
                scheduler_ec = _build_lr_scheduler(
                    optimizer_ec,
                    mode=lr_scheduler_mode,
                    step_size=lr_step_size,
                    gamma=lr_gamma,
                )
                train_result = run_gada_training_loop(
                    model,
                    loaders["source"],
                    loaders["target_train"],
                    optimizer_t=optimizer_t,
                    optimizer_ec=optimizer_ec,
                    scheduler_t=scheduler_t,
                    scheduler_ec=scheduler_ec,
                    epochs=epochs,
                    checkpoint_path=None,
                    device=resolved_device,
                    estimate_steps=estimate_steps,
                    base_tau=base_tau,
                    mu=mu,
                    kl_weight=kl_weight,
                    class_prior=None if source_class_prior is None else source_class_prior.to(resolved_device),
                    class_weight_smoothing=class_weight_smoothing,
                    class_weight_clip_min=class_weight_clip_min,
                    class_weight_clip_max=class_weight_clip_max,
                    class_weight_mean_normalize=class_weight_mean_normalize,
                    kl_estimator_mode=kl_estimator_mode,
                    mine_ma_rate=mine_ma_rate,
                    mine_update_scale=mine_update_scale,
                    pseudo_threshold_mode=pseudo_threshold_mode,
                    pseudo_score_mode=pseudo_score_mode,
                    pseudo_threshold_floor=pseudo_threshold_floor,
                    pseudo_quota_mode=pseudo_quota_mode,
                    pseudo_quota_per_class=pseudo_quota_per_class,
                    class_weight_timing=class_weight_timing,
                    pseudo_state_scope=pseudo_state_scope,
                    batch_pairing=batch_pairing,
                    adapt_start_epoch=adapt_start_epoch,
                    label_smoothing=label_smoothing,
                    target_eval_batches=loaders["target_eval"] if target_model_selection == "target_loss_best" else None,
                    target_model_selection=target_model_selection,
                    weighted_ce_reduction=weighted_ce_reduction,
                    use_domain_alignment=use_domain_alignment,
                    use_pseudo=use_pseudo,
                    use_class_weight=use_class_weight,
                    max_batches_per_epoch=max_batches_per_epoch,
                )
                if source_pretrain_result is not None:
                    train_result["source_pretrain_history"] = source_pretrain_result["history"]
                train_result["source_pretrain_target_audit"] = source_pretrain_target_audit
                torch.save(
                    {
                        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
                        "method": method,
                        "task": task,
                        "model_state_dict": model.inference_state_dict(),
                        "history": train_result["history"],
                        "source_pretrain_history": train_result.get("source_pretrain_history", []),
                        "source_pretrain_target_audit": source_pretrain_target_audit,
                        "adaptation": {
                            "algorithm": train_result["algorithm"],
                            "epochs": train_result["epochs"],
                            "estimate_steps": train_result["estimate_steps"],
                            "base_tau": train_result["base_tau"],
                            "mu": train_result["mu"],
                            "kl_weight": train_result["kl_weight"],
                            "class_prior_mode": class_prior_mode,
                            "class_prior": None
                            if source_class_prior is None
                            else [float(v) for v in source_class_prior.tolist()],
                            "class_weight_smoothing": train_result["class_weight_smoothing"],
                            "class_weight_clip_min": train_result["class_weight_clip_min"],
                            "class_weight_clip_max": train_result["class_weight_clip_max"],
                            "class_weight_mean_normalize": train_result["class_weight_mean_normalize"],
                            "kl_estimator_mode": train_result["kl_estimator_mode"],
                            "mine_ma_rate": train_result["mine_ma_rate"],
                            "mine_update_scale": train_result["mine_update_scale"],
                            "pseudo_threshold_mode": train_result["pseudo_threshold_mode"],
                            "pseudo_score_mode": train_result["pseudo_score_mode"],
                            "pseudo_threshold_floor": train_result["pseudo_threshold_floor"],
                            "pseudo_quota_mode": train_result["pseudo_quota_mode"],
                            "pseudo_quota_per_class": train_result["pseudo_quota_per_class"],
                            "class_weight_timing": train_result["class_weight_timing"],
                            "pseudo_state_scope": train_result["pseudo_state_scope"],
                            "batch_pairing": train_result["batch_pairing"],
                            "adapt_start_epoch": train_result["adapt_start_epoch"],
                            "label_smoothing": train_result["label_smoothing"],
                            "lr_scheduler_mode": lr_scheduler_mode,
                            "lr_step_size": int(lr_step_size),
                            "lr_gamma": float(lr_gamma),
                            "lr_scheduler_active": train_result["lr_scheduler_active"],
                            "target_model_selection": train_result["target_model_selection"],
                            "weighted_ce_reduction": train_result["weighted_ce_reduction"],
                            "use_domain_alignment": train_result["use_domain_alignment"],
                            "use_pseudo": train_result["use_pseudo"],
                            "use_class_weight": train_result["use_class_weight"],
                            "target_eval_history": train_result["target_eval_history"],
                            "best_target_loss_epoch": train_result["best_target_loss_epoch"],
                            "best_target_loss": train_result["best_target_loss"],
                            "state": train_result["state"],
                        },
                    },
                    checkpoint_path,
                )
                train_result["checkpoint_path"] = str(checkpoint_path)
            target_metrics = _evaluate_target_metrics(model, loaders["target_eval"], device=resolved_device, include_loss=False)
            row_claim = (
                reproduction_claim
                if method == "proposed"
                else {
                    "reproduction_profile": "source_only_bounded",
                    "claim_status": "bounded_source_only_baseline",
                    "claim_reasons": ["paper omits optimizer, batch size, epochs, split, and stopping rule"],
                }
            )
            row = {
                "task": task,
                "method": method,
                "status": "completed_diagnostic_only"
                if row_claim["claim_status"] == "diagnostic_only"
                else "completed",
                "execution_status": "completed",
                **row_claim,
                "target_accuracy": float(target_metrics["target_accuracy"]),
                "target_true_hist": target_metrics["target_true_hist"],
                "target_pred_hist": target_metrics["target_pred_hist"],
                "target_correct_by_class": target_metrics["target_correct_by_class"],
                "target_total_by_class": target_metrics["target_total_by_class"],
                "target_accuracy_by_class": target_metrics["target_accuracy_by_class"],
                "target_confusion_matrix": target_metrics["target_confusion_matrix"],
                "target_labels_scope": "evaluation_only",
                "target_label_role": loaders["meta"]["target_label_role"],
                "model_profile": str(model_profile),
                "mu": float(mu),
                "kl_weight": float(kl_weight),
                "use_domain_alignment": bool(use_domain_alignment),
                "use_pseudo": bool(use_pseudo),
                "use_class_weight": bool(use_class_weight),
                "checkpoint_path": str(checkpoint_path),
                "history": train_result["history"],
                "task_meta": loaders["meta"],
            }
            if method == "proposed":
                row["source_pretrain_history"] = train_result.get("source_pretrain_history", [])
                row["source_pretrain_target_audit"] = train_result.get("source_pretrain_target_audit", {})
                row["class_prior_mode"] = class_prior_mode
                row["class_prior"] = None if source_class_prior is None else [float(v) for v in source_class_prior.tolist()]
                row["official_compat"] = bool(official_compat)
                row["official_compat_safe_pseudo"] = bool(official_compat_safe_pseudo)
                row["kl_estimator_mode"] = train_result.get("kl_estimator_mode")
                row["pseudo_threshold_mode"] = train_result.get("pseudo_threshold_mode")
                row["pseudo_score_mode"] = train_result.get("pseudo_score_mode")
                row["pseudo_threshold_floor"] = train_result.get("pseudo_threshold_floor")
                row["pseudo_quota_mode"] = train_result.get("pseudo_quota_mode")
                row["pseudo_quota_per_class"] = train_result.get("pseudo_quota_per_class")
                row["class_weight_timing"] = train_result.get("class_weight_timing")
                row["pseudo_state_scope"] = train_result.get("pseudo_state_scope")
                row["batch_pairing"] = train_result.get("batch_pairing")
                row["target_model_selection"] = train_result.get("target_model_selection")
                row["weighted_ce_reduction"] = train_result.get("weighted_ce_reduction")
                row["lr_scheduler_mode"] = lr_scheduler_mode
                row["lr_step_size"] = int(lr_step_size)
                row["lr_gamma"] = float(lr_gamma)
                row["lr_scheduler_active"] = train_result.get("lr_scheduler_active")
                row["target_eval_history"] = train_result.get("target_eval_history", [])
                row["best_target_loss_epoch"] = train_result.get("best_target_loss_epoch")
                row["best_target_loss"] = train_result.get("best_target_loss")
            rows.append(row)

    return {
        "method_id": "mitigating_receiver_impact_da",
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "artifact_type": "table2_reproduction_run",
        "dataset": "WiSig ManySig",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "lr_scheduler_mode": str(lr_scheduler_mode),
        "lr_step_size": int(lr_step_size),
        "lr_gamma": float(lr_gamma),
        "max_samples_per_combo": max_samples_per_combo,
        "max_batches_per_epoch": max_batches_per_epoch,
        "source_pretrain_epochs": int(epochs if source_pretrain_epochs is None else source_pretrain_epochs),
        "estimate_steps": int(estimate_steps),
        "base_tau": float(base_tau),
        "mu": float(mu),
        "kl_weight": float(kl_weight),
        "class_prior_mode": class_prior_mode,
        "class_weight_smoothing": float(class_weight_smoothing),
        "class_weight_clip_min": None if class_weight_clip_min is None else float(class_weight_clip_min),
        "class_weight_clip_max": None if class_weight_clip_max is None else float(class_weight_clip_max),
        "class_weight_mean_normalize": bool(class_weight_mean_normalize),
        "kl_estimator_mode": str(kl_estimator_mode),
        "mine_ma_rate": float(mine_ma_rate),
        "mine_update_scale": float(mine_update_scale),
        "pseudo_threshold_mode": str(pseudo_threshold_mode),
        "pseudo_score_mode": str(pseudo_score_mode),
        "pseudo_threshold_floor": float(pseudo_threshold_floor),
        "pseudo_quota_mode": str(pseudo_quota_mode),
        "pseudo_quota_per_class": None if pseudo_quota_per_class is None else int(pseudo_quota_per_class),
        "class_weight_timing": str(class_weight_timing),
        "pseudo_state_scope": str(pseudo_state_scope),
        "batch_pairing": str(batch_pairing),
        "adapt_start_epoch": int(adapt_start_epoch),
        "label_smoothing": float(label_smoothing),
        "target_model_selection": str(target_model_selection),
        "weighted_ce_reduction": str(weighted_ce_reduction),
        "official_compat": bool(official_compat),
        "official_compat_safe_pseudo": bool(official_compat_safe_pseudo),
        "model_profile": str(model_profile),
        "use_domain_alignment": bool(use_domain_alignment),
        "use_pseudo": bool(use_pseudo),
        "use_class_weight": bool(use_class_weight),
        "seed": int(seed),
        "device": str(resolved_device),
        **reproduction_claim,
        "result_claim_status": reproduction_claim["claim_status"],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful dry-run entrypoint for the IoTJ 2024 receiver-impact DA RFFI paper.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the reproduction matrix.")
    parser.add_argument("--run-table2", action="store_true", help="Run Table II source-only/proposed rows on a WiSig ManySig pkl.")
    parser.add_argument("--manysig-pkl", type=Path, default=None)
    parser.add_argument("--methods", type=str, default="source_only,proposed")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--lr-scheduler-mode", type=str, default="none", choices=("none", "step"))
    parser.add_argument("--lr-step-size", type=int, default=2500)
    parser.add_argument("--lr-gamma", type=float, default=0.6)
    parser.add_argument("--max-samples-per-combo", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--source-pretrain-epochs", type=int, default=None)
    parser.add_argument("--estimate-steps", type=int, default=7)
    parser.add_argument("--base-tau", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.005)
    parser.add_argument("--class-prior-mode", type=str, default="uniform", choices=("source", "uniform", "none"))
    parser.add_argument("--class-weight-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight-clip-min", type=float, default=None)
    parser.add_argument("--class-weight-clip-max", type=float, default=None)
    parser.add_argument("--class-weight-mean-normalize", action="store_true")
    parser.add_argument("--kl-estimator-mode", type=str, default="dvkl", choices=("dvkl", "mine_ma"))
    parser.add_argument("--mine-ma-rate", type=float, default=0.01)
    parser.add_argument("--mine-update-scale", type=float, default=0.5)
    parser.add_argument("--pseudo-threshold-mode", type=str, default="paper", choices=("paper", "official"))
    parser.add_argument("--pseudo-score-mode", type=str, default="probability", choices=("probability", "logit"))
    parser.add_argument("--pseudo-threshold-floor", type=float, default=0.0)
    parser.add_argument("--pseudo-quota-mode", type=str, default="none", choices=("none", "balanced_topk"))
    parser.add_argument("--pseudo-quota-per-class", type=int, default=None)
    parser.add_argument("--class-weight-timing", type=str, default="previous", choices=("previous", "current"))
    parser.add_argument("--pseudo-state-scope", type=str, default="epoch", choices=("global", "epoch"))
    parser.add_argument("--batch-pairing", type=str, default="zip_min", choices=("cycle_target", "zip_min"))
    parser.add_argument("--adapt-start-epoch", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--target-model-selection", type=str, default="final", choices=("final", "target_loss_best"))
    parser.add_argument(
        "--weighted-ce-reduction",
        type=str,
        default="paper_sample_mean",
        choices=("paper_sample_mean", "pytorch_weighted_mean"),
    )
    parser.add_argument("--official-compat", action="store_true", help="Use details exposed by the released official trainer.")
    parser.add_argument(
        "--official-compat-safe-pseudo",
        action="store_true",
        help="With --official-compat, keep the official optimizer/state path but use paper CPL probabilities instead of raw-logit pseudo-label gating.",
    )
    parser.add_argument(
        "--model-profile",
        type=str,
        default="standard_resnet18",
        choices=("standard_resnet18", "pytorch_template_resnet18_hypothesis_v1"),
    )
    parser.add_argument("--no-domain-alignment", action="store_true")
    parser.add_argument("--no-pseudo", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("paper_reproduction/runs/mitigating_receiver_impact_da"))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for dry-run payload.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.run_table2:
        if args.manysig_pkl is None:
            raise SystemExit("--manysig-pkl is required with --run-table2")
        tasks = [token.strip() for token in args.tasks.split(",") if token.strip()] or None
        methods = [token.strip() for token in args.methods.split(",") if token.strip()]
        payload = run_table2_reproduction(
            args.manysig_pkl,
            tasks=tasks,
            methods=methods,
            output_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_mode=args.lr_scheduler_mode,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            max_samples_per_combo=args.max_samples_per_combo,
            max_batches_per_epoch=args.max_batches_per_epoch,
            source_pretrain_epochs=args.source_pretrain_epochs,
            estimate_steps=args.estimate_steps,
            base_tau=args.base_tau,
            mu=args.mu,
            kl_weight=args.kl_weight,
            class_prior_mode=args.class_prior_mode,
            class_weight_smoothing=args.class_weight_smoothing,
            class_weight_clip_min=args.class_weight_clip_min,
            class_weight_clip_max=args.class_weight_clip_max,
            class_weight_mean_normalize=args.class_weight_mean_normalize,
            kl_estimator_mode=args.kl_estimator_mode,
            mine_ma_rate=args.mine_ma_rate,
            mine_update_scale=args.mine_update_scale,
            pseudo_threshold_mode=args.pseudo_threshold_mode,
            pseudo_score_mode=args.pseudo_score_mode,
            pseudo_threshold_floor=args.pseudo_threshold_floor,
            pseudo_quota_mode=args.pseudo_quota_mode,
            pseudo_quota_per_class=args.pseudo_quota_per_class,
            class_weight_timing=args.class_weight_timing,
            pseudo_state_scope=args.pseudo_state_scope,
            batch_pairing=args.batch_pairing,
            adapt_start_epoch=args.adapt_start_epoch,
            label_smoothing=args.label_smoothing,
            target_model_selection=args.target_model_selection,
            weighted_ce_reduction=args.weighted_ce_reduction,
            official_compat=args.official_compat,
            official_compat_safe_pseudo=args.official_compat_safe_pseudo,
            model_profile=args.model_profile,
            use_domain_alignment=not args.no_domain_alignment,
            use_pseudo=not args.no_pseudo,
            use_class_weight=not args.no_class_weight,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    if not args.dry_run:
        raise SystemExit("formal WiSig training CLI is intentionally gated; use --dry-run or --run-table2")

    payload = build_dry_run_payload(config)
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
