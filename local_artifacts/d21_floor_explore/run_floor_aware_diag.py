"""Floor-aware 256-parameter support-only adaptation on a sealed K10 capsule.

``predict`` never opens query truth.  It derives the fixed 256-D representation
from the single sealed LEO_weak observation, fits diagonal metrics from labelled
support only, and emits immutable-style predictions.  ``score`` is the only
command that opens the truth sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
from cvsrffi.stage2_diag_cosine_exploration import rf_statistics, spectral_logmag_sketch


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
CONFIGS: dict[str, dict[str, float]] = {
    "L6_diag_floor": {
        "lr": 0.04,
        "hard_strength": 2.0,
        "hard_power": 1.5,
        "old_weight": 1.0,
        "distill": 0.0,
        "intrusion": 0.0,
        "intrusion_margin": 0.02,
        "class_cvar": 0.25,
        "pairwise": 0.0,
        "regularizer": 0.01,
    },
    "L6_diag_floor_distill": {
        "lr": 0.035,
        "hard_strength": 2.5,
        "hard_power": 2.0,
        "old_weight": 1.25,
        "distill": 0.5,
        "intrusion": 1.0,
        "intrusion_margin": 0.02,
        "class_cvar": 0.4,
        "pairwise": 0.25,
        "regularizer": 0.02,
    },
    "L6_diag_floor_distill_strong": {
        "lr": 0.03,
        "hard_strength": 3.0,
        "hard_power": 2.0,
        "old_weight": 1.5,
        "distill": 1.0,
        "intrusion": 2.0,
        "intrusion_margin": 0.03,
        "class_cvar": 0.6,
        "pairwise": 0.5,
        "regularizer": 0.03,
    },
}
REPRESENTATION_FAMILIES = (
    "A0_z_fft96",
    "A1_z_fft96_rf32",
    "A2_z_fft96_dp",
    "A3_z_fft96_rf32_dp",
)
METHODS = (
    *REPRESENTATION_FAMILIES,
    *CONFIGS,
    "D256_L5_L6_blend",
    "S_selected_descriptor_floor_safe",
)
EPOCHS = 20
DIM = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalise(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)


def _fixed_representation(z_id: np.ndarray, iq: np.ndarray) -> np.ndarray:
    z = _normalise(z_id)
    fft = _normalise(spectral_logmag_sketch(iq, dim=96))
    return _normalise(np.concatenate([z, 8.0 * fft], axis=1))


def _differential_phase_descriptor(iq: np.ndarray, dim: int) -> np.ndarray:
    """Robust adjacent differential-phase statistics plus a fixed histogram."""
    rows = np.asarray(iq, dtype=np.float32)
    if rows.ndim != 3 or rows.shape[1] != 2 or dim not in {16, 32} or not np.isfinite(rows).all():
        raise RuntimeError("DP descriptor expects finite [N,2,T] IQ and dim in {16,32}")
    result = []
    for row in rows:
        value = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        phase_delta = np.angle(value[1:] * np.conj(value[:-1]))
        phase_mean = float(np.angle(np.mean(np.exp(1j * phase_delta))))
        phase_residual = np.angle(np.exp(1j * (phase_delta - phase_mean)))
        resultant = complex(np.mean(np.exp(1j * phase_delta)))
        histogram, _ = np.histogram(
            phase_residual, bins=dim - 4, range=(-math.pi, math.pi), density=False
        )
        values = np.asarray(
            [
            phase_mean,
            float(abs(resultant)),
            float(np.std(phase_residual)),
            float(np.median(np.abs(phase_residual - np.median(phase_residual)))),
            *[float(v) / max(float(phase_residual.size), 1.0) for v in histogram],
            ],
            dtype=np.float32,
        )
        descriptor = values
        if descriptor.shape != (dim,) or not np.isfinite(descriptor).all():
            raise RuntimeError(f"DP descriptor drift: {descriptor.shape}")
        result.append(descriptor / max(float(np.linalg.norm(descriptor)), 1e-8))
    return np.stack(result).astype(np.float32)


def _branch_representation(
    family: str,
    z_id: np.ndarray,
    fft: np.ndarray,
    rf: np.ndarray,
    dp16: np.ndarray,
    dp32: np.ndarray,
    config: dict[str, float],
) -> np.ndarray:
    z = _normalise(z_id)
    fft_rows = config["fft_weight"] * _normalise(fft)
    if family == "A0_z_fft96":
        return _normalise(np.concatenate([z, fft_rows], axis=1))
    if family == "A1_z_fft96_rf32":
        return _normalise(
            np.concatenate(
                [z, fft_rows, config["rf_weight"] * _normalise(rf)],
                axis=1,
            )
        )
    dp = dp16 if int(config.get("dp_dim", 16)) == 16 else dp32
    if family == "A2_z_fft96_dp":
        return _normalise(
            np.concatenate(
                [z, fft_rows, config["dp_weight"] * _normalise(dp)], axis=1
            )
        )
    if family == "A3_z_fft96_rf32_dp":
        return _normalise(
            np.concatenate(
                [
                    z,
                    fft_rows,
                    config["rf_weight"] * _normalise(rf),
                    config["dp_weight"] * _normalise(dp),
                ],
                axis=1,
            )
        )
    raise KeyError(family)


def _branch_grid(family: str) -> list[dict[str, float]]:
    if family == "A0_z_fft96":
        return [{"fft_weight": 8.0, "rf_weight": 0.0, "dp_weight": 0.0, "dp_dim": 16.0}]
    if family == "A1_z_fft96_rf32":
        return [
            {"fft_weight": fft_weight, "rf_weight": rf_weight, "dp_weight": 0.0, "dp_dim": 16.0}
            for fft_weight in (4.0, 8.0)
            for rf_weight in (1.0, 2.0, 4.0)
        ]
    if family == "A2_z_fft96_dp":
        return [
            {"fft_weight": fft_weight, "rf_weight": 0.0, "dp_weight": dp_weight, "dp_dim": float(dp_dim)}
            for fft_weight in (4.0, 8.0)
            for dp_weight in (1.0, 2.0, 4.0)
            for dp_dim in (16, 32)
        ]
    if family == "A3_z_fft96_rf32_dp":
        return [
            {
                "fft_weight": fft_weight,
                "rf_weight": rf_weight,
                "dp_weight": dp_weight,
                "dp_dim": float(dp_dim),
            }
            for fft_weight in (4.0, 8.0)
            for rf_weight in (1.0, 2.0)
            for dp_weight in (1.0, 2.0)
            for dp_dim in (16, 32)
        ]
    raise KeyError(family)


def _int8_support_roundtrip(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(rows), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, 1e-8).astype(np.float16)
    codes = np.clip(np.rint(rows / scales.astype(np.float32)), -127, 127).astype(np.int8)
    restored = _normalise(codes.astype(np.float32) * scales.astype(np.float32))
    return restored, codes, scales[:, 0]


def _extract(runtime: torch.jit.ScriptModule, iq: np.ndarray) -> tuple[np.ndarray, float]:
    rows = torch.from_numpy(np.asarray(iq, dtype=np.float32)).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        z_id, _ = runtime(rows)
    torch.cuda.synchronize()
    return z_id.detach().cpu().numpy(), time.perf_counter() - start


def _top1_scores(query: np.ndarray, support: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    similarities = query @ support.T
    return np.stack(
        [np.max(similarities[:, labels == class_index], axis=1) for class_index in range(class_count)],
        axis=1,
    ).astype(np.float32)


def _loo_scores_np(rows: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    similarities = rows @ rows.T
    np.fill_diagonal(similarities, -np.inf)
    return np.stack(
        [np.max(similarities[:, labels == class_index], axis=1) for class_index in range(class_count)],
        axis=1,
    ).astype(np.float32)


def _weighted_rows(rows: np.ndarray, log_weight: np.ndarray) -> np.ndarray:
    return _normalise(rows * np.exp(np.asarray(log_weight, dtype=np.float32))[None, :])


def _torch_loo_scores(rows: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    similarities = rows @ rows.T
    similarities = similarities.masked_fill(torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool), -1e4)
    return torch.stack(
        [similarities[:, labels == class_index].max(dim=1).values for class_index in range(class_count)],
        dim=1,
    )


def _metrics(pred: np.ndarray, truth: np.ndarray, old_count: int) -> dict[str, float]:
    per_class = []
    for class_index in sorted(set(int(value) for value in truth.tolist())):
        mask = truth == class_index
        per_class.append((class_index, float(np.mean(pred[mask] == truth[mask]))))
    old = [value for index, value in per_class if index < old_count]
    new = [value for index, value in per_class if index >= old_count]
    old_acc = float(np.mean(pred[truth < old_count] == truth[truth < old_count]))
    new_acc = float(np.mean(pred[truth >= old_count] == truth[truth >= old_count])) if new else 0.0
    harmonic = 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-12) if new else old_acc
    return {
        "accuracy": float(np.mean(pred == truth)),
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "old_floor": min(old),
        "new_floor": min(new) if new else 0.0,
        "joint_floor": min(old + new),
        "H_old_new": harmonic,
    }


def _class_weights(base_scores: np.ndarray, labels: np.ndarray, old_count: int, cfg: dict[str, float]) -> np.ndarray:
    pred = base_scores.argmax(axis=1)
    recalls = np.asarray(
        [float(np.mean(pred[labels == class_index] == class_index)) for class_index in range(base_scores.shape[1])],
        dtype=np.float32,
    )
    weights = np.power(1.0 + cfg["hard_strength"] * (1.0 - recalls), cfg["hard_power"])
    weights[:old_count] *= cfg["old_weight"]
    return weights / max(float(np.mean(weights)), 1e-8)


def _fit_metric(
    support: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    old_count: int,
    cfg: dict[str, float],
    *,
    scenario: str,
    phase: str,
    teacher_old_log_weight: np.ndarray | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if support.ndim != 2:
        raise RuntimeError("metric support must be a 2-D representation")
    base_scores = _loo_scores_np(support, labels, class_count)
    class_weights = _class_weights(base_scores, labels, old_count, cfg)
    x = torch.from_numpy(support).float().cuda()
    y = torch.from_numpy(labels.astype(np.int64)).long().cuda()
    initial_weight = (
        np.asarray(teacher_old_log_weight, dtype=np.float32)
        if teacher_old_log_weight is not None
        else np.zeros(support.shape[1], dtype=np.float32)
    )
    weight = torch.nn.Parameter(torch.from_numpy(initial_weight).cuda())
    optimiser = torch.optim.Adam([weight], lr=cfg["lr"])
    teacher_logits = None
    teacher_old_pairwise = None
    old_rows = y < old_count
    if teacher_old_log_weight is not None:
        teacher_x = F.normalize(x[old_rows] * torch.from_numpy(np.exp(teacher_old_log_weight)).float().cuda(), dim=1)
        teacher_y = y[old_rows]
        with torch.no_grad():
            teacher_logits = _torch_loo_scores(teacher_x, teacher_y, old_count)
            teacher_old_pairwise = teacher_x @ teacher_x.T
    trace: list[dict[str, Any]] = []
    sample_weight = torch.from_numpy(class_weights).float().cuda()[y]
    for epoch in range(1, EPOCHS + 1):
        optimiser.zero_grad(set_to_none=True)
        adapted = F.normalize(x * torch.exp(weight).unsqueeze(0), dim=1)
        logits = _torch_loo_scores(adapted, y, class_count)
        ce_rows = F.cross_entropy(20.0 * logits, y, reduction="none")
        ce = (ce_rows * sample_weight).sum() / sample_weight.sum()
        class_losses = torch.stack([ce_rows[y == c].mean() for c in range(class_count)])
        cvar_count = max(1, int(math.ceil(0.3 * class_count)))
        class_cvar = torch.topk(class_losses, k=cvar_count).values.mean()
        distill = torch.zeros((), device="cuda")
        if teacher_logits is not None and cfg["distill"] > 0:
            temperature = 2.0
            student_log_prob = F.log_softmax(20.0 * logits[old_rows, :old_count] / temperature, dim=1)
            teacher_prob = F.softmax(20.0 * teacher_logits / temperature, dim=1)
            distill = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2
        intrusion = torch.zeros((), device="cuda")
        if class_count > old_count and cfg["intrusion"] > 0:
            old_logits = logits[old_rows]
            old_truth = y[old_rows]
            true_old = old_logits.gather(1, old_truth[:, None]).squeeze(1)
            strongest_new = old_logits[:, old_count:].max(dim=1).values
            intrusion = F.relu(strongest_new - true_old + cfg["intrusion_margin"]).mean()
        pairwise = torch.zeros((), device="cuda")
        if teacher_old_pairwise is not None and cfg["pairwise"] > 0:
            student_old_pairwise = adapted[old_rows] @ adapted[old_rows].T
            pairwise = F.smooth_l1_loss(student_old_pairwise, teacher_old_pairwise)
        regularizer = weight.square().mean()
        total = (
            ce
            + cfg["class_cvar"] * class_cvar
            + cfg["distill"] * distill
            + cfg["pairwise"] * pairwise
            + cfg["intrusion"] * intrusion
            + cfg["regularizer"] * regularizer
        )
        total.backward()
        optimiser.step()
        with torch.no_grad():
            weight.clamp_(-0.8, 0.8)
            pred = logits.argmax(dim=1)
            per_class = [float((pred[y == c] == c).float().mean().item()) for c in range(class_count)]
            old_acc = float((pred[old_rows] == y[old_rows]).float().mean().item())
            new_mask = ~old_rows
            new_acc = float((pred[new_mask] == y[new_mask]).float().mean().item()) if bool(new_mask.any()) else 0.0
        row = {
            "scenario": scenario,
            "phase": phase,
            "epoch": epoch,
            "total_loss": float(total.item()),
            "weighted_ce": float(ce.item()),
            "class_cvar_top30": float(class_cvar.item()),
            "old_distill_kl": float(distill.item()),
            "old_pairwise_preservation": float(pairwise.item()),
            "old_new_intrusion_hinge": float(intrusion.item()),
            "log_weight_l2": float(regularizer.item()),
            "support_loo_old_acc": old_acc,
            "support_loo_seen_new_acc": new_acc,
            "support_loo_old_floor": min(per_class[:old_count]),
            "support_loo_joint_floor": min(per_class),
        }
        trace.append(row)
        print("[LOSS] " + json.dumps(row, sort_keys=True), flush=True)
    return weight.detach().cpu().numpy().astype(np.float32), trace


def _selection_metrics(
    scene_states: list[dict[str, Any]],
    config_name: str,
    alpha: float,
    old_count: int,
) -> dict[str, Any]:
    rows = []
    pooled_before_pred = []
    pooled_before_truth = []
    pooled_after_pred = []
    pooled_after_truth = []
    for state in scene_states:
        base_after = state["base_after_loo"]
        diag_after = state["fits"][config_name]["after_loo"]
        base_before = state["base_before_loo"]
        diag_before = state["fits"][config_name]["before_loo"]
        after_pred = ((1.0 - alpha) * base_after + alpha * diag_after).argmax(axis=1)
        before_pred = ((1.0 - alpha) * base_before + alpha * diag_before).argmax(axis=1)
        after_metrics = _metrics(after_pred, state["labels"], old_count)
        before_acc = float(np.mean(before_pred == state["old_labels"]))
        rows.append(
            {
                "scenario": state["scenario"],
                **after_metrics,
                "old_acc_before_increment": before_acc,
                "average_forgetting": before_acc - after_metrics["old_acc"],
            }
        )
        pooled_before_pred.append(before_pred)
        pooled_before_truth.append(state["old_labels"])
        pooled_after_pred.append(after_pred)
        pooled_after_truth.append(state["labels"])
    pooled_after = _metrics(np.concatenate(pooled_after_pred), np.concatenate(pooled_after_truth), old_count)
    before_acc = float(np.mean(np.concatenate(pooled_before_pred) == np.concatenate(pooled_before_truth)))
    return {
        "config": config_name,
        "alpha": alpha,
        "scenario_rows": rows,
        "aggregate": {
            **pooled_after,
            "old_acc_before_increment": before_acc,
            "average_forgetting": before_acc - pooled_after["old_acc"],
        },
        "worst_scene_old_floor": min(row["old_floor"] for row in rows),
        "worst_scene_joint_floor": min(row["joint_floor"] for row in rows),
        "worst_scene_h": min(row["H_old_new"] for row in rows),
        "worst_scene_forgetting": max(row["average_forgetting"] for row in rows),
    }


def _select_blend(scene_states: list[dict[str, Any]], old_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations = [
        _selection_metrics(scene_states, config_name, alpha, old_count)
        for config_name in CONFIGS
        for alpha in (0.25, 0.5, 0.75, 1.0)
    ]
    selected = max(
        evaluations,
        key=lambda row: (
            row["worst_scene_old_floor"],
            row["worst_scene_joint_floor"],
            row["worst_scene_h"],
            -max(row["worst_scene_forgetting"], 0.0),
            row["aggregate"]["H_old_new"],
            -abs(row["alpha"] - 0.5),
        ),
    )
    return selected, evaluations


def _representation_selection_row(
    scene_states: list[dict[str, Any]],
    family: str,
    config: dict[str, float],
    old_count: int,
) -> dict[str, Any]:
    scenario_rows = []
    pooled_before_pred = []
    pooled_before_truth = []
    pooled_after_pred = []
    pooled_after_truth = []
    for state in scene_states:
        support = _branch_representation(
            family,
            state["support_z"],
            state["support_fft"],
            state["support_rf"],
            state["support_dp16"],
            state["support_dp32"],
            config,
        )
        support, _, _ = _int8_support_roundtrip(support)
        old_support = support[state["labels"] < old_count]
        after_pred = _loo_scores_np(support, state["labels"], len(np.unique(state["labels"]))).argmax(axis=1)
        before_pred = _loo_scores_np(old_support, state["old_labels"], old_count).argmax(axis=1)
        metrics = _metrics(after_pred, state["labels"], old_count)
        before_acc = float(np.mean(before_pred == state["old_labels"]))
        scenario_rows.append(
            {
                "scenario": state["scenario"],
                **metrics,
                "old_acc_before_increment": before_acc,
                "average_forgetting": before_acc - metrics["old_acc"],
            }
        )
        pooled_before_pred.append(before_pred)
        pooled_before_truth.append(state["old_labels"])
        pooled_after_pred.append(after_pred)
        pooled_after_truth.append(state["labels"])
    pooled_after = _metrics(np.concatenate(pooled_after_pred), np.concatenate(pooled_after_truth), old_count)
    before_acc = float(np.mean(np.concatenate(pooled_before_pred) == np.concatenate(pooled_before_truth)))
    return {
        "family": family,
        "config": config,
        "scenario_rows": scenario_rows,
        "aggregate": {
            **pooled_after,
            "old_acc_before_increment": before_acc,
            "average_forgetting": before_acc - pooled_after["old_acc"],
        },
        "worst_scene_old_floor": min(row["old_floor"] for row in scenario_rows),
        "worst_scene_new_floor": min(row["new_floor"] for row in scenario_rows),
        "worst_scene_joint_floor": min(row["joint_floor"] for row in scenario_rows),
        "worst_scene_h": min(row["H_old_new"] for row in scenario_rows),
        "worst_scene_forgetting": max(row["average_forgetting"] for row in scenario_rows),
    }


def _select_representations(
    scene_states: list[dict[str, Any]], old_count: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    evaluations = [
        _representation_selection_row(scene_states, family, config, old_count)
        for family in REPRESENTATION_FAMILIES
        for config in _branch_grid(family)
    ]
    rank = lambda row: (
        row["worst_scene_old_floor"],
        row["worst_scene_new_floor"],
        row["worst_scene_joint_floor"],
        row["worst_scene_h"],
        -max(row["worst_scene_forgetting"], 0.0),
        row["aggregate"]["H_old_new"],
    )
    family_locks = {
        family: max((row for row in evaluations if row["family"] == family), key=rank)
        for family in REPRESENTATION_FAMILIES
    }
    return family_locks, max(family_locks.values(), key=rank), evaluations


def _representation_dim(lock: dict[str, Any]) -> int:
    family = str(lock["family"])
    dp_dim = int(lock["config"].get("dp_dim", 0))
    if family == "A0_z_fft96":
        return 256
    if family == "A1_z_fft96_rf32":
        return 288
    if family == "A2_z_fft96_dp":
        return 256 + dp_dim
    if family == "A3_z_fft96_rf32_dp":
        return 288 + dp_dim
    raise KeyError(family)


def _classifier_latency(
    query: np.ndarray,
    support: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    log_weight: np.ndarray | None,
    alpha: float,
    repeats: int = 200,
) -> tuple[float, float]:
    weighted_support = _weighted_rows(support, log_weight) if log_weight is not None else support
    durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        base = _top1_scores(query, support, labels, class_count)
        if log_weight is not None:
            diag = _top1_scores(_weighted_rows(query, log_weight), weighted_support, labels, class_count)
            scores = (1.0 - alpha) * base + alpha * diag
        else:
            scores = base
        scores.argmax(axis=1)
        durations.append((time.perf_counter_ns() - start) / 1e6 / query.shape[0])
    return float(np.mean(durations)), float(np.quantile(durations, 0.95))


def predict(capsule: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    after_enrollment = capsule / "predictor" / "after" / "enrollment_only"
    after_apply = capsule / "predictor" / "after" / "apply_only_staging"
    before_enrollment = capsule / "predictor" / "before" / "enrollment_only"
    before_apply = capsule / "predictor" / "before" / "apply_only_staging"
    after_manifest = json.loads((after_enrollment / "package_manifest.json").read_text(encoding="utf-8"))
    before_manifest = json.loads((before_enrollment / "package_manifest.json").read_text(encoding="utf-8"))
    class_count = int(after_manifest["registered_class_count"])
    old_count = int(before_manifest["registered_class_count"])
    if class_count != 11 or old_count != 6:
        raise RuntimeError(f"expected K10/new5 registry 6->11, got {old_count}->{class_count}")
    runtime_path = after_enrollment / "sealed_feature_runtime.pt"
    runtime = torch.jit.load(str(runtime_path)).cuda().eval()
    with np.load(after_enrollment / f"support_{SCENARIOS[0]}.npz", allow_pickle=False) as warm:
        warm_rows = torch.from_numpy(warm["support_leo_weak_iq"][:2]).cuda()
    with torch.inference_mode():
        runtime(warm_rows)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    scene_states: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    feature_timings: dict[str, Any] = {}
    quantization_audit: dict[str, Any] = {}
    for scenario in SCENARIOS:
        with np.load(after_enrollment / f"support_{scenario}.npz", allow_pickle=False) as support_file:
            support_iq = support_file["support_leo_weak_iq"]
            labels = support_file["support_class_indices"].astype(np.int64)
        with np.load(after_apply / f"query_{scenario}.npz", allow_pickle=False) as query_file:
            query_iq = query_file["query_leo_weak_iq"]
            query_tokens = query_file["query_tokens"].astype(str)
        with np.load(before_apply / f"query_{scenario}.npz", allow_pickle=False) as before_query_file:
            before_tokens = before_query_file["query_tokens"].astype(str)
        support_z, support_seconds = _extract(runtime, support_iq)
        query_z, query_seconds = _extract(runtime, query_iq)
        auxiliary_start = time.perf_counter()
        support_fft = spectral_logmag_sketch(support_iq, dim=96)
        query_fft = spectral_logmag_sketch(query_iq, dim=96)
        support_rf = rf_statistics(support_iq)
        query_rf = rf_statistics(query_iq)
        support_dp16 = _differential_phase_descriptor(support_iq, 16)
        query_dp16 = _differential_phase_descriptor(query_iq, 16)
        support_dp32 = _differential_phase_descriptor(support_iq, 32)
        query_dp32 = _differential_phase_descriptor(query_iq, 32)
        support_g_fp32 = _normalise(
            np.concatenate([_normalise(support_z), 8.0 * _normalise(support_fft)], axis=1)
        )
        query_g = _normalise(
            np.concatenate([_normalise(query_z), 8.0 * _normalise(query_fft)], axis=1)
        )
        auxiliary_seconds = time.perf_counter() - auxiliary_start
        support_g, codes, scales = _int8_support_roundtrip(support_g_fp32)
        quantization_audit[scenario] = {
            "support_code_rows": int(codes.shape[0]),
            "support_code_dim": int(codes.shape[1]),
            "support_int8_bytes": int(codes.nbytes),
            "support_fp16_scale_bytes": int(scales.nbytes),
            "max_cosine_roundtrip_error": float(np.max(np.abs(1.0 - np.sum(support_g_fp32 * support_g, axis=1)))),
        }
        old_mask = labels < old_count
        old_support = support_g[old_mask]
        old_labels = labels[old_mask]
        if not np.array_equal(np.unique(labels), np.arange(class_count)):
            raise RuntimeError("support registry is not dense 0..C-1")
        if any(int(np.sum(labels == c)) != 10 for c in range(class_count)):
            raise RuntimeError("support is not exact K10 per registered class")
        token_to_row = {token: row for row, token in enumerate(query_tokens.tolist())}
        before_indices = np.asarray([token_to_row[token] for token in before_tokens], dtype=np.int64)
        state: dict[str, Any] = {
            "scenario": scenario,
            "support": support_g,
            "old_support": old_support,
            "labels": labels,
            "old_labels": old_labels,
            "query": query_g,
            "query_tokens": query_tokens,
            "before_tokens": before_tokens,
            "before_indices": before_indices,
            "support_z": support_z,
            "query_z": query_z,
            "support_fft": support_fft,
            "query_fft": query_fft,
            "support_rf": support_rf,
            "query_rf": query_rf,
            "support_dp16": support_dp16,
            "query_dp16": query_dp16,
            "support_dp32": support_dp32,
            "query_dp32": query_dp32,
            "base_after_loo": _loo_scores_np(support_g, labels, class_count),
            "base_before_loo": _loo_scores_np(old_support, old_labels, old_count),
            "fits": {},
        }
        for config_name, config in CONFIGS.items():
            before_weight, before_trace = _fit_metric(
                old_support,
                old_labels,
                old_count,
                old_count,
                config,
                scenario=scenario,
                phase="before_registration",
                teacher_old_log_weight=None,
            )
            after_weight, after_trace = _fit_metric(
                support_g,
                labels,
                class_count,
                old_count,
                config,
                scenario=scenario,
                phase="after_registration",
                teacher_old_log_weight=before_weight,
            )
            state["fits"][config_name] = {
                "before_log_weight": before_weight,
                "after_log_weight": after_weight,
                "before_loo": _loo_scores_np(_weighted_rows(old_support, before_weight), old_labels, old_count),
                "after_loo": _loo_scores_np(_weighted_rows(support_g, after_weight), labels, class_count),
            }
            all_traces.extend(
                [{"candidate": config_name, **row} for row in before_trace + after_trace]
            )
        feature_timings[scenario] = {
            "support_backbone_ms_per_sample": support_seconds * 1000.0 / support_iq.shape[0],
            "query_backbone_ms_per_sample": query_seconds * 1000.0 / query_iq.shape[0],
            "all_fixed_auxiliary_branches_ms_per_physical_sample": auxiliary_seconds * 1000.0
            / (support_iq.shape[0] + query_iq.shape[0]),
        }
        scene_states.append(state)
    selected, selection_grid = _select_blend(scene_states, old_count)
    selected_config = str(selected["config"])
    selected_alpha = float(selected["alpha"])
    print(f"[SUPPORT-LOCK] config={selected_config} alpha={selected_alpha}", flush=True)
    family_locks, selected_representation, representation_grid = _select_representations(
        scene_states, old_count
    )
    print(
        "[REPRESENTATION-LOCK] "
        + json.dumps(
            {
                "family": selected_representation["family"],
                "config": selected_representation["config"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    selected_metric_states: list[dict[str, Any]] = []
    for state in scene_states:
        selected_support_fp32 = _branch_representation(
            str(selected_representation["family"]),
            state["support_z"],
            state["support_fft"],
            state["support_rf"],
            state["support_dp16"],
            state["support_dp32"],
            dict(selected_representation["config"]),
        )
        selected_query = _branch_representation(
            str(selected_representation["family"]),
            state["query_z"],
            state["query_fft"],
            state["query_rf"],
            state["query_dp16"],
            state["query_dp32"],
            dict(selected_representation["config"]),
        )
        selected_support, _, _ = _int8_support_roundtrip(selected_support_fp32)
        selected_old_support = selected_support[state["labels"] < old_count]
        selected_fits: dict[str, Any] = {}
        for config_name, config in CONFIGS.items():
            before_weight, before_trace = _fit_metric(
                selected_old_support,
                state["old_labels"],
                old_count,
                old_count,
                config,
                scenario=state["scenario"],
                phase="selected_descriptor_before_registration",
                teacher_old_log_weight=None,
            )
            after_weight, after_trace = _fit_metric(
                selected_support,
                state["labels"],
                class_count,
                old_count,
                config,
                scenario=state["scenario"],
                phase="selected_descriptor_after_registration",
                teacher_old_log_weight=before_weight,
            )
            selected_fits[config_name] = {
                "before_log_weight": before_weight,
                "after_log_weight": after_weight,
                "before_loo": _loo_scores_np(
                    _weighted_rows(selected_old_support, before_weight), state["old_labels"], old_count
                ),
                "after_loo": _loo_scores_np(
                    _weighted_rows(selected_support, after_weight), state["labels"], class_count
                ),
            }
            all_traces.extend(
                [
                    {"candidate": "selected_descriptor__" + config_name, **row}
                    for row in before_trace + after_trace
                ]
            )
        state["selected_support"] = selected_support
        state["selected_old_support"] = selected_old_support
        state["selected_query"] = selected_query
        state["selected_fits"] = selected_fits
        selected_metric_states.append(
            {
                "scenario": state["scenario"],
                "labels": state["labels"],
                "old_labels": state["old_labels"],
                "base_after_loo": _loo_scores_np(selected_support, state["labels"], class_count),
                "base_before_loo": _loo_scores_np(selected_old_support, state["old_labels"], old_count),
                "fits": selected_fits,
            }
        )
    selected_descriptor_metric, selected_descriptor_metric_grid = _select_blend(
        selected_metric_states, old_count
    )
    print(
        "[SELECTED-DESCRIPTOR-METRIC-LOCK] "
        + json.dumps(
            {
                "config": selected_descriptor_metric["config"],
                "alpha": selected_descriptor_metric["alpha"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    arrays: dict[str, np.ndarray] = {}
    resources: dict[str, Any] = {}
    for method in METHODS:
        after_predictions = []
        after_tokens = []
        after_scenarios = []
        before_predictions = []
        before_tokens = []
        before_scenarios = []
        latencies = []
        for state in scene_states:
            base_after = _top1_scores(state["query"], state["support"], state["labels"], class_count)
            base_before = _top1_scores(
                state["query"][state["before_indices"]], state["old_support"], state["old_labels"], old_count
            )
            branch_family = None
            branch_config = None
            branch_dim = None
            if method in REPRESENTATION_FAMILIES:
                branch_lock = family_locks[method]
                branch_family = str(branch_lock["family"])
                branch_config = dict(branch_lock["config"])
                branch_support_fp32 = _branch_representation(
                    branch_family,
                    state["support_z"],
                    state["support_fft"],
                    state["support_rf"],
                    state["support_dp16"],
                    state["support_dp32"],
                    branch_config,
                )
                branch_query = _branch_representation(
                    branch_family,
                    state["query_z"],
                    state["query_fft"],
                    state["query_rf"],
                    state["query_dp16"],
                    state["query_dp32"],
                    branch_config,
                )
                branch_support, _, _ = _int8_support_roundtrip(branch_support_fp32)
                branch_dim = int(branch_support.shape[1])
                branch_old_support = branch_support[state["labels"] < old_count]
                after_scores = _top1_scores(
                    branch_query, branch_support, state["labels"], class_count
                )
                before_scores = _top1_scores(
                    branch_query[state["before_indices"]],
                    branch_old_support,
                    state["old_labels"],
                    old_count,
                )
                log_weight = None
                alpha = 0.0
                latency_query = branch_query
                latency_support = branch_support
            elif method == "S_selected_descriptor_floor_safe":
                config_name = str(selected_descriptor_metric["config"])
                alpha = float(selected_descriptor_metric["alpha"])
                fit = state["selected_fits"][config_name]
                selected_base_after = _top1_scores(
                    state["selected_query"], state["selected_support"], state["labels"], class_count
                )
                selected_base_before = _top1_scores(
                    state["selected_query"][state["before_indices"]],
                    state["selected_old_support"],
                    state["old_labels"],
                    old_count,
                )
                selected_diag_after = _top1_scores(
                    _weighted_rows(state["selected_query"], fit["after_log_weight"]),
                    _weighted_rows(state["selected_support"], fit["after_log_weight"]),
                    state["labels"],
                    class_count,
                )
                selected_diag_before = _top1_scores(
                    _weighted_rows(
                        state["selected_query"][state["before_indices"]], fit["before_log_weight"]
                    ),
                    _weighted_rows(state["selected_old_support"], fit["before_log_weight"]),
                    state["old_labels"],
                    old_count,
                )
                after_scores = (1.0 - alpha) * selected_base_after + alpha * selected_diag_after
                before_scores = (1.0 - alpha) * selected_base_before + alpha * selected_diag_before
                log_weight = fit["after_log_weight"]
                latency_query = state["selected_query"]
                latency_support = state["selected_support"]
            else:
                config_name = selected_config if method == "D256_L5_L6_blend" else method
                alpha = selected_alpha if method == "D256_L5_L6_blend" else 1.0
                fit = state["fits"][config_name]
                diag_after = _top1_scores(
                    _weighted_rows(state["query"], fit["after_log_weight"]),
                    _weighted_rows(state["support"], fit["after_log_weight"]),
                    state["labels"],
                    class_count,
                )
                diag_before = _top1_scores(
                    _weighted_rows(state["query"][state["before_indices"]], fit["before_log_weight"]),
                    _weighted_rows(state["old_support"], fit["before_log_weight"]),
                    state["old_labels"],
                    old_count,
                )
                after_scores = (1.0 - alpha) * base_after + alpha * diag_after
                before_scores = (1.0 - alpha) * base_before + alpha * diag_before
                log_weight = fit["after_log_weight"]
                latency_query = state["query"]
                latency_support = state["support"]
            after_pred = after_scores.argmax(axis=1).astype(np.int64)
            before_pred = before_scores.argmax(axis=1).astype(np.int64)
            after_predictions.append(after_pred)
            after_tokens.append(state["query_tokens"])
            after_scenarios.append(np.full(after_pred.shape[0], state["scenario"]))
            before_predictions.append(before_pred)
            before_tokens.append(state["before_tokens"])
            before_scenarios.append(np.full(before_pred.shape[0], state["scenario"]))
            latencies.append(
                _classifier_latency(
                    latency_query,
                    latency_support,
                    state["labels"],
                    class_count,
                    log_weight,
                    alpha,
                )
            )
        arrays[f"{method}__after_predictions"] = np.concatenate(after_predictions)
        arrays[f"{method}__after_tokens"] = np.concatenate(after_tokens)
        arrays[f"{method}__after_scenarios"] = np.concatenate(after_scenarios)
        arrays[f"{method}__before_predictions"] = np.concatenate(before_predictions)
        arrays[f"{method}__before_tokens"] = np.concatenate(before_tokens)
        arrays[f"{method}__before_scenarios"] = np.concatenate(before_scenarios)
        representation_method = method in REPRESENTATION_FAMILIES
        selected_descriptor_method = method == "S_selected_descriptor_floor_safe"
        trained = not representation_method
        blend = method == "D256_L5_L6_blend"
        if representation_method:
            lock = family_locks[method]
            state_dim = _representation_dim(lock)
        elif selected_descriptor_method:
            state_dim = _representation_dim(selected_representation)
        else:
            state_dim = DIM
        support_state = class_count * 10 * state_dim + class_count * 10 * 2
        resources[method] = {
            "trainable_parameters": state_dim if trained else 0,
            "adaptation_epochs": EPOCHS if trained else 0,
            "dense_query_graph": False,
            "persistent_state_bytes_after": support_state
            + (state_dim * 4 if trained else 0)
            + (4 if blend or selected_descriptor_method else 0),
            "support_codes": "rowwise_int8_plus_fp16_scale",
            "optimizer_state_persisted": False,
            "adaptation_training_MAC_estimate_after": (
                EPOCHS * (class_count * 10) ** 2 * state_dim if trained else 0
            ),
            "enrollment_metric_transform_MAC_after": class_count * 10 * 4 * state_dim if trained else 0,
            "representation_family": (
                family_locks[method]["family"]
                if representation_method
                else selected_representation["family"]
                if selected_descriptor_method
                else "z_id160_plus_8x_fft96"
            ),
            "representation_config": (
                family_locks[method]["config"]
                if representation_method
                else selected_representation["config"]
                if selected_descriptor_method
                else None
            ),
            "query_classifier_MAC_after": (
                class_count * 10 * state_dim
                if not trained
                else (
                    2
                    if (blend and selected_alpha < 1.0)
                    or (
                        selected_descriptor_method
                        and float(selected_descriptor_metric["alpha"]) < 1.0
                    )
                    else 1
                )
                * class_count
                * 10
                * state_dim
                + 4 * state_dim
            ),
            "classifier_ms_per_sample_mean": float(np.mean([row[0] for row in latencies])),
            "classifier_ms_per_sample_p95": float(np.max([row[1] for row in latencies])),
        }
    arrays["schema_json"] = np.asarray(
        json.dumps(
            {
                "schema": "cvs.phase2.d21_floor_aware_diag_predictions.v1",
                "fixed_representation": "normalize(concat(normalize(z_id160),8*normalize(FFT96)))",
                "support_only_fit": True,
                "query_truth_or_role_in_predictor_input": False,
                "query_fit": False,
                "query_quota_or_global_assignment": False,
                "all_registered_classes_per_sample": True,
                "methods": METHODS,
            },
            sort_keys=True,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    trace_path = output.parent / "loss_trace.jsonl"
    if trace_path.exists():
        raise FileExistsError(trace_path)
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_traces:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "schema": "cvs.phase2.d21_floor_aware_diag_predict_receipt.v1",
        "prediction_sha256": _sha256(output),
        "loss_trace_sha256": _sha256(trace_path),
        "loss_trace_record_count": len(all_traces),
        "capsule_offline_receipt_sha256": _sha256(capsule / "offline_build_receipt.json"),
        "sealed_runtime_sha256": _sha256(runtime_path),
        "support_only_selection": {
            "selected": selected,
            "grid": selection_grid,
            "query_used_for_selection": False,
        },
        "support_only_representation_selection": {
            "family_locks": family_locks,
            "selected": selected_representation,
            "grid": representation_grid,
            "query_used_for_selection": False,
        },
        "support_only_selected_descriptor_metric_selection": {
            "selected": selected_descriptor_metric,
            "grid": selected_descriptor_metric_grid,
            "query_used_for_selection": False,
        },
        "resources": resources,
        "feature_runtime_timings": feature_timings,
        "quantization_audit": quantization_audit,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "query_truth_opened": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
    }
    _json_dump(output.with_suffix(".receipt.json"), receipt)


def _class_metrics(pred: np.ndarray, truth_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    truth = np.asarray([int(row["true_class_index"]) for row in truth_rows], dtype=np.int64)
    result: dict[str, dict[str, Any]] = {}
    for class_index in sorted(set(truth.tolist())):
        mask = truth == class_index
        first = truth_rows[int(np.flatnonzero(mask)[0])]
        result[first["true_class_handle"]] = {
            "class_index": class_index,
            "transmitter_label": first["transmitter_label"],
            "count": int(mask.sum()),
            "accuracy": float(np.mean(pred[mask] == truth[mask])),
        }
    return result


def score(prediction: Path, truth_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    truth_doc = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_token = {row["query_token"]: row for row in truth_doc["rows"]}
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d21_floor_aware_diag_score.v1",
        "prediction_sha256": _sha256(prediction),
        "truth_sidecar_sha256": _sha256(truth_path),
        "query_truth_joined_only_after_immutable_predictions": True,
        "scorer_feedback_to_predictor": False,
        "methods": {},
    }
    with np.load(prediction, allow_pickle=False) as data:
        for method in METHODS:
            scenario_rows = []
            pooled_before_pred = []
            pooled_before_truth = []
            pooled_after_pred = []
            pooled_after_truth = []
            pooled_after_rows: list[dict[str, Any]] = []
            for scenario in SCENARIOS:
                before_mask = data[f"{method}__before_scenarios"] == scenario
                after_mask = data[f"{method}__after_scenarios"] == scenario
                before_tokens = data[f"{method}__before_tokens"][before_mask].astype(str)
                after_tokens = data[f"{method}__after_tokens"][after_mask].astype(str)
                before_rows = [truth_by_token[token] for token in before_tokens]
                after_rows = [truth_by_token[token] for token in after_tokens]
                before_truth = np.asarray([row["true_class_index"] for row in before_rows], dtype=np.int64)
                after_truth = np.asarray([row["true_class_index"] for row in after_rows], dtype=np.int64)
                before_pred = data[f"{method}__before_predictions"][before_mask]
                after_pred = data[f"{method}__after_predictions"][after_mask]
                old_count = len(set(before_truth.tolist()))
                after_metrics = _metrics(after_pred, after_truth, old_count)
                old_mask = after_truth < old_count
                old_class = _class_metrics(after_pred[old_mask], [row for row in after_rows if int(row["true_class_index"]) < old_count])
                new_class = _class_metrics(after_pred[~old_mask], [row for row in after_rows if int(row["true_class_index"]) >= old_count])
                before_acc = float(np.mean(before_pred == before_truth))
                scenario_rows.append(
                    {
                        "scenario": scenario,
                        "old_acc_before_increment": before_acc,
                        "old_acc": after_metrics["old_acc"],
                        "min_old_class_acc": min(row["accuracy"] for row in old_class.values()),
                        "seen_new_acc": after_metrics["seen_new_acc"],
                        "min_seen_new_class_acc": min(row["accuracy"] for row in new_class.values()),
                        "H_old_new": after_metrics["H_old_new"],
                        "average_forgetting": before_acc - after_metrics["old_acc"],
                        "old_per_class": old_class,
                        "seen_new_per_class": new_class,
                    }
                )
                pooled_before_pred.append(before_pred)
                pooled_before_truth.append(before_truth)
                pooled_after_pred.append(after_pred)
                pooled_after_truth.append(after_truth)
                pooled_after_rows.extend(after_rows)
            before_pred = np.concatenate(pooled_before_pred)
            before_truth = np.concatenate(pooled_before_truth)
            after_pred = np.concatenate(pooled_after_pred)
            after_truth = np.concatenate(pooled_after_truth)
            old_count = len(set(before_truth.tolist()))
            aggregate = _metrics(after_pred, after_truth, old_count)
            old_mask = after_truth < old_count
            old_class = _class_metrics(after_pred[old_mask], [row for row in pooled_after_rows if int(row["true_class_index"]) < old_count])
            new_class = _class_metrics(after_pred[~old_mask], [row for row in pooled_after_rows if int(row["true_class_index"]) >= old_count])
            before_acc = float(np.mean(before_pred == before_truth))
            result["methods"][method] = {
                "scenario_rows": scenario_rows,
                "aggregate": {
                    "old_acc_before_increment": before_acc,
                    "old_acc": aggregate["old_acc"],
                    "min_old_class_acc": min(row["accuracy"] for row in old_class.values()),
                    "seen_new_acc": aggregate["seen_new_acc"],
                    "min_seen_new_class_acc": min(row["accuracy"] for row in new_class.values()),
                    "H_old_new": aggregate["H_old_new"],
                    "average_forgetting": before_acc - aggregate["old_acc"],
                    "old_per_class": old_class,
                    "seen_new_per_class": new_class,
                },
            }
    _json_dump(output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--capsule", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--prediction", type=Path, required=True)
    scorer.add_argument("--truth", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.capsule.resolve(), args.output.resolve())
    else:
        score(args.prediction.resolve(), args.truth.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
