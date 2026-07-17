"""M1 dual-state old/new competition protection on a sealed Phase2 capsule.

Predict fits theta_B and theta_C from registered support only. Query truth is
opened exclusively by the separate score command after predictions are sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
BASE_PATH = REPO_ROOT / "local_artifacts" / "d21_floor_explore" / "run_floor_aware_diag.py"
SPEC = importlib.util.spec_from_file_location("d21_floor_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load D21 fixed-representation helper")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SCENARIOS = BASE.SCENARIOS
METHODS = ("L6q_single_metric", "M1_dual_metric", "M1_dual_metric_radius")
EPOCHS = 10
BASE.EPOCHS = EPOCHS
DIM = 256
OLD_INVASION_UPPER_BOUND = 0.25
NEW_FIDELITY_LOWER_BOUND = 0.45


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
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "old_floor": min(old),
        "new_floor": min(new) if new else 0.0,
        "joint_floor": min(old + new),
        "H_old_new": harmonic,
    }


def _fit_theta_c(
    support: np.ndarray,
    labels: np.ndarray,
    old_count: int,
    theta_b: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = torch.from_numpy(support).float().cuda()
    y = torch.from_numpy(labels.astype(np.int64)).long().cuda()
    old_mask = y < old_count
    new_mask = ~old_mask
    new_labels = y[new_mask] - old_count
    theta_b_t = torch.from_numpy(theta_b).float().cuda()
    theta_c = torch.nn.Parameter(theta_b_t.clone())
    optimiser = torch.optim.Adam([theta_c], lr=0.035)
    with torch.no_grad():
        b_all = F.normalize(x * torch.exp(theta_b_t).unsqueeze(0), dim=1)
        b_old = b_all[old_mask]
        old_labels = y[old_mask]
        old_b_loo = BASE._torch_loo_scores(b_old, old_labels, old_count)
        old_b_true = old_b_loo.gather(1, old_labels[:, None]).squeeze(1)
        new_to_old_b = torch.stack(
            [(b_all[new_mask] @ b_old[old_labels == c].T).max(dim=1).values for c in range(old_count)],
            dim=1,
        )
        old_b_pairwise = b_old @ b_old.T
    trace = []
    for epoch in range(1, EPOCHS + 1):
        optimiser.zero_grad(set_to_none=True)
        c_all = F.normalize(x * torch.exp(theta_c).unsqueeze(0), dim=1)
        c_old = c_all[old_mask]
        c_new = c_all[new_mask]
        new_count = int(new_labels.max().item()) + 1
        new_loo = BASE._torch_loo_scores(c_new, new_labels, new_count)
        ce_rows = F.cross_entropy(20.0 * new_loo, new_labels, reduction="none")
        new_ce = ce_rows.mean()
        class_losses = torch.stack([ce_rows[new_labels == c].mean() for c in range(new_count)])
        class_cvar = torch.topk(class_losses, k=max(1, int(math.ceil(0.4 * new_count)))).values.mean()
        old_to_new = torch.stack(
            [(c_old @ c_new[new_labels == c].T).max(dim=1).values for c in range(new_count)],
            dim=1,
        )
        intrusion = F.relu(old_to_new.max(dim=1).values - old_b_true + 0.02).mean()
        new_true = new_loo.gather(1, new_labels[:, None]).squeeze(1)
        fidelity = F.relu(new_to_old_b.max(dim=1).values - new_true + 0.02).mean()
        pairwise = F.smooth_l1_loss(c_old @ c_old.T, old_b_pairwise)
        delta_l2 = (theta_c - theta_b_t).square().mean()
        total = new_ce + 0.5 * class_cvar + intrusion + fidelity + 0.25 * pairwise + 0.02 * delta_l2
        total.backward()
        optimiser.step()
        with torch.no_grad():
            theta_c.clamp_(-0.8, 0.8)
            new_pred = new_loo.argmax(dim=1)
            new_recall = [
                float((new_pred[new_labels == c] == c).float().mean().item()) for c in range(new_count)
            ]
            invasion_rate = float((old_to_new.max(dim=1).values >= old_b_true).float().mean().item())
            fidelity_rate = float((new_true > new_to_old_b.max(dim=1).values).float().mean().item())
        row = {
            "candidate": "M1_theta_C",
            "scenario": scenario,
            "epoch": epoch,
            "total_loss": float(total.item()),
            "new_ce": float(new_ce.item()),
            "new_class_cvar": float(class_cvar.item()),
            "old_intrusion_hinge": float(intrusion.item()),
            "new_fidelity_hinge": float(fidelity.item()),
            "old_pairwise_preservation": float(pairwise.item()),
            "theta_delta_l2": float(delta_l2.item()),
            "support_new_acc": float((new_pred == new_labels).float().mean().item()),
            "support_new_floor": min(new_recall),
            "support_old_invasion_rate": invasion_rate,
            "support_new_fidelity_rate": fidelity_rate,
        }
        trace.append(row)
        print("[M1-LOSS] " + json.dumps(row, sort_keys=True), flush=True)
    return theta_c.detach().cpu().numpy().astype(np.float32), trace


def _class_radii(rows: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    loo = BASE._loo_scores_np(rows, labels, class_count)
    radius = []
    for class_index in range(class_count):
        mask = labels == class_index
        distances = 1.0 - loo[mask, class_index]
        radius.append(float(np.quantile(distances, 0.9, method="higher")))
    return np.maximum(np.asarray(radius, dtype=np.float32), 1e-3)


def _radius_standardize(scores: np.ndarray, radii: np.ndarray) -> np.ndarray:
    boundary = 1.0 - radii
    return (scores - boundary[None, :]) / radii[None, :]


def _dual_support_scores(state: dict[str, Any], *, radius: bool) -> tuple[np.ndarray, np.ndarray]:
    labels = state["labels"]
    old_count = state["old_count"]
    old_mask = labels < old_count
    new_mask = ~old_mask
    b = BASE._weighted_rows(state["support"], state["theta_b"])
    c = BASE._weighted_rows(state["support"], state["theta_c"])
    b_old = b[old_mask]
    c_new = c[new_mask]
    old_labels = labels[old_mask]
    new_labels_local = labels[new_mask] - old_count
    old_loo = BASE._loo_scores_np(b_old, old_labels, old_count)
    new_loo = BASE._loo_scores_np(c_new, new_labels_local, state["class_count"] - old_count)
    old_to_new = BASE._top1_scores(c[old_mask], c_new, new_labels_local, state["class_count"] - old_count)
    new_to_old = BASE._top1_scores(b[new_mask], b_old, old_labels, old_count)
    old_scores = np.concatenate([old_loo, old_to_new], axis=1)
    new_scores = np.concatenate([new_to_old, new_loo], axis=1)
    scores = np.empty((labels.shape[0], state["class_count"]), dtype=np.float32)
    scores[old_mask] = old_scores
    scores[new_mask] = new_scores
    before = old_loo
    if radius:
        old_radii = _class_radii(b_old, old_labels, old_count)
        new_radii = _class_radii(c_new, new_labels_local, state["class_count"] - old_count)
        scores[:, :old_count] = _radius_standardize(scores[:, :old_count], old_radii)
        scores[:, old_count:] = _radius_standardize(scores[:, old_count:], new_radii)
        before = _radius_standardize(before, old_radii)
    return scores, before


def _apply_calibration(scores: np.ndarray, old_count: int, config: dict[str, float]) -> np.ndarray:
    result = scores.copy()
    result[:, :old_count] /= config["old_temperature"]
    result[:, old_count:] /= config["new_temperature"]
    result[:, old_count:] -= config["new_offset"]
    return result


def _lock_calibration(
    states: list[dict[str, Any]], *, radius: bool
) -> tuple[dict[str, float], dict[str, Any]]:
    evaluations = []
    for old_temperature in (0.9, 1.0, 1.1):
        for new_temperature in (0.9, 1.0, 1.1):
            for new_offset in (0.0, 0.02, 0.04):
                config = {
                    "old_temperature": old_temperature,
                    "new_temperature": new_temperature,
                    "new_offset": new_offset,
                }
                scene_rows = []
                for state in states:
                    support_scores, before_scores = _dual_support_scores(state, radius=radius)
                    calibrated = _apply_calibration(support_scores, state["old_count"], config)
                    pred = calibrated.argmax(axis=1)
                    before_pred = before_scores.argmax(axis=1)
                    metrics = _metrics(pred, state["labels"], state["old_count"])
                    old_mask = state["labels"] < state["old_count"]
                    new_mask = ~old_mask
                    before_acc = float(np.mean(before_pred == state["labels"][old_mask]))
                    invasion = float(np.mean(pred[old_mask] >= state["old_count"]))
                    fidelity = float(np.mean(pred[new_mask] == state["labels"][new_mask]))
                    scene_rows.append(
                        {
                            "scenario": state["scenario"],
                            **metrics,
                            "old_acc_before_increment": before_acc,
                            "average_forgetting": before_acc - metrics["old_acc"],
                            "old_support_invasion_rate": invasion,
                            "new_support_fidelity": fidelity,
                        }
                    )
                gate = all(
                    row["old_support_invasion_rate"] <= OLD_INVASION_UPPER_BOUND
                    and row["new_support_fidelity"] >= NEW_FIDELITY_LOWER_BOUND
                    for row in scene_rows
                )
                evaluations.append(
                    {
                        "config": config,
                        "radius_standardized": radius,
                        "gate_pass": gate,
                        "scenario_rows": scene_rows,
                        "worst_old_floor": min(row["old_floor"] for row in scene_rows),
                        "worst_new_floor": min(row["new_floor"] for row in scene_rows),
                        "worst_h": min(row["H_old_new"] for row in scene_rows),
                        "worst_forgetting": max(row["average_forgetting"] for row in scene_rows),
                    }
                )
    passing = [row for row in evaluations if row["gate_pass"]]
    selected = max(
        passing if passing else evaluations,
        key=lambda row: (
            row["worst_old_floor"],
            row["worst_new_floor"],
            row["worst_h"],
            -max(row["worst_forgetting"], 0.0),
            -abs(row["config"]["new_offset"]),
        ),
    )
    selected["selection_status"] = (
        "SUPPORT_GATES_PASS" if passing else "SUPPORT_GATE_BLOCKED_DIAGNOSTIC"
    )
    return dict(selected["config"]), {"selected": selected, "grid": evaluations}


def _latency(
    query: np.ndarray,
    old_support: np.ndarray,
    new_support: np.ndarray,
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    theta_b: np.ndarray,
    theta_c: np.ndarray,
    repeats: int = 200,
) -> tuple[float, float]:
    old_codes = BASE._weighted_rows(old_support, theta_b)
    new_codes = BASE._weighted_rows(new_support, theta_c)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        BASE._top1_scores(BASE._weighted_rows(query, theta_b), old_codes, old_labels, len(np.unique(old_labels)))
        BASE._top1_scores(BASE._weighted_rows(query, theta_c), new_codes, new_labels, len(np.unique(new_labels)))
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
    if (old_count, class_count) != (6, 11):
        raise RuntimeError(f"expected 6->11 registry, got {old_count}->{class_count}")
    runtime_path = after_enrollment / "sealed_feature_runtime.pt"
    runtime = torch.jit.load(str(runtime_path)).cuda().eval()
    states = []
    traces = []
    torch.cuda.reset_peak_memory_stats()
    for scenario in SCENARIOS:
        with np.load(after_enrollment / f"support_{scenario}.npz", allow_pickle=False) as support_file:
            support_iq = support_file["support_leo_weak_iq"]
            labels = support_file["support_class_indices"].astype(np.int64)
        with np.load(after_apply / f"query_{scenario}.npz", allow_pickle=False) as query_file:
            query_iq = query_file["query_leo_weak_iq"]
            query_tokens = query_file["query_tokens"].astype(str)
        with np.load(before_apply / f"query_{scenario}.npz", allow_pickle=False) as before_query_file:
            before_tokens = before_query_file["query_tokens"].astype(str)
        support_z, _ = BASE._extract(runtime, support_iq)
        query_z, _ = BASE._extract(runtime, query_iq)
        support_g_fp32 = BASE._fixed_representation(support_z, support_iq)
        query_g = BASE._fixed_representation(query_z, query_iq)
        support_g, _, _ = BASE._int8_support_roundtrip(support_g_fp32)
        old_mask = labels < old_count
        old_support = support_g[old_mask]
        old_labels = labels[old_mask]
        config = BASE.CONFIGS["L6_diag_floor"]
        theta_b, trace_b = BASE._fit_metric(
            old_support,
            old_labels,
            old_count,
            old_count,
            config,
            scenario=scenario,
            phase="M1_theta_B_stage2b",
            teacher_old_log_weight=None,
        )
        theta_single, trace_single = BASE._fit_metric(
            support_g,
            labels,
            class_count,
            old_count,
            config,
            scenario=scenario,
            phase="L6q_theta_C_all",
            teacher_old_log_weight=theta_b,
        )
        theta_c, trace_c = _fit_theta_c(support_g, labels, old_count, theta_b, scenario)
        traces.extend([{"candidate": "theta_B", **row} for row in trace_b])
        traces.extend([{"candidate": "L6q", **row} for row in trace_single])
        traces.extend(trace_c)
        token_to_row = {token: row for row, token in enumerate(query_tokens.tolist())}
        before_indices = np.asarray([token_to_row[token] for token in before_tokens], dtype=np.int64)
        states.append(
            {
                "scenario": scenario,
                "support": support_g,
                "labels": labels,
                "old_count": old_count,
                "class_count": class_count,
                "old_support": old_support,
                "new_support": support_g[~old_mask],
                "old_labels": old_labels,
                "new_labels": labels[~old_mask] - old_count,
                "query": query_g,
                "query_tokens": query_tokens,
                "before_tokens": before_tokens,
                "before_indices": before_indices,
                "theta_b": theta_b,
                "theta_single": theta_single,
                "theta_c": theta_c,
            }
        )
    raw_config, raw_lock = _lock_calibration(states, radius=False)
    radius_config, radius_lock = _lock_calibration(states, radius=True)
    print("[M1-LOCK-RAW] " + json.dumps(raw_config, sort_keys=True), flush=True)
    print("[M1-LOCK-RADIUS] " + json.dumps(radius_config, sort_keys=True), flush=True)
    arrays: dict[str, np.ndarray] = {}
    timings = []
    for method in METHODS:
        after_predictions = []
        after_tokens = []
        after_scenarios = []
        before_predictions = []
        before_tokens = []
        before_scenarios = []
        for state in states:
            if method == "L6q_single_metric":
                adapted_support = BASE._weighted_rows(state["support"], state["theta_single"])
                after_scores = BASE._top1_scores(
                    BASE._weighted_rows(state["query"], state["theta_single"]),
                    adapted_support,
                    state["labels"],
                    state["class_count"],
                )
                before_support = BASE._weighted_rows(state["old_support"], state["theta_b"])
                before_scores = BASE._top1_scores(
                    BASE._weighted_rows(state["query"][state["before_indices"]], state["theta_b"]),
                    before_support,
                    state["old_labels"],
                    state["old_count"],
                )
            else:
                b_query = BASE._weighted_rows(state["query"], state["theta_b"])
                c_query = BASE._weighted_rows(state["query"], state["theta_c"])
                b_old = BASE._weighted_rows(state["old_support"], state["theta_b"])
                c_new = BASE._weighted_rows(state["new_support"], state["theta_c"])
                old_scores = BASE._top1_scores(
                    b_query, b_old, state["old_labels"], state["old_count"]
                )
                new_scores = BASE._top1_scores(
                    c_query, c_new, state["new_labels"], state["class_count"] - state["old_count"]
                )
                before_scores = old_scores[state["before_indices"]]
                radius = method == "M1_dual_metric_radius"
                if radius:
                    old_radii = _class_radii(b_old, state["old_labels"], state["old_count"])
                    new_radii = _class_radii(
                        c_new, state["new_labels"], state["class_count"] - state["old_count"]
                    )
                    old_scores = _radius_standardize(old_scores, old_radii)
                    new_scores = _radius_standardize(new_scores, new_radii)
                    before_scores = _radius_standardize(before_scores, old_radii)
                after_scores = _apply_calibration(
                    np.concatenate([old_scores, new_scores], axis=1),
                    state["old_count"],
                    radius_config if radius else raw_config,
                )
            after_pred = after_scores.argmax(axis=1).astype(np.int64)
            before_pred = before_scores.argmax(axis=1).astype(np.int64)
            after_predictions.append(after_pred)
            after_tokens.append(state["query_tokens"])
            after_scenarios.append(np.full(after_pred.shape[0], state["scenario"]))
            before_predictions.append(before_pred)
            before_tokens.append(state["before_tokens"])
            before_scenarios.append(np.full(before_pred.shape[0], state["scenario"]))
        arrays[f"{method}__after_predictions"] = np.concatenate(after_predictions)
        arrays[f"{method}__after_tokens"] = np.concatenate(after_tokens)
        arrays[f"{method}__after_scenarios"] = np.concatenate(after_scenarios)
        arrays[f"{method}__before_predictions"] = np.concatenate(before_predictions)
        arrays[f"{method}__before_tokens"] = np.concatenate(before_tokens)
        arrays[f"{method}__before_scenarios"] = np.concatenate(before_scenarios)
    for state in states:
        timings.append(
            _latency(
                state["query"],
                state["old_support"],
                state["new_support"],
                state["old_labels"],
                state["new_labels"],
                state["theta_b"],
                state["theta_c"],
            )
        )
    arrays["schema_json"] = np.asarray(
        json.dumps(
            {
                "schema": "cvs.phase2.d21_m1_dual_state_predictions.v1",
                "fixed_representation": "normalize(concat(normalize(z_id160),8*normalize(FFT96)))",
                "methods": METHODS,
                "support_only_fit": True,
                "query_fit": False,
                "query_truth_or_role_input": False,
                "all_registered_classes_per_sample": True,
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
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    old_state = old_count * 10 * DIM + old_count * 10 * 2 + DIM * 4
    new_state = (class_count - old_count) * 10 * DIM + (class_count - old_count) * 10 * 2 + DIM * 4
    receipt = {
        "schema": "cvs.phase2.d21_m1_dual_state_receipt.v1",
        "prediction_sha256": _sha256(output),
        "loss_trace_sha256": _sha256(trace_path),
        "loss_trace_record_count": len(traces),
        "sealed_runtime_sha256": _sha256(runtime_path),
        "raw_calibration_lock": raw_lock,
        "radius_calibration_lock": radius_lock,
        "support_gates": {
            "old_support_invasion_upper_bound": OLD_INVASION_UPPER_BOUND,
            "new_support_fidelity_lower_bound": NEW_FIDELITY_LOWER_BOUND,
        },
        "resources": {
            "L6q_single_metric": {
                "trainable_parameters": DIM,
                "adaptation_epochs": EPOCHS,
                "persistent_state_bytes_after": class_count * 10 * DIM + class_count * 10 * 2 + DIM * 4,
                "query_classifier_MAC_after": class_count * 10 * DIM + 4 * DIM,
            },
            "M1_dual_metric": {
                "trainable_parameters": 2 * DIM,
                "adaptation_epochs_B_plus_C": 2 * EPOCHS,
                "persistent_state_bytes_after": old_state + new_state,
                "query_classifier_MAC_after": class_count * 10 * DIM + 8 * DIM,
            },
            "M1_dual_metric_radius": {
                "trainable_parameters": 2 * DIM,
                "adaptation_epochs_B_plus_C": 2 * EPOCHS,
                "persistent_state_bytes_after": old_state + new_state + class_count * 4,
                "query_classifier_MAC_after": class_count * 10 * DIM + 8 * DIM + class_count,
            },
        },
        "dual_classifier_ms_per_sample_mean": float(np.mean([row[0] for row in timings])),
        "dual_classifier_ms_per_sample_p95": float(np.max([row[1] for row in timings])),
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
    result = {}
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
        "schema": "cvs.phase2.d21_m1_dual_state_score.v1",
        "prediction_sha256": _sha256(prediction),
        "truth_sidecar_sha256": _sha256(truth_path),
        "query_truth_joined_only_after_immutable_predictions": True,
        "scorer_feedback_to_predictor": False,
        "methods": {},
    }
    with np.load(prediction, allow_pickle=False) as data:
        for method in METHODS:
            scene_rows = []
            pooled_before_pred = []
            pooled_before_truth = []
            pooled_after_pred = []
            pooled_after_truth = []
            pooled_rows = []
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
                metrics = _metrics(after_pred, after_truth, old_count)
                old_mask = after_truth < old_count
                old_class = _class_metrics(
                    after_pred[old_mask], [row for row in after_rows if int(row["true_class_index"]) < old_count]
                )
                new_class = _class_metrics(
                    after_pred[~old_mask], [row for row in after_rows if int(row["true_class_index"]) >= old_count]
                )
                before_acc = float(np.mean(before_pred == before_truth))
                scene_rows.append(
                    {
                        "scenario": scenario,
                        "old_acc_before_increment": before_acc,
                        "old_acc": metrics["old_acc"],
                        "min_old_class_acc": min(row["accuracy"] for row in old_class.values()),
                        "seen_new_acc": metrics["seen_new_acc"],
                        "min_seen_new_class_acc": min(row["accuracy"] for row in new_class.values()),
                        "H_old_new": metrics["H_old_new"],
                        "average_forgetting": before_acc - metrics["old_acc"],
                        "old_per_class": old_class,
                        "seen_new_per_class": new_class,
                    }
                )
                pooled_before_pred.append(before_pred)
                pooled_before_truth.append(before_truth)
                pooled_after_pred.append(after_pred)
                pooled_after_truth.append(after_truth)
                pooled_rows.extend(after_rows)
            bp = np.concatenate(pooled_before_pred)
            bt = np.concatenate(pooled_before_truth)
            ap = np.concatenate(pooled_after_pred)
            at = np.concatenate(pooled_after_truth)
            old_count = len(set(bt.tolist()))
            metrics = _metrics(ap, at, old_count)
            old_mask = at < old_count
            old_class = _class_metrics(ap[old_mask], [row for row in pooled_rows if int(row["true_class_index"]) < old_count])
            new_class = _class_metrics(ap[~old_mask], [row for row in pooled_rows if int(row["true_class_index"]) >= old_count])
            before_acc = float(np.mean(bp == bt))
            result["methods"][method] = {
                "scenario_rows": scene_rows,
                "aggregate": {
                    "old_acc_before_increment": before_acc,
                    "old_acc": metrics["old_acc"],
                    "min_old_class_acc": min(row["accuracy"] for row in old_class.values()),
                    "seen_new_acc": metrics["seen_new_acc"],
                    "min_seen_new_class_acc": min(row["accuracy"] for row in new_class.values()),
                    "H_old_new": metrics["H_old_new"],
                    "average_forgetting": before_acc - metrics["old_acc"],
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
