"""M6 support-only low-rank id projection adaptation.

This runner intentionally has no capsule-root, query, token, truth, prediction,
or score interface. It accepts only an ``after/enrollment_only`` package and
uses class-balanced folds of registered support to decide a support-only gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
WEIGHT_NAME = "model.id_backbone.cls_head.id_proj.0.weight"
BIAS_NAME = "model.id_backbone.cls_head.id_proj.0.bias"
ALLOWED_PARAMETER_NAMES = {WEIGHT_NAME, BIAS_NAME}
OLD_COUNT = 6
CLASS_COUNT = 11
EPOCHS = 5
MAX_STEPS = 50
LR = 0.05
FOLDS = 2
RANKS = (2, 4)
LOSS_PRESETS = {
    "balanced": {"cvar": 0.5, "old_pair": 1.0, "new_sep": 1.0, "prox": 0.01},
    "old_guard": {"cvar": 0.5, "old_pair": 4.0, "new_sep": 1.0, "prox": 0.02},
}
GATE_MIN_H_GAIN = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _guard_enrollment_root(root: Path) -> None:
    if root.name != "enrollment_only" or root.parent.name != "after":
        raise RuntimeError("input must be the exact predictor/after/enrollment_only directory")
    forbidden = ("query", "truth", "scorer", "apply_only", "before")
    lowered = str(root).lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError("forbidden non-enrollment path component")


def _load_and_verify_manifest(root: Path, access_log: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = root / "package_manifest.json"
    access_log.append({"purpose": "manifest", "path": str(manifest_path)})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "cvs.phase2.somph_predictor_bundle.v1"
        or manifest.get("profile") != "enrollment_only"
        or manifest.get("registration_state") != "after"
    ):
        raise RuntimeError("package is not after/enrollment_only")
    if int(manifest.get("registered_class_count", -1)) != CLASS_COUNT or int(manifest.get("k_shot", -1)) != 10:
        raise RuntimeError("M6 is locked to formal K10/new5 6->11 support")
    required_false = (
        "clean_sample_access",
        "clean_derived_signal_access",
        "phase2_clean_dataset_reachable",
        "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
        "phase2_source_sample_access",
        "phase2_source_derived_signal_access",
        "phase2_cross_scenario_physical_sample_reuse",
    )
    if any(manifest.get(field) is not False for field in required_false):
        raise RuntimeError("clean/source/single-view manifest guard failed")
    if manifest.get("phase2_sample_view_policy") != "leo_weak_only_no_clean_access":
        raise RuntimeError("LEO_weak-only guard failed")
    return manifest


def _member_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    members = {str(row["relative_path"]): row for row in manifest["members"]}
    expected = {
        "sealed_feature_runtime.pt": "feature_runtime",
        "method_lock.json": "method_lock",
        "overlay_provenance.json": "overlay_provenance",
        **{f"support_{scenario}.npz": f"support:{scenario}" for scenario in SCENARIOS},
    }
    if set(members) != set(expected):
        raise RuntimeError(f"manifest member allowlist mismatch: {sorted(set(members) ^ set(expected))}")
    forbidden = ("query", "truth", "scorer", "apply_only", "before")
    for relative_path, expected_kind in expected.items():
        path = Path(relative_path)
        row = members[relative_path]
        if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
            raise RuntimeError("manifest member path is not a single safe relative member")
        if any(token in relative_path.lower() for token in forbidden):
            raise RuntimeError("manifest contains a forbidden non-enrollment member")
        if row.get("kind") != expected_kind:
            raise RuntimeError(f"manifest member kind mismatch for {relative_path}")
    return members


def _verified_member(root: Path, members: dict[str, dict[str, Any]], name: str, purpose: str, access_log: list[dict[str, Any]]) -> Path:
    if name not in members:
        raise RuntimeError(f"member absent from sealed manifest: {name}")
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise RuntimeError("member escaped enrollment root")
    actual = _sha256(path)
    expected = str(members[name]["sha256"])
    if actual != expected:
        raise RuntimeError(f"pre-open hash mismatch for {name}")
    access_log.append({"purpose": purpose, "path": str(path), "sha256": actual})
    return path


def _load_runtime(path: Path) -> tuple[torch.jit.ScriptModule, torch.Tensor, torch.Tensor]:
    runtime = torch.jit.load(str(path), map_location="cuda").cuda().eval()
    parameters = dict(runtime.named_parameters())
    if not ALLOWED_PARAMETER_NAMES.issubset(parameters):
        raise RuntimeError("exact id_proj whitelist missing")
    if tuple(parameters[WEIGHT_NAME].shape) != (160, 160) or tuple(parameters[BIAS_NAME].shape) != (160,):
        raise RuntimeError("id_proj shape drift")
    for name, parameter in runtime.named_parameters():
        parameter.requires_grad_(name in ALLOWED_PARAMETER_NAMES)
    if {name for name, parameter in runtime.named_parameters() if parameter.requires_grad} != ALLOWED_PARAMETER_NAMES:
        raise RuntimeError("requires_grad escaped exact whitelist")
    return runtime, parameters[WEIGHT_NAME].detach().clone(), parameters[BIAS_NAME].detach().clone()


def _extract(runtime: torch.jit.ScriptModule, iq: np.ndarray) -> torch.Tensor:
    rows = torch.from_numpy(np.asarray(iq, dtype=np.float32)).cuda()
    z_id, _ = runtime(rows)
    return F.normalize(z_id.float(), dim=1)


def _scores(query: torch.Tensor, support: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    similarities = query @ support.T
    return torch.stack([similarities[:, labels == c].max(dim=1).values for c in range(CLASS_COUNT)], dim=1)


def _loo_scores(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    similarities = rows @ rows.T
    similarities = similarities.masked_fill(torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool), -1e4)
    return torch.stack([similarities[:, labels == c].max(dim=1).values for c in range(CLASS_COUNT)], dim=1)


def _metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    pred = scores.argmax(dim=1)
    per_class = [float((pred[labels == c] == c).float().mean().item()) for c in range(CLASS_COUNT)]
    old_acc = float((pred[labels < OLD_COUNT] == labels[labels < OLD_COUNT]).float().mean().item())
    new_acc = float((pred[labels >= OLD_COUNT] == labels[labels >= OLD_COUNT]).float().mean().item())
    return {
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "old_floor": min(per_class[:OLD_COUNT]),
        "new_floor": min(per_class[OLD_COUNT:]),
        "H_old_new": 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-12),
        "per_class_accuracy": per_class,
    }


def _fold_indices(labels: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train, validation = [], []
    for class_index in range(CLASS_COUNT):
        indices = np.flatnonzero(labels == class_index)
        if indices.size != 10:
            raise RuntimeError("formal support must have exactly K=10 per class")
        validation_indices = indices[fold * 5 : (fold + 1) * 5]
        train_indices = np.setdiff1d(indices, validation_indices, assume_unique=True)
        train.extend(train_indices.tolist())
        validation.extend(validation_indices.tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def _seed(scenario: str, rank: int, preset: str, fold: int) -> int:
    digest = hashlib.sha256(f"{scenario}|{rank}|{preset}|{fold}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _fit_fold(
    runtime_path: Path,
    train_iq: np.ndarray,
    train_labels_np: np.ndarray,
    validation_iq: np.ndarray,
    validation_labels_np: np.ndarray,
    scenario: str,
    rank: int,
    preset_name: str,
    fold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray], float]:
    runtime, base_weight, base_bias = _load_runtime(runtime_path)
    named = dict(runtime.named_parameters())
    model_weight = named[WEIGHT_NAME]
    model_bias = named[BIAS_NAME]
    train_labels = torch.from_numpy(train_labels_np).long().cuda()
    validation_labels = torch.from_numpy(validation_labels_np).long().cuda()
    old_mask = train_labels < OLD_COUNT
    generator = torch.Generator(device="cuda")
    generator.manual_seed(_seed(scenario, rank, preset_name, fold))
    q, _ = torch.linalg.qr(torch.randn(160, rank, generator=generator, device="cuda"), mode="reduced")
    left = torch.nn.Parameter(q.contiguous())
    right = torch.nn.Parameter(torch.zeros(rank, 160, device="cuda"))
    delta_bias = torch.nn.Parameter(torch.zeros(160, device="cuda"))
    optimiser = torch.optim.SGD([left, right, delta_bias], lr=LR, momentum=0.0)
    weights = LOSS_PRESETS[preset_name]
    with torch.no_grad():
        base_train = _extract(runtime, train_iq)
        base_old_pair = base_train[old_mask] @ base_train[old_mask].T
    trace: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        optimiser.zero_grad(set_to_none=True)
        with torch.no_grad():
            model_weight.copy_(base_weight + left @ right)
            model_bias.copy_(base_bias + delta_bias)
        model_weight.grad = None
        model_bias.grad = None
        train_z = _extract(runtime, train_iq)
        scores = _loo_scores(train_z, train_labels)
        ce_rows = F.cross_entropy(20.0 * scores, train_labels, reduction="none")
        class_losses = torch.stack([ce_rows[train_labels == c].mean() for c in range(CLASS_COUNT)])
        cvar = torch.topk(class_losses, k=int(math.ceil(0.4 * CLASS_COUNT))).values.mean()
        true_score = scores.gather(1, train_labels[:, None]).squeeze(1)
        new_sep = 0.5 * (
            F.relu(scores[train_labels >= OLD_COUNT, :OLD_COUNT].max(dim=1).values - true_score[train_labels >= OLD_COUNT] + 0.02).mean()
            + F.relu(scores[train_labels < OLD_COUNT, OLD_COUNT:].max(dim=1).values - true_score[train_labels < OLD_COUNT] + 0.02).mean()
        )
        old_pair = F.smooth_l1_loss(train_z[old_mask] @ train_z[old_mask].T, base_old_pair)
        delta = left @ right
        prox = delta.square().mean() + delta_bias.square().mean()
        total = ce_rows.mean() + weights["cvar"] * cvar + weights["old_pair"] * old_pair + weights["new_sep"] * new_sep + weights["prox"] * prox
        total.backward()
        if model_weight.grad is None or model_bias.grad is None:
            raise RuntimeError("scripted id_proj gradient missing")
        grad_weight = model_weight.grad.detach()
        grad_bias = model_bias.grad.detach()
        left.grad = grad_weight @ right.detach().T
        right.grad = left.detach().T @ grad_weight
        delta_bias.grad = grad_bias
        torch.nn.utils.clip_grad_norm_([left, right, delta_bias], 1.0)
        optimiser.step()
        train_metrics = _metrics(scores.detach(), train_labels)
        row = {
            "record_type": "train_epoch",
            "scenario": scenario,
            "rank": rank,
            "preset": preset_name,
            "fold": fold,
            "epoch": epoch,
            "step": epoch,
            "total_loss": float(total.detach().item()),
            "support_ce": float(ce_rows.mean().detach().item()),
            "class_cvar": float(cvar.detach().item()),
            "old_pair_retention": float(old_pair.detach().item()),
            "new_separation": float(new_sep.detach().item()),
            "identity_proximal": float(prox.detach().item()),
            **{f"train_{key}": value for key, value in train_metrics.items() if key != "per_class_accuracy"},
        }
        trace.append(row)
        print("[M6-SUPPORT] " + json.dumps(row, sort_keys=True), flush=True)
    if len(trace) > MAX_STEPS:
        raise RuntimeError("per-adaptation step cap exceeded")
    with torch.no_grad():
        model_weight.copy_(base_weight + left @ right)
        model_bias.copy_(base_bias + delta_bias)
        adapted_train = _extract(runtime, train_iq)
        adapted_validation = _extract(runtime, validation_iq)
        adapted_metrics = _metrics(_scores(adapted_validation, adapted_train, train_labels), validation_labels)
        model_weight.copy_(base_weight)
        model_bias.copy_(base_bias)
        base_train = _extract(runtime, train_iq)
        base_validation = _extract(runtime, validation_iq)
        base_metrics = _metrics(_scores(base_validation, base_train, train_labels), validation_labels)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    evaluation = {
        "record_type": "fold_validation",
        "scenario": scenario,
        "rank": rank,
        "preset": preset_name,
        "fold": fold,
        "base": base_metrics,
        "adapted": adapted_metrics,
        "delta_H": adapted_metrics["H_old_new"] - base_metrics["H_old_new"],
        "delta_old_floor": adapted_metrics["old_floor"] - base_metrics["old_floor"],
        "delta_new_floor": adapted_metrics["new_floor"] - base_metrics["new_floor"],
        "elapsed_seconds": elapsed,
    }
    factors = {
        "left": left.detach().cpu().numpy().astype(np.float16),
        "right": right.detach().cpu().numpy().astype(np.float16),
        "bias": delta_bias.detach().cpu().numpy().astype(np.float16),
    }
    del runtime
    torch.cuda.empty_cache()
    return trace, evaluation, factors, elapsed


def _aggregate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    base = [row["base"] for row in evaluations]
    adapted = [row["adapted"] for row in evaluations]
    return {
        "fold_count": len(evaluations),
        "base_mean_old_acc": float(np.mean([row["old_acc"] for row in base])),
        "base_mean_seen_new_acc": float(np.mean([row["seen_new_acc"] for row in base])),
        "base_mean_H": float(np.mean([row["H_old_new"] for row in base])),
        "base_worst_old_floor": min(row["old_floor"] for row in base),
        "base_worst_new_floor": min(row["new_floor"] for row in base),
        "adapted_mean_old_acc": float(np.mean([row["old_acc"] for row in adapted])),
        "adapted_mean_seen_new_acc": float(np.mean([row["seen_new_acc"] for row in adapted])),
        "adapted_mean_H": float(np.mean([row["H_old_new"] for row in adapted])),
        "adapted_worst_old_floor": min(row["old_floor"] for row in adapted),
        "adapted_worst_new_floor": min(row["new_floor"] for row in adapted),
    }


def run(enrollment_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _guard_enrollment_root(enrollment_root)
    output_dir.mkdir(parents=True)
    access_log: list[dict[str, Any]] = []
    manifest = _load_and_verify_manifest(enrollment_root, access_log)
    members = _member_map(manifest)
    runtime_path = _verified_member(enrollment_root, members, "sealed_feature_runtime.pt", "sealed_runtime", access_log)
    supports: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scenario in SCENARIOS:
        name = f"support_{scenario}.npz"
        path = _verified_member(enrollment_root, members, name, f"registered_support:{scenario}", access_log)
        with np.load(path, allow_pickle=False) as data:
            iq = data["support_leo_weak_iq"].astype(np.float32)
            labels = data["support_class_indices"].astype(np.int64)
        if iq.shape[0] != 110 or labels.shape != (110,) or not np.isfinite(iq).all():
            raise RuntimeError("support shape/finite guard failed")
        supports[scenario] = (iq, labels)
    candidates = [(rank, preset) for rank in RANKS for preset in LOSS_PRESETS]
    all_trace: list[dict[str, Any]] = []
    all_evaluations: list[dict[str, Any]] = []
    all_factors: dict[str, np.ndarray] = {}
    torch.cuda.reset_peak_memory_stats()
    for rank, preset in candidates:
        for scenario in SCENARIOS:
            iq, labels = supports[scenario]
            for fold in range(FOLDS):
                train_indices, validation_indices = _fold_indices(labels, fold)
                trace, evaluation, factors, _ = _fit_fold(
                    runtime_path,
                    iq[train_indices],
                    labels[train_indices],
                    iq[validation_indices],
                    labels[validation_indices],
                    scenario,
                    rank,
                    preset,
                    fold,
                )
                all_trace.extend(trace)
                all_evaluations.append(evaluation)
                prefix = f"rank{rank}__{preset}__{scenario}__fold{fold}"
                for factor_name, value in factors.items():
                    all_factors[f"{prefix}__{factor_name}"] = value
    candidate_rows = []
    for rank, preset in candidates:
        rows = [row for row in all_evaluations if row["rank"] == rank and row["preset"] == preset]
        aggregate = _aggregate(rows)
        aggregate["mean_H_gain"] = aggregate["adapted_mean_H"] - aggregate["base_mean_H"]
        gate_pass = (
            aggregate["mean_H_gain"] >= GATE_MIN_H_GAIN
            and aggregate["adapted_worst_old_floor"] >= aggregate["base_worst_old_floor"]
            and aggregate["adapted_worst_new_floor"] >= aggregate["base_worst_new_floor"]
        )
        candidate_rows.append({"rank": rank, "preset": preset, "gate_pass": gate_pass, **aggregate})
    selected = max(
        candidate_rows,
        key=lambda row: (
            row["gate_pass"],
            row["adapted_worst_old_floor"],
            row["adapted_worst_new_floor"],
            row["adapted_mean_H"],
            -row["rank"],
        ),
    )
    decision = "GO_SUPPORT_GATE" if selected["gate_pass"] else "NO_GO_SUPPORT_GATE"
    trace_path = output_dir / "support_fold_log.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in [*all_trace, *all_evaluations]:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    factors_path = output_dir / "fold_fp16_factors.npz"
    np.savez_compressed(factors_path, **all_factors)
    selector = {
        "schema": "cvs.phase2.d21_m6_support_fold_selector.v1",
        "decision": decision,
        "selection_boundary": "SUPPORT_ONLY_NO_QUERY",
        "candidates": candidate_rows,
        "selected_support_row": selected,
        "gate": {
            "minimum_mean_H_gain": GATE_MIN_H_GAIN,
            "old_floor_non_degradation": True,
            "new_floor_non_degradation": True,
        },
        "fold_protocol": "two class-balanced folds; each class 5 train and 5 validation rows",
        "query_used_for_selection": False,
        "full_support_refit_performed": False,
        "reason": "No full-support refit or downstream prediction is permitted in this support-only diagnostic task.",
    }
    selector_path = output_dir / "selector_lock.json"
    _json_dump(selector_path, selector)
    access_paths = [Path(row["path"]).resolve() for row in access_log]
    allowed_paths = {runtime_path.resolve(), (enrollment_root / "package_manifest.json").resolve()}
    allowed_paths.update((enrollment_root / f"support_{scenario}.npz").resolve() for scenario in SCENARIOS)
    if set(access_paths) != allowed_paths:
        raise RuntimeError("runtime input access audit escaped exact allowlist")
    proof = {
        "schema": "cvs.phase2.d21_m6_query_unreachable_proof.v1",
        "input_interface": {"accepted": ["enrollment_root", "output_dir"], "enrollment_root_required_suffix": "predictor/after/enrollment_only"},
        "exact_input_allowlist": sorted(str(path) for path in allowed_paths),
        "input_manifest_schema": "cvs.phase2.somph_predictor_bundle.v1",
        "input_manifest_member_allowlist_exact": True,
        "input_manifest_extra_member_rejected": True,
        "observed_input_accesses": access_log,
        "observed_equals_allowlist": True,
        "loaded_npz_members": ["support_leo_weak_iq", "support_class_indices"],
        "query_access": False,
        "query_fit": False,
        "query_truth_opened": False,
        "query_iq_access": False,
        "query_token_access": False,
        "truth_sidecar_access": False,
        "query_calibration": False,
        "query_selection": False,
        "query_early_stop": False,
        "query_rollback": False,
        "query_candidate_ranking": False,
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "score_operation_available": False,
        "prediction_artifact_emitted": False,
    }
    proof_path = output_dir / "query_unreachable_proof.json"
    _json_dump(proof_path, proof)
    resources = {
        "schema": "cvs.phase2.d21_m6_resources.v1",
        "exact_model_parameter_whitelist": [WEIGHT_NAME, BIAS_NAME],
        "base_parameter_shapes": {WEIGHT_NAME: [160, 160], BIAS_NAME: [160]},
        "candidate_trainable_parameters": {"rank2": 800, "rank4": 1440},
        "updated_original_parameters_after_merge": 25760,
        "fp16_factor_payload_bytes": {"rank2": 1600, "rank4": 2880},
        "selected_support_candidate_rank": int(selected["rank"]),
        "selected_fp16_factor_patch_bytes_upper_bound": int(2 * (320 * int(selected["rank"]) + 160)),
        "int8_registered_head_bytes_pre_registered_upper_bound": CLASS_COUNT * 160 + CLASS_COUNT * 2,
        "combined_patch_plus_int8_head_bytes_upper_bound": int(2 * (320 * int(selected["rank"]) + 160) + CLASS_COUNT * 160 + CLASS_COUNT * 2),
        "deployment_state_budget_bytes": 262144,
        "deployment_state_budget_arithmetic_pass": int(2 * (320 * int(selected["rank"]) + 160) + CLASS_COUNT * 160 + CLASS_COUNT * 2) <= 262144,
        "deployment_export_authorized": False,
        "final_patch_materialized": False,
        "int8_head_materialized_by_m6": False,
        "state_budget_claim_scope": "support-only pre-registered upper-bound audit; NO_GO forbids final refit and export",
        "adaptation_epochs_per_fold": EPOCHS,
        "adaptation_steps_per_fold": EPOCHS,
        "maximum_allowed_steps": MAX_STEPS,
        "optimizer": "SGD(lr=0.05,momentum=0)",
        "optimizer_state_persisted": False,
        "fold_fit_count": len(all_evaluations),
        "fold_fit_seconds_total": float(sum(row["elapsed_seconds"] for row in all_evaluations)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "merged_inference_added_MAC": 0,
        "full_rank_weight_delta_persisted": False,
        "support_fold_log_sha256": _sha256(trace_path),
        "selector_lock_sha256": _sha256(selector_path),
        "factor_archive_sha256": _sha256(factors_path),
    }
    _json_dump(output_dir / "resource_audit.json", resources)
    print("[M6-DECISION] " + json.dumps({"decision": decision, "selected": selected}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrollment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.enrollment_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
