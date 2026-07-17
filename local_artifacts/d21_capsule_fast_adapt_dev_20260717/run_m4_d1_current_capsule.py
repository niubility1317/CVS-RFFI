"""Legal current-capsule rerun of the historical D1 lightweight head.

Prediction fits only registered support. Query truth is opened solely by the
separate scorer inherited from ``run_capsule_fast_adapt_dev``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_capsule_fast_adapt_dev as base

from cvsrffi.stage2_diag_cosine_exploration import rf_statistics, spectral_logmag_sketch


METHOD = "M4_d1_a1_diag_classweights"
FEATURE_DIM = 288
EPOCHS = 20


def _a1(zid: np.ndarray, iq: np.ndarray) -> np.ndarray:
    fft = spectral_logmag_sketch(iq, dim=96)
    rf = rf_statistics(iq)
    if zid.shape[1] != 160 or fft.shape[1] != 96 or rf.shape[1] != 32:
        raise RuntimeError(f"unexpected A1 dimensions: {zid.shape}, {fft.shape}, {rf.shape}")
    return base._normalise(np.concatenate([zid, fft, 4.0 * rf], axis=1))


def _fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    old_count: int,
    *,
    initial_state: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).cuda()
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).cuda()
    masks = [y == class_index for class_index in range(class_count)]
    raw_prototypes = torch.stack(
        [torch.nn.functional.normalize(x[mask].mean(dim=0), dim=0) for mask in masks], dim=0
    )
    if initial_state is None:
        theta_init = np.zeros(FEATURE_DIM, dtype=np.float32)
        weight_init = raw_prototypes.detach().cpu().numpy().astype(np.float32)
    else:
        theta_init = np.asarray(initial_state["theta"], dtype=np.float32)
        weight_init = raw_prototypes.detach().cpu().numpy().astype(np.float32)
        previous = np.asarray(initial_state["weights"], dtype=np.float32)
        weight_init[: previous.shape[0]] = previous
    theta = torch.from_numpy(theta_init).cuda().detach().clone().requires_grad_(True)
    weights = torch.from_numpy(weight_init).cuda().detach().clone().requires_grad_(True)
    theta_ref = torch.from_numpy(theta_init).cuda()
    weight_ref = torch.nn.functional.normalize(torch.from_numpy(weight_init).cuda(), dim=1)
    old_rows = y < old_count
    with torch.no_grad():
        ref_scale = torch.exp(torch.clamp(theta_ref, -1.0, 1.0))
        ref_old = torch.nn.functional.normalize(x[old_rows] * ref_scale[None, :], dim=1)
        ref_pair = ref_old @ ref_old.T
    optimizer = torch.optim.Adam([theta, weights], lr=0.03)
    trace: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(torch.clamp(theta, -1.0, 1.0))
        transformed = torch.nn.functional.normalize(x * scale[None, :], dim=1)
        normal_weights = torch.nn.functional.normalize(weights, dim=1)
        logits = transformed @ normal_weights.T
        sample_ce = torch.nn.functional.cross_entropy(logits / 0.07, y, reduction="none")
        all_class_ce = sample_ce.mean()
        class_ce = torch.stack([sample_ce[mask].mean() for mask in masks])
        cvar = torch.topk(class_ce, k=min(2, class_count), largest=True).values.mean()
        current_prototypes = torch.stack(
            [torch.nn.functional.normalize(transformed[mask].mean(dim=0), dim=0) for mask in masks],
            dim=0,
        )
        prototype_anchor = (1.0 - torch.sum(normal_weights * current_prototypes, dim=1)).mean()
        pair_loss = torch.mean(
            ((transformed[old_rows] @ transformed[old_rows].T) - ref_pair).square()
        )
        old_weight_anchor = torch.mean(
            (normal_weights[:old_count] - weight_ref[:old_count]).square()
        )
        if class_count > old_count:
            old_indices = torch.nonzero(old_rows, as_tuple=False).squeeze(1)
            old_true = logits[old_indices, y[old_rows]]
            max_new = logits[old_rows, old_count:].max(dim=1).values
            invasion = torch.relu(max_new - old_true + 0.01).mean()
        else:
            invasion = torch.zeros((), device="cuda")
        theta_reg = torch.mean((theta - theta_ref).square())
        loss = (
            all_class_ce
            + 0.2 * cvar
            + 0.1 * prototype_anchor
            + 0.1 * pair_loss
            + 0.1 * old_weight_anchor
            + 0.2 * invasion
            + 0.01 * theta_reg
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prediction = logits.argmax(dim=1)
            per_class = [
                torch.mean((prediction[mask] == y[mask]).float()).item() for mask in masks
            ]
            old_acc = torch.mean((prediction[old_rows] == y[old_rows]).float()).item()
            new_acc = (
                torch.mean((prediction[~old_rows] == y[~old_rows]).float()).item()
                if class_count > old_count
                else 0.0
            )
        trace.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "all_registered_class_cross_entropy": float(all_class_ce.detach().item()),
                "cvar_top2_class_ce": float(cvar.detach().item()),
                "prototype_anchor": float(prototype_anchor.detach().item()),
                "old_pair_preservation_mse": float(pair_loss.detach().item()),
                "old_class_weight_anchor_mse": float(old_weight_anchor.detach().item()),
                "old_invasion_hinge": float(invasion.detach().item()),
                "theta_initial_state_regularization": float(theta_reg.detach().item()),
                "support_accuracy": float((prediction == y).float().mean().item()),
                "support_old_accuracy": float(old_acc),
                "support_new_accuracy": float(new_acc),
                "support_class_floor": float(min(per_class)),
            }
        )
    return {
        "theta": theta.detach().cpu().numpy().astype(np.float32),
        "weights": torch.nn.functional.normalize(weights.detach(), dim=1)
        .cpu()
        .numpy()
        .astype(np.float32),
    }, trace


def _predict_head(features: np.ndarray, state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scale = np.exp(np.clip(state["theta"], -1.0, 1.0)).astype(np.float32)
    transformed = base._normalise(features * scale[None, :])
    scores = transformed @ state["weights"].T
    return scores.argmax(axis=1).astype(np.int64), transformed


def predict(capsule: Path, output: Path) -> None:
    after_enrollment = capsule / "predictor" / "after" / "enrollment_only"
    after_apply = capsule / "predictor" / "after" / "apply_only_staging"
    before_enrollment = capsule / "predictor" / "before" / "enrollment_only"
    before_apply = capsule / "predictor" / "before" / "apply_only_staging"
    after_manifest = json.loads((after_enrollment / "package_manifest.json").read_text(encoding="utf-8"))
    before_manifest = json.loads((before_enrollment / "package_manifest.json").read_text(encoding="utf-8"))
    class_count = int(after_manifest["registered_class_count"])
    old_count = int(before_manifest["registered_class_count"])
    runtime_path = after_enrollment / "sealed_feature_runtime.pt"
    runtime = torch.jit.load(str(runtime_path)).eval()
    arrays: dict[str, np.ndarray] = {}
    loss_traces: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    after_predictions: list[np.ndarray] = []
    after_tokens: list[np.ndarray] = []
    after_scenarios: list[np.ndarray] = []
    before_predictions: list[np.ndarray] = []
    before_tokens: list[np.ndarray] = []
    before_scenarios: list[np.ndarray] = []
    classifier_latencies: list[tuple[float, float]] = []
    torch.cuda.reset_peak_memory_stats()
    for scenario in base.SCENARIOS:
        with np.load(after_enrollment / f"support_{scenario}.npz", allow_pickle=False) as src:
            support_iq = src["support_leo_weak_iq"]
            support_labels = src["support_class_indices"].astype(np.int64)
        with np.load(after_apply / f"query_{scenario}.npz", allow_pickle=False) as src:
            query_iq = src["query_leo_weak_iq"]
            query_tokens = src["query_tokens"].astype(str)
        with np.load(before_apply / f"query_{scenario}.npz", allow_pickle=False) as src:
            scene_before_tokens = src["query_tokens"].astype(str)
        support_zid, support_forward = base._extract(runtime, support_iq)
        query_zid, query_forward = base._extract(runtime, query_iq)
        descriptor_start = time.perf_counter()
        support_a1 = _a1(support_zid, support_iq)
        support_descriptor_seconds = time.perf_counter() - descriptor_start
        descriptor_start = time.perf_counter()
        query_a1 = _a1(query_zid, query_iq)
        query_descriptor_seconds = time.perf_counter() - descriptor_start
        old_mask = support_labels < old_count
        before_state, before_trace = _fit_head(
            support_a1[old_mask], support_labels[old_mask], old_count, old_count
        )
        after_state, after_trace = _fit_head(
            support_a1,
            support_labels,
            class_count,
            old_count,
            initial_state=before_state,
        )
        pred_after, transformed_after = _predict_head(query_a1, after_state)
        token_to_index = {token: index for index, token in enumerate(query_tokens.tolist())}
        before_indices = np.asarray([token_to_index[token] for token in scene_before_tokens], dtype=np.int64)
        pred_before, _ = _predict_head(query_a1[before_indices], before_state)
        after_predictions.append(pred_after)
        after_tokens.append(query_tokens)
        after_scenarios.append(np.full(pred_after.shape[0], scenario))
        before_predictions.append(pred_before)
        before_tokens.append(scene_before_tokens)
        before_scenarios.append(np.full(pred_before.shape[0], scenario))
        classifier_latencies.append(base._classifier_latency(transformed_after, after_state["weights"]))
        loss_traces[scenario] = {
            "before_old_registered_support_only": before_trace,
            "after_all_registered_support_only_initialised_from_before": after_trace,
        }
        timings[scenario] = {
            "support_backbone_ms_per_sample": support_forward * 1000 / support_iq.shape[0],
            "query_backbone_ms_per_sample": query_forward * 1000 / query_iq.shape[0],
            "support_fft96_plus_rf32_ms_per_sample": support_descriptor_seconds * 1000 / support_iq.shape[0],
            "query_fft96_plus_rf32_ms_per_sample": query_descriptor_seconds * 1000 / query_iq.shape[0],
        }
    arrays[f"{METHOD}__after_predictions"] = np.concatenate(after_predictions)
    arrays[f"{METHOD}__after_tokens"] = np.concatenate(after_tokens)
    arrays[f"{METHOD}__after_scenarios"] = np.concatenate(after_scenarios)
    arrays[f"{METHOD}__before_predictions"] = np.concatenate(before_predictions)
    arrays[f"{METHOD}__before_tokens"] = np.concatenate(before_tokens)
    arrays[f"{METHOD}__before_scenarios"] = np.concatenate(before_scenarios)
    arrays["schema_json"] = np.asarray(
        json.dumps(
            {
                "schema": "cvs.phase2.d21_m4_d1_current_capsule_predictions.v1",
                "truth_or_role_in_predictor_input": False,
                "query_fit": False,
                "query_quota_or_global_assignment": False,
                "all_registered_classes_per_sample": True,
                "methods": [METHOD],
                "registered_feature": "z160_plus_fft96_plus_4x_rf32_from_same_received_iq",
            },
            sort_keys=True,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **arrays)
    loss_path = output.with_suffix(".loss_trace.json")
    base._json_dump(
        loss_path,
        {
            "schema": "cvs.phase2.d21_m4_d1_support_loss_trace.v1",
            "support_only": True,
            "query_access": False,
            "epochs": EPOCHS,
            "feature_dim": FEATURE_DIM,
            "parameter_count_before": FEATURE_DIM + old_count * FEATURE_DIM,
            "parameter_count_after": FEATURE_DIM + class_count * FEATURE_DIM,
            "scenarios": loss_traces,
        },
    )
    parameter_before = FEATURE_DIM + old_count * FEATURE_DIM
    parameter_after = FEATURE_DIM + class_count * FEATURE_DIM
    resource = {
        "trainable_parameters_before": parameter_before,
        "trainable_parameters_after": parameter_after,
        "adaptation_epochs": EPOCHS,
        "adaptation_type": "SUPPORT_ONLY_SHARED_DIAGONAL_PLUS_REGISTERED_CLASS_WEIGHTS",
        "persistent_state_bytes_before": parameter_before * 2,
        "persistent_state_bytes_after": parameter_after * 2,
        "query_classifier_MAC_before": parameter_before,
        "query_classifier_MAC_after": parameter_after,
        "adaptation_forward_MAC_estimate_after": EPOCHS * class_count * 10 * parameter_after,
        "classifier_ms_per_sample_mean": float(np.mean([v[0] for v in classifier_latencies])),
        "classifier_ms_per_sample_p95": float(np.max([v[1] for v in classifier_latencies])),
        "dense_query_graph": False,
    }
    base._json_dump(
        output.with_suffix(".receipt.json"),
        {
            "schema": "cvs.phase2.d21_m4_d1_current_capsule_receipt.v1",
            "prediction_sha256": base._sha256(output),
            "capsule_offline_receipt_sha256": base._sha256(capsule / "offline_build_receipt.json"),
            "sealed_runtime_sha256": base._sha256(runtime_path),
            "resources": {METHOD: resource},
            "timings": timings,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "loss_trace_relative_path": loss_path.name,
            "loss_trace_sha256": base._sha256(loss_path),
            "query_truth_opened": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
        },
    )


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
        base.METHODS = (METHOD,)
        base.score(args.prediction.resolve(), args.truth.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
