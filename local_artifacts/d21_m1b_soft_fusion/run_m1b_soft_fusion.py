"""M1b soft old-state fusion using support-only rebuilt M1 theta_B/theta_C."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
M1_PATH = REPO_ROOT / "local_artifacts" / "d21_m1_dual_state" / "run_m1_dual_state.py"
SPEC = importlib.util.spec_from_file_location("d21_m1", M1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M1 implementation")
M1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1)
BASE = M1.BASE
SCENARIOS = M1.SCENARIOS
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
OFFSETS = (0.0, 0.01, 0.02)
ARMS = tuple(
    f"beta_{str(beta).replace('.', 'p')}__offset_{str(offset).replace('.', 'p')}"
    for beta in BETAS
    for offset in OFFSETS
)


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


def _arm_name(beta: float, offset: float) -> str:
    return f"beta_{str(beta).replace('.', 'p')}__offset_{str(offset).replace('.', 'p')}"


def _metrics(pred: np.ndarray, truth: np.ndarray, old_count: int) -> dict[str, float]:
    per_class = []
    for class_index in sorted(set(int(value) for value in truth.tolist())):
        mask = truth == class_index
        per_class.append((class_index, float(np.mean(pred[mask] == truth[mask]))))
    old = [value for index, value in per_class if index < old_count]
    new = [value for index, value in per_class if index >= old_count]
    old_acc = float(np.mean(pred[truth < old_count] == truth[truth < old_count]))
    new_acc = float(np.mean(pred[truth >= old_count] == truth[truth >= old_count]))
    return {
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "old_floor": min(old),
        "new_floor": min(new),
        "H_old_new": 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-12),
    }


def _support_scores(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = state["labels"]
    old_count = state["old_count"]
    old_mask = labels < old_count
    new_mask = ~old_mask
    b = BASE._weighted_rows(state["support"], state["theta_b"])
    c = BASE._weighted_rows(state["support"], state["theta_c"])
    b_old = b[old_mask]
    c_old = c[old_mask]
    c_new = c[new_mask]
    old_labels = labels[old_mask]
    new_labels = labels[new_mask] - old_count
    b_old_loo = BASE._loo_scores_np(b_old, old_labels, old_count)
    c_old_loo = BASE._loo_scores_np(c_old, old_labels, old_count)
    c_new_loo = BASE._loo_scores_np(c_new, new_labels, state["class_count"] - old_count)
    b_new_to_old = BASE._top1_scores(b[new_mask], b_old, old_labels, old_count)
    c_new_to_old = BASE._top1_scores(c[new_mask], c_old, old_labels, old_count)
    c_old_to_new = BASE._top1_scores(c[old_mask], c_new, new_labels, state["class_count"] - old_count)
    b_scores = np.empty((labels.shape[0], old_count), dtype=np.float32)
    c_old_scores = np.empty_like(b_scores)
    c_new_scores = np.empty((labels.shape[0], state["class_count"] - old_count), dtype=np.float32)
    b_scores[old_mask] = b_old_loo
    b_scores[new_mask] = b_new_to_old
    c_old_scores[old_mask] = c_old_loo
    c_old_scores[new_mask] = c_new_to_old
    c_new_scores[old_mask] = c_old_to_new
    c_new_scores[new_mask] = c_new_loo
    return b_scores, c_old_scores, c_new_scores


def _lock(states: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for beta in BETAS:
        for offset in OFFSETS:
            scene_rows = []
            for state in states:
                b_old, c_old, c_new = _support_scores(state)
                scores = np.concatenate(
                    [beta * b_old + (1.0 - beta) * c_old, c_new - offset], axis=1
                )
                pred = scores.argmax(axis=1)
                before_pred = b_old[state["labels"] < state["old_count"]].argmax(axis=1)
                metrics = _metrics(pred, state["labels"], state["old_count"])
                before_acc = float(
                    np.mean(before_pred == state["labels"][state["labels"] < state["old_count"]])
                )
                scene_rows.append(
                    {
                        "scenario": state["scenario"],
                        **metrics,
                        "old_acc_before_increment": before_acc,
                        "average_forgetting": before_acc - metrics["old_acc"],
                    }
                )
            rows.append(
                {
                    "beta": beta,
                    "new_offset": offset,
                    "scenario_rows": scene_rows,
                    "worst_old_floor": min(row["old_floor"] for row in scene_rows),
                    "worst_new_floor": min(row["new_floor"] for row in scene_rows),
                    "worst_h": min(row["H_old_new"] for row in scene_rows),
                    "worst_forgetting": max(row["average_forgetting"] for row in scene_rows),
                }
            )
    selected = max(
        rows,
        key=lambda row: (
            row["worst_old_floor"],
            row["worst_new_floor"],
            row["worst_h"],
            -max(row["worst_forgetting"], 0.0),
            -abs(row["new_offset"]),
        ),
    )
    return {"beta": selected["beta"], "new_offset": selected["new_offset"]}, rows


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
        raise RuntimeError("M1b requires the fixed K10/new5 6->11 capsule")
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
        with np.load(before_apply / f"query_{scenario}.npz", allow_pickle=False) as before_file:
            before_tokens = before_file["query_tokens"].astype(str)
        support_z, _ = BASE._extract(runtime, support_iq)
        query_z, _ = BASE._extract(runtime, query_iq)
        support_fp32 = BASE._fixed_representation(support_z, support_iq)
        query = BASE._fixed_representation(query_z, query_iq)
        support, _, _ = BASE._int8_support_roundtrip(support_fp32)
        old_mask = labels < old_count
        theta_b, trace_b = BASE._fit_metric(
            support[old_mask],
            labels[old_mask],
            old_count,
            old_count,
            BASE.CONFIGS["L6_diag_floor"],
            scenario=scenario,
            phase="M1b_rebuild_theta_B",
            teacher_old_log_weight=None,
        )
        theta_c, trace_c = M1._fit_theta_c(support, labels, old_count, theta_b, scenario)
        traces.extend([{"candidate": "theta_B", **row} for row in trace_b])
        traces.extend(trace_c)
        token_to_row = {token: row for row, token in enumerate(query_tokens.tolist())}
        states.append(
            {
                "scenario": scenario,
                "support": support,
                "labels": labels,
                "old_count": old_count,
                "class_count": class_count,
                "query": query,
                "query_tokens": query_tokens,
                "before_tokens": before_tokens,
                "before_indices": np.asarray([token_to_row[token] for token in before_tokens], dtype=np.int64),
                "theta_b": theta_b,
                "theta_c": theta_c,
            }
        )
    selected, support_grid = _lock(states)
    print("[M1B-LOCK] " + json.dumps(selected, sort_keys=True), flush=True)
    arrays: dict[str, np.ndarray] = {}
    latency = []
    for beta in BETAS:
        for offset in OFFSETS:
            arm = _arm_name(beta, offset)
            after_predictions = []
            after_tokens = []
            after_scenarios = []
            before_predictions = []
            before_tokens = []
            before_scenarios = []
            for state in states:
                labels = state["labels"]
                old_mask = labels < state["old_count"]
                b_support = BASE._weighted_rows(state["support"], state["theta_b"])
                c_support = BASE._weighted_rows(state["support"], state["theta_c"])
                b_query = BASE._weighted_rows(state["query"], state["theta_b"])
                c_query = BASE._weighted_rows(state["query"], state["theta_c"])
                start = time.perf_counter_ns()
                b_old = BASE._top1_scores(
                    b_query, b_support[old_mask], labels[old_mask], state["old_count"]
                )
                c_old = BASE._top1_scores(
                    c_query, c_support[old_mask], labels[old_mask], state["old_count"]
                )
                c_new = BASE._top1_scores(
                    c_query,
                    c_support[~old_mask],
                    labels[~old_mask] - state["old_count"],
                    state["class_count"] - state["old_count"],
                )
                latency.append((time.perf_counter_ns() - start) / 1e6 / state["query"].shape[0])
                scores = np.concatenate([beta * b_old + (1.0 - beta) * c_old, c_new - offset], axis=1)
                after_pred = scores.argmax(axis=1).astype(np.int64)
                before_pred = b_old[state["before_indices"]].argmax(axis=1).astype(np.int64)
                after_predictions.append(after_pred)
                after_tokens.append(state["query_tokens"])
                after_scenarios.append(np.full(after_pred.shape[0], state["scenario"]))
                before_predictions.append(before_pred)
                before_tokens.append(state["before_tokens"])
                before_scenarios.append(np.full(before_pred.shape[0], state["scenario"]))
            arrays[f"{arm}__after_predictions"] = np.concatenate(after_predictions)
            arrays[f"{arm}__after_tokens"] = np.concatenate(after_tokens)
            arrays[f"{arm}__after_scenarios"] = np.concatenate(after_scenarios)
            arrays[f"{arm}__before_predictions"] = np.concatenate(before_predictions)
            arrays[f"{arm}__before_tokens"] = np.concatenate(before_tokens)
            arrays[f"{arm}__before_scenarios"] = np.concatenate(before_scenarios)
    arrays["schema_json"] = np.asarray(
        json.dumps(
            {
                "schema": "cvs.phase2.d21_m1b_soft_fusion_predictions.v1",
                "arms": ARMS,
                "support_only_selection": True,
                "query_fit": False,
                "query_role_oracle": False,
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
    receipt = {
        "schema": "cvs.phase2.d21_m1b_soft_fusion_receipt.v1",
        "prediction_sha256": _sha256(output),
        "loss_trace_sha256": _sha256(trace_path),
        "loss_trace_record_count": len(traces),
        "selected_from_support_only": selected,
        "support_beta_offset_grid": support_grid,
        "complete_beta_pareto_predictions_emitted": True,
        "resources": {
            "trainable_parameters": 2 * BASE.DIM,
            "adaptation_epochs_B_plus_C": 2 * M1.EPOCHS,
            "persistent_state_bytes_after": 30428 + 8,
            "query_classifier_MAC_after": 45585,
            "classifier_ms_per_sample_mean": float(np.mean(latency)),
            "classifier_ms_per_sample_p95": float(np.quantile(latency, 0.95)),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
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
        "schema": "cvs.phase2.d21_m1b_soft_fusion_score.v1",
        "prediction_sha256": _sha256(prediction),
        "truth_sidecar_sha256": _sha256(truth_path),
        "query_truth_joined_only_after_immutable_predictions": True,
        "scorer_feedback_to_predictor": False,
        "arms": {},
    }
    with np.load(prediction, allow_pickle=False) as data:
        for arm in ARMS:
            scene_rows = []
            pooled_before_pred = []
            pooled_before_truth = []
            pooled_after_pred = []
            pooled_after_truth = []
            pooled_rows = []
            for scenario in SCENARIOS:
                bm = data[f"{arm}__before_scenarios"] == scenario
                am = data[f"{arm}__after_scenarios"] == scenario
                btokens = data[f"{arm}__before_tokens"][bm].astype(str)
                atokens = data[f"{arm}__after_tokens"][am].astype(str)
                brows = [truth_by_token[token] for token in btokens]
                arows = [truth_by_token[token] for token in atokens]
                bt = np.asarray([row["true_class_index"] for row in brows], dtype=np.int64)
                at = np.asarray([row["true_class_index"] for row in arows], dtype=np.int64)
                bp = data[f"{arm}__before_predictions"][bm]
                ap = data[f"{arm}__after_predictions"][am]
                old_count = len(set(bt.tolist()))
                metrics = _metrics(ap, at, old_count)
                old_mask = at < old_count
                old_class = _class_metrics(ap[old_mask], [row for row in arows if int(row["true_class_index"]) < old_count])
                new_class = _class_metrics(ap[~old_mask], [row for row in arows if int(row["true_class_index"]) >= old_count])
                before_acc = float(np.mean(bp == bt))
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
                    }
                )
                pooled_before_pred.append(bp); pooled_before_truth.append(bt)
                pooled_after_pred.append(ap); pooled_after_truth.append(at); pooled_rows.extend(arows)
            bp=np.concatenate(pooled_before_pred); bt=np.concatenate(pooled_before_truth)
            ap=np.concatenate(pooled_after_pred); at=np.concatenate(pooled_after_truth)
            old_count=len(set(bt.tolist())); metrics=_metrics(ap,at,old_count); old_mask=at<old_count
            old_class=_class_metrics(ap[old_mask],[row for row in pooled_rows if int(row["true_class_index"])<old_count])
            new_class=_class_metrics(ap[~old_mask],[row for row in pooled_rows if int(row["true_class_index"])>=old_count])
            before_acc=float(np.mean(bp==bt))
            result["arms"][arm]={
                "scenario_rows":scene_rows,
                "aggregate":{
                    "old_acc_before_increment":before_acc,
                    "old_acc":metrics["old_acc"],
                    "min_old_class_acc":min(row["accuracy"] for row in old_class.values()),
                    "seen_new_acc":metrics["seen_new_acc"],
                    "min_seen_new_class_acc":min(row["accuracy"] for row in new_class.values()),
                    "H_old_new":metrics["H_old_new"],
                    "average_forgetting":before_acc-metrics["old_acc"],
                },
            }
    _json_dump(output,result)


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    pred=sub.add_parser("predict"); pred.add_argument("--capsule",type=Path,required=True); pred.add_argument("--output",type=Path,required=True)
    scorer=sub.add_parser("score"); scorer.add_argument("--prediction",type=Path,required=True); scorer.add_argument("--truth",type=Path,required=True); scorer.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="predict": predict(args.capsule.resolve(),args.output.resolve())
    else: score(args.prediction.resolve(),args.truth.resolve(),args.output.resolve())


if __name__=="__main__": main()
