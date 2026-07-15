"""Run and score a query-truth-isolated EvidenceNorm Stage2-C diagnostic.

The ``predict`` subcommand can only see the sealed predictor package.  It fits
the class-symmetric closed-form head from registered LEO-weak support and
publishes a truth-free NPZ.  The separate ``score`` subcommand verifies that
prediction before opening the scorer-only truth sidecar.

This is a local algorithm diagnostic, not a replacement for the Landlock
formal runner or an independent-confirmation performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
for value in (str(ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_stage2_predictor_bundle,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    SYMMETRIC_HEAD_SCHEMA_V2,
    build_formal_support_state,
    load_json_artifact_same_fd,
    load_torchscript_backbone_same_fd,
    predict_formal_scenario_streams,
)


PREDICTION_SCHEMA = "cvs.phase2.evidencenorm_diagnostic_prediction.v1"
SCORING_SCHEMA = "cvs.phase2.evidencenorm_diagnostic_score.v1"
PREDICTION_ARRAYS = {
    "query_tokens",
    "scenarios",
    "candidate_after",
    "candidate_before",
    "identity_after",
    "identity_before",
    "direct",
    "shared_view_counts",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(raw)


def build_evidence_head_config(
    base: Mapping[str, Any],
    *,
    negative_quantile: float,
    prior_physical_shots: float,
    scale_floor: float,
    inverse_scale_cap: float,
) -> dict[str, Any]:
    expected_base_keys = {
        "use_alignment",
        "prototype_rule",
        "ridge",
        "gram_mix",
        "uncertainty_penalty",
    }
    selected = base.get("selected")
    if (
        base.get("schema") != "cvs.phase2.symmetric_locked_head.v1"
        or base.get("mode") != "three_leo_support_symmetric_locked"
        or base.get("storage_dtype") != "fp16"
        or not isinstance(selected, Mapping)
        or set(selected) != expected_base_keys
    ):
        raise ValueError("EvidenceNorm requires the exact strict v1 support-head base")
    if (
        selected.get("use_alignment") is not False
        or selected.get("ridge") is not None
        or selected.get("gram_mix") != 0.0
        or selected.get("uncertainty_penalty") != 0.0
    ):
        raise ValueError("EvidenceNorm base must have no alignment/Gram/uncertainty transform")
    return {
        "schema": SYMMETRIC_HEAD_SCHEMA_V2,
        "mode": "three_leo_support_symmetric_evidence_locked",
        "selected": {
            **dict(selected),
            "evidence_calibration": {
                "mode": "robust_lopo_class_symmetric",
                "negative_quantile": float(negative_quantile),
                "prior_physical_shots": float(prior_physical_shots),
                "scale_floor": float(scale_floor),
                "inverse_scale_cap": float(inverse_scale_cap),
            },
        },
        "source_feature_mean": list(base["source_feature_mean"]),
        "source_feature_std": list(base["source_feature_std"]),
        "variance_floor": float(base["variance_floor"]),
        "storage_dtype": "fp16",
    }


def _member_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ValueError("predictor manifest members are missing")
    return {str(item["artifact_role"]): dict(item) for item in members}


def run_predict(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    expected_seal = str(args.expected_seal_sha256).lower()
    if sha256_file(args.detached_seal) != expected_seal:
        raise ValueError("detached seal SHA does not match the explicit trust root")

    support, query, manifest, package_audit = load_verified_stage2_predictor_bundle(
        args.predictor_root,
        detached_seal_path=args.detached_seal,
        expected_seal_sha256=expected_seal,
    )
    if tuple(manifest["target_channel_scenarios"]) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("diagnostic package does not contain the exact formal LEO scenarios")
    members = _member_map(manifest)
    adapter = load_json_artifact_same_fd(args.predictor_root, members["adapter"])
    base_head = load_json_artifact_same_fd(args.predictor_root, members["head"])
    tta = load_json_artifact_same_fd(args.predictor_root, members["tta_policy"])
    head = build_evidence_head_config(
        base_head,
        negative_quantile=float(args.negative_quantile),
        prior_physical_shots=float(args.prior_physical_shots),
        scale_floor=float(args.scale_floor),
        inverse_scale_cap=float(args.inverse_scale_cap),
    )
    device = torch.device(args.device)
    candidate_model = load_torchscript_backbone_same_fd(
        args.predictor_root, members["checkpoint"], device=device
    )
    base_model = load_torchscript_backbone_same_fd(
        args.predictor_root, members["base_checkpoint"], device=device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    state = build_formal_support_state(
        candidate_model,
        base_model,
        support,
        scenarios=FORMAL_LEO_WEAK_SCENARIOS,
        k_shot=int(args.k_shot),
        registered_class_count=int(manifest["registered_class_count"]),
        new_class_count=int(manifest["new_class_count"]),
        adapter_config=adapter,
        head_config=head,
        device=device,
        batch_size=int(args.batch_size),
    )
    old_class_count = int(manifest["registered_class_count"]) - int(
        manifest["new_class_count"]
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in PREDICTION_ARRAYS}
    scenario_resources: dict[str, Any] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        predictions, resources = predict_formal_scenario_streams(
            candidate_model,
            base_model,
            query[scenario],
            state,
            scenario=scenario,
            old_class_count=old_class_count,
            adapter_config=adapter,
            tta_config=tta,
            device=device,
            batch_size=int(args.batch_size),
        )
        tokens = np.asarray(query[scenario]["query_tokens"]).astype(str)
        parts["query_tokens"].append(tokens)
        parts["scenarios"].append(np.asarray([scenario] * len(tokens)))
        for name, values in predictions.items():
            parts[name].append(np.asarray(values))
        scenario_resources[scenario] = resources
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "truth_free_predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    prediction_sha = sha256_file(prediction_path)
    after_diagnostics = state["candidate_after"]["evidence_diagnostics"]
    before_diagnostics = state["candidate_before"]["evidence_diagnostics"]
    result = {
        "schema": PREDICTION_SCHEMA,
        "status": "PREDICTION_FROZEN_TRUTH_UNREAD",
        "predictor_package_root": str(Path(args.predictor_root).resolve()),
        "predictor_package_root_sha256": str(manifest["package_root_sha256"]),
        "predictor_package_seal_sha256": expected_seal,
        "prediction_npz": prediction_path.name,
        "prediction_npz_sha256": prediction_sha,
        "prediction_row_count": int(len(arrays["query_tokens"])),
        "receiver": str(manifest["receiver"]),
        "seed": int(manifest["seed"]),
        "k_shot": int(args.k_shot),
        "old_class_count": old_class_count,
        "new_class_count": int(manifest["new_class_count"]),
        "registered_class_count": int(manifest["registered_class_count"]),
        "head_config": head,
        "head_config_sha256": canonical_sha256(head),
        "head_after_diagnostics": after_diagnostics,
        "head_before_diagnostics": before_diagnostics,
        "scenario_resources": scenario_resources,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "on_orbit_trainable_parameters": 0,
        "on_orbit_optimizer_steps": 0,
        "on_orbit_adapt_epochs": 0,
        "head_incremental_state_bytes_fp16": int(after_diagnostics["state_bytes_fp16"]),
        "query_truth_access": False,
        "query_role_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "package_preopen_audit": package_audit,
    }
    _write_json_new(output / "prediction_manifest.json", result)
    return result


def _accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(prediction == truth)) if len(truth) else float("nan")


def _harmonic(old: float, new: float) -> float:
    return float(2.0 * old * new / (old + new)) if old + new > 0.0 else 0.0


def summarize_predictions(
    arrays: Mapping[str, np.ndarray],
    truth_rows: Sequence[Mapping[str, Any]],
    *,
    old_class_count: int,
) -> dict[str, Any]:
    truth_by_token = {str(row["query_token"]): dict(row) for row in truth_rows}
    tokens = np.asarray(arrays["query_tokens"]).astype(str)
    if any(token not in truth_by_token for token in tokens):
        raise ValueError("truth sidecar does not cover every frozen prediction token")
    truth = np.asarray(
        [int(truth_by_token[token]["true_class_index"]) for token in tokens], dtype=np.int64
    )
    roles = np.asarray(
        [str(truth_by_token[token]["evaluation_role"]) for token in tokens]
    )
    scenarios = np.asarray(arrays["scenarios"]).astype(str)
    streams = {
        name: np.asarray(arrays[name], dtype=np.int64)
        for name in (
            "candidate_after",
            "candidate_before",
            "identity_after",
            "identity_before",
            "direct",
        )
    }
    rows: list[dict[str, Any]] = []
    per_old_class: list[dict[str, Any]] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        scenario_mask = scenarios == scenario
        old_mask = scenario_mask & (roles == "target_old")
        new_mask = scenario_mask & (roles == "target_new")
        old_truth = truth[old_mask]
        new_truth = truth[new_mask]
        old_before = _accuracy(streams["candidate_before"][old_mask], old_truth)
        old_after = _accuracy(streams["candidate_after"][old_mask], old_truth)
        seen_new = _accuracy(streams["candidate_after"][new_mask], new_truth)
        direct = _accuracy(streams["direct"][old_mask], old_truth)
        class_accuracies: list[float] = []
        for class_index in range(int(old_class_count)):
            mask = old_mask & (truth == class_index)
            if not np.any(mask):
                raise ValueError(f"old class lacks query evidence: {class_index}")
            before = _accuracy(streams["candidate_before"][mask], truth[mask])
            after = _accuracy(streams["candidate_after"][mask], truth[mask])
            class_accuracies.append(after)
            labels = {
                str(truth_by_token[token]["transmitter_label"])
                for token in tokens[mask]
            }
            per_old_class.append(
                {
                    "scenario": scenario,
                    "class_index": class_index,
                    "transmitter_label": sorted(labels)[0],
                    "sample_count": int(np.sum(mask)),
                    "old_acc_before_increment": before,
                    "old_acc_after_increment": after,
                    "average_forgetting": before - after,
                }
            )
        old_to_new = float(
            np.mean(streams["candidate_after"][old_mask] >= int(old_class_count))
        )
        new_to_old = float(
            np.mean(streams["candidate_after"][new_mask] < int(old_class_count))
        )
        new_wrong_new = float(
            np.mean(
                (streams["candidate_after"][new_mask] >= int(old_class_count))
                & (streams["candidate_after"][new_mask] != new_truth)
            )
        )
        counts = np.asarray(arrays["shared_view_counts"], dtype=np.int64)[scenario_mask]
        rows.append(
            {
                "scenario": scenario,
                "old_acc_before_increment": old_before,
                "old_acc_after_increment": old_after,
                "direct_adv3b02_old_acc": direct,
                "delta_vs_direct_adv3b02": old_after - direct,
                "seen_new_acc": seen_new,
                "H_old_new": _harmonic(old_after, seen_new),
                "min_old_class_acc": float(min(class_accuracies)),
                "average_forgetting": old_before - old_after,
                "old_to_new_rate": old_to_new,
                "new_to_old_rate": new_to_old,
                "new_to_wrong_new_rate": new_wrong_new,
                "mean_backbone_forwards": float(np.mean(counts)),
                "p95_backbone_forwards": int(np.percentile(counts, 95, method="higher")),
                "view1_rate": float(np.mean(counts == 1)),
                "view3_rate": float(np.mean(counts == 3)),
                "view5_rate": float(np.mean(counts == 5)),
            }
        )
    keys = (
        "old_acc_before_increment",
        "old_acc_after_increment",
        "direct_adv3b02_old_acc",
        "delta_vs_direct_adv3b02",
        "seen_new_acc",
        "H_old_new",
        "average_forgetting",
        "old_to_new_rate",
        "new_to_old_rate",
        "new_to_wrong_new_rate",
        "mean_backbone_forwards",
        "view1_rate",
        "view3_rate",
        "view5_rate",
    )
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    aggregate["mean_scenario_min_old_class_acc"] = float(
        np.mean([row["min_old_class_acc"] for row in rows])
    )
    aggregate["global_min_old_class_acc"] = float(
        min(item["old_acc_after_increment"] for item in per_old_class)
    )
    aggregate["p95_backbone_forwards"] = int(
        np.percentile(np.asarray(arrays["shared_view_counts"]), 95, method="higher")
    )
    return {"scenario_rows": rows, "per_old_class": per_old_class, "aggregate": aggregate}


def _markdown_score(result: Mapping[str, Any]) -> str:
    lines = [
        "# EvidenceNorm Round1单cell诊断",
        "",
        "|场景|注册前old|注册后old|direct|seen-new|H|最低旧类|遗忘|平均View|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["metrics"]["scenario_rows"]:
        lines.append(
            "|{scenario}|{old_acc_before_increment:.2%}|{old_acc_after_increment:.2%}|"
            "{direct_adv3b02_old_acc:.2%}|{seen_new_acc:.2%}|{H_old_new:.2%}|"
            "{min_old_class_acc:.2%}|{average_forgetting:.2%}|"
            "{mean_backbone_forwards:.3f}|".format(**row)
        )
    row = result["metrics"]["aggregate"]
    lines.extend(
        [
            "",
            "## 三场景等权聚合",
            "",
            "|注册前old|注册后old|direct|相对direct|seen-new|H|全局最低旧类|遗忘|平均View|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            "|{old_acc_before_increment:.2%}|{old_acc_after_increment:.2%}|"
            "{direct_adv3b02_old_acc:.2%}|{delta_vs_direct_adv3b02:+.2%}|"
            "{seen_new_acc:.2%}|{H_old_new:.2%}|{global_min_old_class_acc:.2%}|"
            "{average_forgetting:.2%}|{mean_backbone_forwards:.3f}|".format(**row),
            "",
            "该结果是单receiver×单seed×20新类本地算法诊断，不是独立确认矩阵或部署成功声明。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite score output: {output}")
    prediction_root = Path(args.prediction_root)
    manifest = _read_json(prediction_root / "prediction_manifest.json")
    if manifest.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("prediction manifest schema mismatch")
    prediction_path = prediction_root / str(manifest["prediction_npz"])
    if sha256_file(prediction_path) != manifest["prediction_npz_sha256"]:
        raise ValueError("frozen prediction SHA mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        if set(payload.files) != PREDICTION_ARRAYS:
            raise ValueError("truth-free prediction exact array schema drift")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    scorer_root = Path(args.scorer_root)
    scoring_manifest = _read_json(scorer_root / "scoring_manifest.json")
    truth_path = scorer_root / str(scoring_manifest["truth_sidecar_json"])
    if sha256_file(truth_path) != scoring_manifest["truth_sidecar_sha256"]:
        raise ValueError("truth sidecar SHA mismatch")
    truth = _read_json(truth_path)
    if not isinstance(truth.get("rows"), list):
        raise ValueError("truth sidecar rows are missing")
    metrics = summarize_predictions(
        arrays, truth["rows"], old_class_count=int(manifest["old_class_count"])
    )
    result = {
        "schema": SCORING_SCHEMA,
        "status": "SCORED_AFTER_FROZEN_PREDICTION",
        "prediction_manifest_sha256": sha256_file(
            prediction_root / "prediction_manifest.json"
        ),
        "prediction_npz_sha256": manifest["prediction_npz_sha256"],
        "truth_sidecar_sha256": scoring_manifest["truth_sidecar_sha256"],
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "k_shot": manifest["k_shot"],
        "new_class_count": manifest["new_class_count"],
        "metrics": metrics,
        "scorer_output_used_by_predictor": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / "score.json", result)
    with (output / "summary.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_markdown_score(result))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--predictor-root", type=Path, required=True)
    predict.add_argument("--detached-seal", type=Path, required=True)
    predict.add_argument("--expected-seal-sha256", required=True)
    predict.add_argument("--k-shot", type=int, default=10)
    predict.add_argument("--negative-quantile", type=float, default=0.95)
    predict.add_argument("--prior-physical-shots", type=float, default=8.0)
    predict.add_argument("--scale-floor", type=float, default=0.05)
    predict.add_argument("--inverse-scale-cap", type=float, default=10.0)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--batch-size", type=int, default=256)
    predict.add_argument("--output-root", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--scorer-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_predict(args) if args.command == "predict" else run_score(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
