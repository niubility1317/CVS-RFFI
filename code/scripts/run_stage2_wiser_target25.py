#!/usr/bin/env python3
"""Prepare, predict, truth-last score, and analyze WISER Target25/K10 shards."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_bisage_runner import frozen_checkpoint  # noqa: E402
from cvsrffi.stage2_wiser_pilot import (  # noqa: E402
    SCENARIOS, load_query_package, load_support_package,
)
from cvsrffi.stage2_wiser_runner import (  # noqa: E402
    WISERP3TrainingConfig, predict_wiser_representation_probes, train_wiser_p3_arm,
)
from cvsrffi.stage2_wiser_scoring import (  # noqa: E402
    compare_wiser_score_rows, score_wiser_predictions,
)
from cvsrffi.stage2_wiser_target25 import (  # noqa: E402
    WISERTarget25Error, build_wiser_target25_manifest, target25_promotion_decision,
    validate_wiser_target25_manifest,
)
from cvsrffi.wiser_source_summary import load_quantized_source_summary  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _new_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output root exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _tensor(values: np.ndarray, device: str, *, labels: bool = False) -> torch.Tensor:
    dtype = torch.long if labels else torch.float32
    return torch.from_numpy(np.ascontiguousarray(values)).to(device=torch.device(device), dtype=dtype)


def _packages(job: Mapping[str, Any], role: str) -> Path:
    packages = job.get("packages")
    if not isinstance(packages, Mapping) or not isinstance(packages.get(role), Mapping):
        raise ValueError("Target25 package registry drift")
    root = packages[role].get("package_root")
    if not isinstance(root, str) or not root:
        raise ValueError("Target25 package root missing")
    return Path(root)


def _jobs_for_shard(manifest: Mapping[str, Any], shard_index: int) -> list[Mapping[str, Any]]:
    validate_wiser_target25_manifest(manifest)
    if shard_index not in range(8):
        raise ValueError("Target25 shard index must be 0 through 7")
    jobs = [job for job in manifest["jobs"] if job["planned_shard_index"] == shard_index]
    if not jobs:
        raise ValueError("Target25 shard is empty")
    return jobs


def _prediction_root(root: Path, job: Mapping[str, Any], scenario: str, arm: str) -> Path:
    return root / str(job["outer_key"]) / scenario / arm / "prediction"


def _validate_source_binding(path: Path, job: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = _load_json(path)
    if binding.get("schema") != "cvs.phase1.wiser_rf.source_binding.v1":
        raise ValueError("Target25 source-binding schema drift")
    if binding.get("checkpoint_id") != job.get("champion_checkpoint_id"):
        raise ValueError("Target25 champion checkpoint binding drift")
    registry = binding.get("class_registry")
    if binding.get("feature_schema") != "ADV3B02:z_id:unit_l2:160:v1" or binding.get("feature_dim") != 160 or not isinstance(registry, list) or len(registry) != 6 or len(set(map(str, registry))) != 6:
        raise ValueError("Target25 source-binding semantic drift")
    return binding


def _prepare(args: argparse.Namespace) -> Mapping[str, Any]:
    source = _load_json(args.source_manifest)
    marker = _load_json(args.pilot_marker)
    # Validate all immutable inputs before claiming an output root.
    manifest = build_wiser_target25_manifest(source, marker, str(args.output_root), phase=args.phase)
    destination = _new_root(args.output_root)
    _write_json_new(destination / "manifest.json", manifest)
    for shard in range(8):
        jobs = [job for job in manifest["jobs"] if job["planned_shard_index"] == shard]
        _write_json_new(destination / "shards" / f"shard_{shard}.json", {
            "schema": "cvs.phase2.wiser_rf.target25.shard_manifest.v1", "status": "PREPARED",
            "validation_phase": manifest["validation_phase"], "shard_index": shard,
            "query_opened": False, "truth_opened": False, "jobs": jobs,
        })
    return {"schema": "cvs.phase2.wiser_rf.target25.prepare.v1", "status": "PREPARED",
            "output_root": str(destination), "validation_phase": manifest["validation_phase"],
            "outer_count": len(manifest["jobs"]), "scene_unit_count": len(manifest["jobs"]) * 3,
            "shard_count": 8, "query_opened": False, "truth_opened": False}


def _train_candidate(job: Mapping[str, Any], support: Any, *, checkpoint: Path, source_summary: Any, binding: Mapping[str, Any], device: str) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    model = frozen_checkpoint(checkpoint, device)
    audit = train_wiser_p3_arm(
        model, _tensor(support.iq, device), _tensor(support.labels, device, labels=True),
        support_tokens=support.tokens, source_summary=source_summary,
        expected_source_class_registry=tuple(binding["class_registry"]),
        expected_source_feature_schema=str(binding["feature_schema"]), arm=str(job["champion_arm"]),
        config=WISERP3TrainingConfig(seed=int(job["seed"])),
    )
    result = {**asdict(audit), "outer_arm": str(job["champion_arm"]), "trainer_arm": str(job["champion_arm"]), "support_state_frozen": True}
    if result["query_rows_used"] != 0:
        raise ValueError("Target25 support adaptation used query rows")
    return model, result


def _run_shard(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(args.manifest)
    jobs = _jobs_for_shard(manifest, int(args.shard_index))
    binding = _validate_source_binding(args.source_binding, jobs[0])
    if any(job["champion_checkpoint_id"] != jobs[0]["champion_checkpoint_id"] for job in jobs):
        raise ValueError("Target25 shard champion checkpoint drift")
    source_summary = load_quantized_source_summary(args.source_summary)
    if tuple(source_summary.class_registry) != tuple(binding["class_registry"]) or source_summary.feature_schema != binding["feature_schema"] or tuple(source_summary.centers.shape) != (6, 160):
        raise ValueError("Target25 source summary binding drift")
    destination = _new_root(args.output_root)
    completed: list[dict[str, Any]] = []
    # Each unit fresh-loads N0 and the champion; all support state freezes before its query.
    for job in jobs:
        for scenario in SCENARIOS:
            support = load_support_package(_packages(job, "before_enrollment") / f"support_{scenario}.npz")
            support_iq, support_labels = _tensor(support.iq, args.device), _tensor(support.labels, args.device, labels=True)
            baseline = frozen_checkpoint(args.checkpoint, args.device)
            candidate, audit = _train_candidate(job, support, checkpoint=args.checkpoint, source_summary=source_summary, binding=binding, device=args.device)
            query = load_query_package(_packages(job, "before_apply") / f"query_{scenario}.npz")
            query_iq = _tensor(query.iq, args.device)
            for arm, model, training_audit in (("N0", baseline, {"outer_arm": "N0", "trainer_arm": None, "query_rows_used": 0, "support_state_frozen": True}), (str(job["champion_arm"]), candidate, audit)):
                prediction = _prediction_root(destination, job, scenario, arm)
                prediction.mkdir(parents=True, exist_ok=False)
                arrays = predict_wiser_representation_probes(model, support_iq, support_labels, query_iq, query_tokens=query.tokens, source_summary=source_summary, seed=int(job["seed"]))
                np.savez_compressed(prediction / "predictions.npz", **arrays)
                _write_json_new(prediction / "prediction_receipt.json", {
                    "schema": "cvs.phase2.wiser_rf.target25.prediction_receipt.v1", "status": "PREDICTIONS_COMPLETE",
                    **{field: job[field] for field in ("outer_key", "capsule_id", "split_id", "receiver", "seed", "k_shot", "new_class_count", "planned_shard_index")},
                    "scenario": scenario, "arm": arm, "champion_arm": job["champion_arm"], "champion_commit": job["champion_commit"],
                    "query_rows": len(query.tokens), "expected_query_tokens": list(query.tokens), "query_rows_used": 0,
                    "query_truth_opened": False, "query_role_opened": False, "support_state_frozen_before_query": True,
                    "training_audit": training_audit,
                })
                completed.append({"outer_key": job["outer_key"], "scenario": scenario, "arm": arm, "query_rows": len(query.tokens)})
    _write_json_new(destination / "run_receipt.json", {"schema": "cvs.phase2.wiser_rf.target25.run_shard.v1", "status": "ARTIFACTS_COMPLETE", "shard_index": int(args.shard_index), "units": completed, "truth_opened": False})
    return {"status": "ARTIFACTS_COMPLETE", "shard_index": int(args.shard_index), "prediction_root": str(destination), "prediction_unit_count": len(completed), "truth_opened": False}


def _validate_prediction_npz(path: Path, receipt: Mapping[str, Any]) -> None:
    required = {"query_tokens", "query_z_id", "p1_predictions", "p1_logits", "p2_predictions", "p2_logits", "p3_predictions", "p3_logits"}
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != required:
            raise ValueError("Target25 prediction NPZ member registry drift")
        tokens = np.asarray(arrays["query_tokens"]).astype(str)
        expected = tuple(str(value) for value in receipt.get("expected_query_tokens", ()))
        count = receipt.get("query_rows")
        if not isinstance(count, int) or count <= 0 or tokens.ndim != 1 or len(tokens) != count or tuple(tokens.tolist()) != expected or len(set(expected)) != len(expected):
            raise ValueError("Target25 prediction token closure drift")
        identity = np.asarray(arrays["query_z_id"], dtype=np.float64)
        if identity.shape != (count, 160) or not np.isfinite(identity).all():
            raise ValueError("Target25 prediction identity closure drift")
        for prefix in ("p1", "p2", "p3"):
            prediction = np.asarray(arrays[f"{prefix}_predictions"])
            logits = np.asarray(arrays[f"{prefix}_logits"], dtype=np.float64)
            if prediction.ndim != 1 or prediction.shape != (count,) or not np.issubdtype(prediction.dtype, np.integer) or logits.shape != (count, 6) or not np.isfinite(logits).all():
                raise ValueError("Target25 prediction probe geometry drift")
            indices = prediction.astype(np.int64, copy=False)
            if bool(((indices < 0) | (indices >= 6)).any()) or not np.array_equal(indices, logits.argmax(axis=1).astype(np.int64)):
                raise ValueError("Target25 prediction probe/argmax drift")


def _validate_prediction_registry(manifest: Mapping[str, Any], prediction_root: Path, shard_index: int) -> list[tuple[Mapping[str, Any], str, str, Path, Mapping[str, Any]]]:
    """Prevalidate every N0/champion prediction before a scorer may open truth."""

    rows: list[tuple[Mapping[str, Any], str, str, Path, Mapping[str, Any]]] = []
    for job in _jobs_for_shard(manifest, shard_index):
        for scenario in SCENARIOS:
            for arm in ("N0", str(job["champion_arm"])):
                root = _prediction_root(prediction_root, job, scenario, arm)
                receipt = _load_json(root / "prediction_receipt.json")
                if receipt.get("schema") != "cvs.phase2.wiser_rf.target25.prediction_receipt.v1" or receipt.get("status") != "PREDICTIONS_COMPLETE":
                    raise ValueError("Target25 prediction receipt schema/status drift")
                for field in ("outer_key", "capsule_id", "split_id", "receiver", "seed", "k_shot", "new_class_count", "planned_shard_index"):
                    if receipt.get(field) != job[field]: raise ValueError("Target25 prediction receipt binding drift")
                if receipt.get("scenario") != scenario or receipt.get("arm") != arm or receipt.get("champion_arm") != job["champion_arm"] or receipt.get("champion_commit") != job["champion_commit"] or receipt.get("query_rows_used") != 0 or receipt.get("query_truth_opened") is not False or receipt.get("query_role_opened") is not False or receipt.get("support_state_frozen_before_query") is not True:
                    raise ValueError("Target25 prediction receipt truth-last drift")
                audit = receipt.get("training_audit")
                if not isinstance(audit, Mapping) or audit.get("query_rows_used") != 0 or audit.get("support_state_frozen") is not True:
                    raise ValueError("Target25 training audit drift")
                _validate_prediction_npz(root / "predictions.npz", receipt)
                rows.append((job, scenario, arm, root, receipt))
    return rows


def _score_shard(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(args.manifest)
    # This must complete before the first truth_sidecar path is read.
    validated = _validate_prediction_registry(manifest, args.prediction_root, int(args.shard_index))
    destination = _new_root(args.output_root)
    scores: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for job, scenario, arm, root, _receipt in validated:
        score = score_wiser_predictions(root / "predictions.npz", root / "prediction_receipt.json", Path(str(job["truth_sidecar"])))
        _write_json_new(destination / str(job["outer_key"]) / scenario / arm / "score.json", score)
        scores[(str(job["outer_key"]), scenario, arm)] = score
    paired: list[dict[str, Any]] = []
    for job in _jobs_for_shard(manifest, int(args.shard_index)):
        for scenario in SCENARIOS:
            candidate_arm = str(job["champion_arm"])
            comparison = dict(compare_wiser_score_rows(scores[(str(job["outer_key"]), scenario, "N0")], scores[(str(job["outer_key"]), scenario, candidate_arm)]))
            candidate_receipt = _load_json(_prediction_root(args.prediction_root, job, scenario, candidate_arm) / "prediction_receipt.json")
            comparison.update({field: job[field] for field in ("seed", "k_shot", "new_class_count", "planned_shard_index")})
            comparison["expected_query_tokens"] = candidate_receipt["expected_query_tokens"]
            comparison["query_rows_used"] = 0
            comparison["candidate_training_audit"] = candidate_receipt["training_audit"]
            paired.append(comparison)
    result = {"schema": "cvs.phase2.wiser_rf.target25.score_shard.v1", "status": "ANALYZED", "shard_index": int(args.shard_index), "paired_rows": paired, "truth_join_after_prediction_only": True}
    _write_json_new(destination / "score_collection.json", result)
    return {"status": "ANALYZED", "shard_index": int(args.shard_index), "score_root": str(destination), "paired_scene_unit_count": len(paired)}


def _analyze(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(args.manifest)
    validate_wiser_target25_manifest(manifest)
    paired: list[Mapping[str, Any]] = []
    for shard in range(8):
        payload = _load_json(args.score_root / f"shard_{shard}" / "score_collection.json")
        if payload.get("schema") != "cvs.phase2.wiser_rf.target25.score_shard.v1" or payload.get("status") != "ANALYZED" or payload.get("shard_index") != shard or not isinstance(payload.get("paired_rows"), list):
            raise ValueError("Target25 score-shard collection drift")
        paired.extend(payload["paired_rows"])
    # Analyzer sees only scored JSON rows, never raw query packages or truth sidecars.
    decision = target25_promotion_decision(paired, phase=str(manifest["validation_phase"]))
    destination = _new_root(args.output_root)
    _write_json_new(destination / "analysis.json", {**decision, "manifest_schema": manifest["schema"], "query_opened": False, "truth_opened": False})
    return decision


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--pilot-marker", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--phase", choices=("target25", "k10"), default="target25")
    run = commands.add_parser("run-shard")
    run.add_argument("--manifest", type=Path, required=True); run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--checkpoint", type=Path, required=True); run.add_argument("--source-summary", type=Path, required=True); run.add_argument("--source-binding", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True); run.add_argument("--device", required=True)
    score = commands.add_parser("score-shard")
    score.add_argument("--manifest", type=Path, required=True); score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--prediction-root", type=Path, required=True); score.add_argument("--output-root", type=Path, required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--manifest", type=Path, required=True); analyze.add_argument("--score-root", type=Path, required=True); analyze.add_argument("--output-root", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    result = {"prepare": _prepare, "run-shard": _run_shard, "score-shard": _score_shard, "analyze": _analyze}[args.command](args)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
