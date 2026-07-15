"""Score immutable ADV3B02 Stage2-B predictions in a separate process."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS, sha256_file  # noqa: E402
from cvsrffi.phase2_runtime_contract import validate_phase2_contract  # noqa: E402
from cvsrffi.stage2_scoring_sidecar import load_verified_scoring_sidecar  # noqa: E402
from paper_reproduction.cvs_aligned.class_incremental import _detailed_breakdown  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int64_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return torch.frombuffer(memoryview(array), dtype=torch.int64).reshape(array.shape).clone()


def score(run_dir: Path, scoring_manifest_path: Path, runtime_evidence_path: Path) -> dict[str, Any]:
    prediction_manifest_path = run_dir / "prediction_manifest.json"
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8-sig"))
    if prediction_manifest.get("schema") != "adv3b02_stage2b_prediction_artifact_v1":
        raise ValueError("prediction manifest schema drift")
    prediction_path = run_dir / str(prediction_manifest["prediction_npz"])
    if sha256_file(prediction_path) != str(prediction_manifest["prediction_npz_sha256"]):
        raise ValueError("immutable prediction artifact hash mismatch")
    split_manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8-sig"))
    runtime_evidence = json.loads(runtime_evidence_path.read_text(encoding="utf-8-sig"))
    contract_record = dict(split_manifest)
    contract_record["phase2_runtime_isolation_evidence"] = runtime_evidence
    validate_phase2_contract(contract_record, evidence_phase="post_run")
    if runtime_evidence["prediction_artifact_sha256"] != sha256_file(prediction_path):
        raise ValueError("runtime evidence prediction artifact hash mismatch")
    if runtime_evidence["prediction_seal_sha256"] != sha256_file(prediction_manifest_path):
        raise ValueError("runtime evidence prediction seal hash mismatch")
    access_audit_path = run_dir / "filesystem_access_audit.json"
    if runtime_evidence["filesystem_access_audit_sha256"] != sha256_file(access_audit_path):
        raise ValueError("runtime evidence filesystem access audit hash mismatch")
    access_audit = json.loads(access_audit_path.read_text(encoding="utf-8-sig"))
    if access_audit.get("status") != "PASS" or access_audit.get("forbidden_access_hits") != []:
        raise ValueError("filesystem access audit is not clean")
    truth, scoring_manifest, sidecar_audit = load_verified_scoring_sidecar(scoring_manifest_path)
    if str(scoring_manifest.get("predictor_package_seal_sha256", "")) != str(
        split_manifest.get("target_predictor_package_seal_sha256", "")
    ):
        raise ValueError("scorer/predictor package seal hash mismatch")
    if str(scoring_manifest.get("predictor_package_root_sha256", "")) != str(
        split_manifest.get("target_predictor_bundle_audit", {})
        .get("seal", {}).get("package_root_sha256", "")
    ):
        raise ValueError("scorer/predictor package root hash mismatch")
    if scoring_manifest.get("scorer_output_must_not_feed_predictor") is not True:
        raise ValueError("scorer feedback guard missing")

    with np.load(prediction_path, allow_pickle=False) as archive:
        expected = {
            "sample_ids", "scenarios", "before_predicted_labels", "predicted_labels"
        }
        if set(archive.files) != expected:
            raise ValueError("prediction member allowlist drift")
        sample_ids = np.asarray(archive["sample_ids"]).astype(str)
        scenarios = np.asarray(archive["scenarios"]).astype(str)
        before = np.asarray(archive["before_predicted_labels"], dtype=np.int64)
        predicted = np.asarray(archive["predicted_labels"], dtype=np.int64)
    if not (len(sample_ids) == len(scenarios) == len(before) == len(predicted)):
        raise ValueError("prediction arrays have inconsistent lengths")
    truth_by_id = {str(row["query_token"]): dict(row) for row in truth["rows"]}
    if len(truth_by_id) != len(truth["rows"]):
        raise ValueError("truth sidecar contains duplicate opaque query IDs")

    score_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    scenario_metrics: dict[str, dict[str, Any]] = {}
    runtime = dict(prediction_manifest.get("scenario_runtime", {}))
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        indices = np.flatnonzero(scenarios == scenario)
        ids = sample_ids[indices].tolist()
        if set(ids) != set(truth_by_id):
            raise ValueError(f"prediction/truth opaque ID join mismatch for {scenario}")
        truth_values = np.asarray(
            [int(truth_by_id[value]["true_class_index"]) for value in ids]
        )
        before_values = before[indices]
        after_values = predicted[indices]
        before_acc = float(np.mean(before_values == truth_values))
        after_acc = float(np.mean(after_values == truth_values))
        scenario_metrics[scenario] = {
            "target_old_accuracy": after_acc,
            "target_old_accuracy_before_adaptation": before_acc,
            "target_old_accuracy_delta": after_acc - before_acc,
            **runtime[scenario],
        }
        metadata = [
            {
                **truth_by_id[value],
                "rx_label": truth_by_id[value]["receiver_label"],
                "tx_label": truth_by_id[value]["transmitter_label"],
                "day_i": truth_by_id[value]["day_label"],
                "sig_i": truth_by_id[value]["signal_label"],
                "role": truth_by_id[value]["evaluation_role"],
            }
            for value in ids
        ]
        detailed.extend(
            _detailed_breakdown(
                _int64_tensor(after_values),
                _int64_tensor(truth_values),
                metadata,
                scenario=scenario,
            )
        )
        for opaque_id, truth_value, before_value, predicted_value in zip(
            ids, truth_values.tolist(), before_values.tolist(), after_values.tolist()
        ):
            meta = truth_by_id[opaque_id]
            score_rows.append({
                "sample_id": opaque_id,
                "receiver_label": meta["receiver_label"],
                "transmitter_label": meta["transmitter_label"],
                "day_i": meta["day_label"],
                "sig_i": meta["signal_label"],
                "role": meta["evaluation_role"],
                "true_label": truth_value,
                "before_predicted_label": before_value,
                "predicted_label": predicted_value,
                "correct": int(truth_value == predicted_value),
                "scenario": scenario,
            })
    aggregate = {
        key + "_mean": float(sum(float(row[key]) for row in scenario_metrics.values()) / 3.0)
        for key in (
            "target_old_accuracy", "target_old_accuracy_before_adaptation",
            "target_old_accuracy_delta", "adaptation_latency_sec",
        )
    }
    if not all(math.isfinite(value) for value in aggregate.values()):
        raise FloatingPointError("non-finite aggregate metric")
    result = {
        "schema": "adv3b02_stage2b_supervised_da_v2",
        "experiment_id": prediction_manifest["experiment_id"],
        "method_id": prediction_manifest["method_id"],
        "seed": int(prediction_manifest["seed"]),
        "target_receiver_label": prediction_manifest["target_receiver_label"],
        "k_shot": int(prediction_manifest["k_shot"]),
        "metrics": aggregate,
        "metrics_by_scenario": scenario_metrics,
    }
    scoring_audit = {
        "schema": "adv3b02_stage2b_isolated_scoring_audit_v1",
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "prediction_npz_sha256": sha256_file(prediction_path),
        "runtime_evidence_sha256": sha256_file(runtime_evidence_path),
        "truth_join_after_prediction_only": True,
        "predictor_process_exited_before_truth_open": True,
        "scorer_output_must_not_feed_predictor": True,
        "score_row_count": len(score_rows),
        **sidecar_audit,
    }
    _write_json(run_dir / "metrics.json", result)
    _write_json(run_dir / "detailed_metrics.json", detailed)
    _write_json(run_dir / "scoring_audit.json", scoring_audit)
    _write_csv(run_dir / "score_table.csv", score_rows)
    _write_csv(run_dir / "detailed_metrics.csv", detailed)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.run_dir, args.scoring_manifest, args.runtime_evidence)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
