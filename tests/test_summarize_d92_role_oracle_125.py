from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import summarize_d92_role_oracle_125 as summary


@pytest.fixture(autouse=True)
def _small_query_count(monkeypatch):
    monkeypatch.setattr(summary, "FORMAL_QUERY_PER_TX", 1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    records: list[dict] = []
    scenarios = list(summary.EXPECTED_SCENARIOS)
    slices = list(summary.EXPECTED_SLICES)
    for receiver in summary.EXPECTED_RECEIVERS:
        for seed in summary.EXPECTED_SEEDS:
            for k_shot, new_count in slices:
                row_id = f"{receiver}-{seed}-{k_shot}-{new_count}"
                old = [f"o{i}" for i in range(6)]
                new = [f"n{i}" for i in range(new_count)]
                for scenario in scenarios:
                    for state, classes in (("before", old), ("after", old + new)):
                        for true_class in classes:
                            for query_index in range(summary.FORMAL_QUERY_PER_TX):
                                role = "target_old" if true_class.startswith("o") else "target_new"
                                token = f"{state}-{scenario}-{true_class}-{query_index}"
                                common = {
                                    "row_id": row_id, "receiver": receiver, "seed": seed,
                                    "k_shot": k_shot, "new_class_count": new_count,
                                    "scenario": scenario, "state": state, "query_token": token,
                                    "true_class": true_class, "true_role": role,
                                    "score_contract_sha256": "1" * 64,
                                    "model_state_sha256": "2" * 64,
                                    "score_vector_sha256": hashlib.sha256(token.encode()).hexdigest(),
                                    "query_payload_sha256": "3" * 64,
                                    "registered_classes_sha256": summary._registry_sha256(old, [] if state == "before" else new),
                                    "old_registered_classes": old,
                                    "new_registered_classes": [] if state == "before" else new,
                                }
                                baseline_pred = true_class
                                if state == "after" and true_class == "o0":
                                    baseline_pred = "n0"
                                for variant in summary.VARIANTS:
                                    predicted = true_class if variant == "role_oracle" else baseline_pred
                                    records.append({**common, "variant": variant, "predicted_class": predicted})
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in records), encoding="utf-8")
    manifest = {
        "schema": summary.SCHEMA,
        "claim_scope": summary.CLAIM_SCOPE,
        "fresh_run": True,
        "fresh_pairing_id": "fresh-pair-1",
        "run_id": "d92-role-oracle-125-test",
        "candidate": "d92_role_oracle_licensed_upper_bound",
        "formal_protocol_valid": False,
        "promotion_eligible": False,
        "job_count": 125,
        "scenario_pair_count": 375,
        "variants": list(summary.VARIANTS),
        "fresh_no_oracle_bit_exact_to_d92_retry2": True,
        "old_class_count": 6,
        "receivers": list(summary.EXPECTED_RECEIVERS),
        "seeds": list(summary.EXPECTED_SEEDS),
        "slices": [{"k_shot": k, "new_class_count": n} for k, n in slices],
        "scenarios": scenarios,
        "records_format": "jsonl",
        "records_path": records_path.name,
        "records_sha256": _sha(records_path),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _mutate_records(manifest_path: Path, mutate) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = manifest_path.parent / manifest["records_path"]
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    records_path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    manifest["records_sha256"] = _sha(records_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_full_fresh_pair_summary_writes_all_required_tables(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    output = tmp_path / "out"
    result = summary.summarize(manifest, output)
    assert result["status"] == summary.CLAIM_SCOPE
    assert result["promotable"] is False
    assert result["protocol_legal_performance"] is False
    assert result["job_count"] == 125
    assert result["scenario_pair_count"] == 375
    assert result["before_prediction_identity"] == "PASS"
    expected = {
        "row_metrics.csv": 125,
        "scene_metrics.csv": 375,
        "receiver_slice_metrics.csv": 25,
        "scene_slice_metrics.csv": 15,
        "receiver_scene_slice_metrics.csv": 75,
        "seed_slice_metrics.csv": 25,
        "slice_metrics.csv": 5,
    }
    assert result["tables"] == expected
    with (output / "row_metrics.csv").open(encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    assert float(first["delta_a_old_acc"]) > 0.0
    assert float(first["delta_forgetting"]) < 0.0
    with pytest.raises(FileExistsError):
        summary.summarize(manifest, output)


def test_rejects_score_vector_drift_within_pair(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    _mutate_records(
        manifest,
        lambda rows: rows.__setitem__(1, {**rows[1], "score_vector_sha256": "f" * 64}),
    )
    with pytest.raises(summary.OracleSummaryError, match="score_vector_sha256"):
        summary.summarize(manifest, tmp_path / "out")


def test_rejects_different_five_by_five_by_five_axes(tmp_path: Path) -> None:
    manifest_path = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["receivers"] = ["fake-0", "fake-1", "fake-2", "fake-3", "fake-4"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(summary.OracleSummaryError, match="frozen D92 125 matrix"):
        summary.summarize(manifest_path, tmp_path / "out")


def test_rejects_cross_role_oracle_prediction(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def mutate(rows: list[dict]) -> None:
        row = next(r for r in rows if r["variant"] == "role_oracle" and r["state"] == "after" and r["true_role"] == "target_old")
        row["predicted_class"] = "n0"

    _mutate_records(manifest, mutate)
    with pytest.raises(summary.OracleSummaryError, match="cross-role"):
        summary.summarize(manifest, tmp_path / "out")


def test_rejects_oracle_correctness_regression(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def mutate(rows: list[dict]) -> None:
        row = next(r for r in rows if r["variant"] == "role_oracle" and r["state"] == "after" and r["true_class"] == "o1")
        row["predicted_class"] = "o2"

    _mutate_records(manifest, mutate)
    with pytest.raises(summary.OracleSummaryError, match="correctness monotonicity"):
        summary.summarize(manifest, tmp_path / "out")


def test_rejects_before_prediction_difference(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def mutate(rows: list[dict]) -> None:
        row = next(r for r in rows if r["variant"] == "role_oracle" and r["state"] == "before")
        row["predicted_class"] = "o2" if row["true_class"] != "o2" else "o3"

    _mutate_records(manifest, mutate)
    with pytest.raises(summary.OracleSummaryError, match="before predictions"):
        summary.summarize(manifest, tmp_path / "out")


def test_rejects_change_when_baseline_is_already_within_true_role(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def mutate(rows: list[dict]) -> None:
        selected = [
            row
            for row in rows
            if row["state"] == "after"
            and row["true_class"] == "o1"
            and row["query_token"].endswith("-0")
        ]
        baseline = next(row for row in selected if row["variant"] == "baseline")
        oracle = next(row for row in selected if row["variant"] == "role_oracle")
        baseline["predicted_class"] = "o2"
        oracle["predicted_class"] = "o3"

    _mutate_records(manifest, mutate)
    with pytest.raises(summary.OracleSummaryError, match="already within-role"):
        summary.summarize(manifest, tmp_path / "out")


def test_rejects_baseline_prediction_outside_registry(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def mutate(rows: list[dict]) -> None:
        row = next(r for r in rows if r["variant"] == "baseline" and r["state"] == "after")
        row["predicted_class"] = "not-registered"

    _mutate_records(manifest, mutate)
    with pytest.raises(summary.OracleSummaryError, match="outside all registered"):
        summary.summarize(manifest, tmp_path / "out")


def test_csv_input_contract(tmp_path: Path) -> None:
    manifest_path = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jsonl = tmp_path / manifest["records_path"]
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    csv_path = tmp_path / "records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "old_registered_classes": json.dumps(row["old_registered_classes"]), "new_registered_classes": json.dumps(row["new_registered_classes"])})
    manifest["records_format"] = "csv"
    manifest["records_path"] = csv_path.name
    manifest["records_sha256"] = _sha(csv_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = summary.summarize(manifest_path, tmp_path / "csv-out")
    assert result["query_pair_count"] > 0


def test_rejects_missing_retry2_bit_exact_baseline_gate(tmp_path: Path) -> None:
    manifest_path = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fresh_no_oracle_bit_exact_to_d92_retry2"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        summary.OracleSummaryError,
        match="fresh_no_oracle_bit_exact_to_d92_retry2",
    ):
        summary.summarize(manifest_path, tmp_path / "out")
