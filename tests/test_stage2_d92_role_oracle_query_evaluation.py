from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d81_query_evaluation as d81_eval
from cvsrffi import stage2_d92_role_oracle_query_evaluation as paired
from cvsrffi.stage2_d92_role_oracle_records import (
    build_d92_role_oracle_row_records,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_d92_licensed_role_oracle import ROLE_CAPSULE_SCHEMA


OLD = ("old0", "old1")
ALL = OLD + ("new0", "new1")


def _token(role: str, index: int) -> str:
    return f"qid_{(index + (0 if role == 'old' else 100)):032x}"


def _role_capsule(path: Path) -> Path:
    rows = []
    for index, _scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows.extend(
            [
                {
                    "query_token": _token("old", index),
                    "evaluation_role": "target_old",
                },
                {
                    "query_token": _token("new", index),
                    "evaluation_role": "target_new",
                },
            ]
        )
    payload = {
        "schema": ROLE_CAPSULE_SCHEMA,
        "rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, stat.S_IREAD)
    return path


def _install_fake_baseline(monkeypatch, tmp_path: Path, *, fail: bool = False):
    score_calls: list[tuple[str, ...]] = []

    def fake_score(state, features):
        score_calls.append(tuple(state.classes))
        return np.asarray(features, dtype=np.float32)

    def fake_baseline(*, output_root, **kwargs):
        root = Path(output_root)
        before_rows = []
        after_rows = []
        for scenario_index, _scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
            before_rows.append(
                np.asarray(
                    d81_eval.predict_d42_unified_shrinkage_lda(
                        SimpleNamespace(classes=OLD),
                        np.asarray([[0.8, 0.2]], dtype=np.float32),
                    )
                )
            )
            if fail:
                raise RuntimeError("synthetic baseline failure")
            after_rows.append(
                np.asarray(
                    d81_eval.predict_d42_unified_shrinkage_lda(
                        SimpleNamespace(classes=ALL),
                        np.asarray(
                            [[0.7, 0.2, 0.9, 0.1], [0.8, 0.1, 0.2, 0.6]],
                            dtype=np.float32,
                        ),
                    )
                )
            )
        states = {}
        for state, tokens, classes, predictions in (
            (
                "before",
                np.asarray(
                    [_token("old", index) for index in range(len(FORMAL_LEO_WEAK_SCENARIOS))]
                ),
                OLD,
                np.concatenate(before_rows),
            ),
            (
                "after",
                np.asarray(
                    [
                        token
                        for index in range(len(FORMAL_LEO_WEAK_SCENARIOS))
                        for token in (_token("old", index), _token("new", index))
                    ]
                ),
                ALL,
                np.concatenate(after_rows),
            ),
        ):
            destination = root / state
            destination.mkdir()
            scenarios = np.concatenate(
                [
                    np.asarray([scenario] * (1 if state == "before" else 2))
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                ]
            )
            prediction_sha = paired._write_npz_new(
                destination / "prediction_artifact.npz",
                query_tokens=tokens,
                scenarios=scenarios,
                predicted_class_handles=np.asarray(predictions).astype(str),
            )
            receipt_sha = paired._write_json_new(
                destination / "execution_receipt.json",
                {
                    "schema": "cvs.phase2.diag_cosine_exploration_receipt.v1",
                    "apply_package_root_sha256": (
                        "b" if state == "before" else "c"
                    )
                    * 64,
                },
            )
            members = [
                {
                    "relative_path": name,
                    "sha256": paired._sha256_file(destination / name),
                    "size_bytes": (destination / name).stat().st_size,
                }
                for name in ("prediction_artifact.npz", "execution_receipt.json")
            ]
            commit_sha = paired._write_json_new(
                destination / "COMMIT.json",
                {
                    "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
                    "execution_receipt_sha256": receipt_sha,
                    "prediction_artifact_sha256": prediction_sha,
                    "members": members,
                },
            )
            states[state] = {
                "prediction_artifact_sha256": prediction_sha,
                "execution_receipt_sha256": receipt_sha,
                "commit_sha256": commit_sha,
            }
        return {
            "receiver": "rx",
            "seed": 7,
            "k_shot": 5,
            "new_class_count": 2,
            "states": states,
            "resource": {"persistent_state_bytes_peak": 123},
        }

    monkeypatch.setattr(paired, "score_d42_unified_shrinkage_lda", fake_score)
    state_counter = iter(range(100))
    monkeypatch.setattr(
        paired,
        "_model_state_sha256",
        lambda _state: f"{next(state_counter):064x}",
    )
    monkeypatch.setattr(paired, "run_d92_query_evaluation", fake_baseline)
    return score_calls


def test_paired_evaluation_commits_baseline_before_opening_role_capsule(
    monkeypatch, tmp_path: Path
):
    calls = _install_fake_baseline(monkeypatch, tmp_path)
    capsule = _role_capsule(tmp_path / "role.json")
    original_reader = paired._read_role_capsule
    opened = []

    def guarded_reader(path, **kwargs):
        root = tmp_path / "out" / "baseline"
        assert (root / "before" / "COMMIT.json").is_file()
        assert (root / "after" / "COMMIT.json").is_file()
        opened.append(True)
        return original_reader(path, **kwargs)

    monkeypatch.setattr(paired, "_read_role_capsule", guarded_reader)
    supplied = []

    def factory():
        root = tmp_path / "out" / "baseline"
        assert (root / "before" / "COMMIT.json").is_file()
        assert (root / "after" / "COMMIT.json").is_file()
        supplied.append(True)
        return capsule, {
            "path": str(capsule),
            "source_truth_sha256": "a" * 64,
            "capsule_sha256": hashlib.sha256(capsule.read_bytes()).hexdigest(),
            "readonly": True,
        }

    result = paired.run_d92_role_oracle_query_evaluation(
        role_capsule_factory=factory, output_root=tmp_path / "out"
    )
    assert opened == [True]
    assert supplied == [True]
    assert result["role_capsule_projection"]["readonly"] is True
    assert len(calls) == 2 * len(FORMAL_LEO_WEAK_SCENARIOS)
    assert result["status"] == "LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE"
    assert result["promotion_eligible"] is False
    with np.load(
        tmp_path / "out" / "baseline" / "after" / "prediction_artifact.npz",
        allow_pickle=False,
    ) as archive:
        baseline = archive["predicted_class_handles"].astype(str)
    with np.load(
        tmp_path / "out" / "oracle" / "after" / "prediction_artifact.npz",
        allow_pickle=False,
    ) as archive:
        oracle = archive["predicted_class_handles"].astype(str)
    assert baseline.tolist() == ["new0", "old0"] * 3
    assert oracle.tolist() == ["old0", "new1"] * 3
    with np.load(
        tmp_path / "out" / "oracle" / "before" / "prediction_artifact.npz",
        allow_pickle=False,
    ) as archive:
        before_oracle = archive["predicted_class_handles"].astype(str)
    assert before_oracle.tolist() == ["old0"] * 3

    truth_rows = []
    for index, _scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        truth_rows.extend(
            [
                {
                    "query_token": _token("old", index),
                    "true_class_handle": "old1",
                    "evaluation_role": "target_old",
                },
                {
                    "query_token": _token("new", index),
                    "true_class_handle": "new1",
                    "evaluation_role": "target_new",
                },
            ]
        )
    truth = {
        "schema": "cvs.phase2.query_truth_sidecar.v2",
        "rows": truth_rows,
    }
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    record_result = build_d92_role_oracle_row_records(
        paired_evaluation_root=tmp_path / "out",
        truth_sidecar_path=truth_path,
        output_path=tmp_path / "records.jsonl",
        row_id="row",
        receiver="rx",
        seed=7,
        k_shot=5,
        new_class_count=2,
    )
    assert record_result["query_pair_count"] == 9
    rows = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 18
    for baseline_row, oracle_row in zip(rows[0::2], rows[1::2]):
        assert baseline_row["score_vector_sha256"] == oracle_row["score_vector_sha256"]
        assert baseline_row["model_state_sha256"] == oracle_row["model_state_sha256"]
        assert baseline_row["query_token"] == oracle_row["query_token"]


def test_predictor_monkeypatch_is_restored_when_baseline_raises(
    monkeypatch, tmp_path: Path
):
    _install_fake_baseline(monkeypatch, tmp_path, fail=True)
    capsule = _role_capsule(tmp_path / "role.json")
    original = d81_eval.predict_d42_unified_shrinkage_lda
    with pytest.raises(RuntimeError, match="synthetic baseline failure"):
        paired.run_d92_role_oracle_query_evaluation(
            role_capsule_path=capsule,
            expected_role_capsule_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),
            output_root=tmp_path / "out",
        )
    assert d81_eval.predict_d42_unified_shrinkage_lda is original
