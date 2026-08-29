from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase2_fasttrust_receiver_matrix import (
    FORMAL_SCENARIOS,
    FROZEN_CHECKPOINT_PATH,
    TARGET_RECEIVERS,
    build_receiver_matrix,
)
from cvsrffi.phase2_fasttrust_staging import stage_receiver_arrays
from cvsrffi.stage2_structured_late_block_scorer import (
    StructuredLateBlockScoringError,
    score_stage2c_predictions,
)
from cvsrffi import stage2_structured_late_block_runner as runner
from scripts import build_cvs_stage2_support_prototypes as prototype_builder
from scripts import run_phase2_fasttrust_receiver_confirmation as confirmation


def test_receiver_matrix_is_exactly_seven_receivers_by_three_scenarios() -> None:
    """Catch missing, duplicate, or cross-GPU receiver rows before launch."""

    rows = build_receiver_matrix(
        run_root="/runs/formal",
        checkpoint_path=FROZEN_CHECKPOINT_PATH,
        seed=713104,
    )

    assert len(rows) == 21
    assert {row["receiver"] for row in rows} == set(TARGET_RECEIVERS)
    assert {row["scenario"] for row in rows} == set(FORMAL_SCENARIOS)
    assert len({row["row_id"] for row in rows}) == 21
    for gpu, receiver in enumerate(TARGET_RECEIVERS):
        receiver_rows = [row for row in rows if row["receiver"] == receiver]
        assert len(receiver_rows) == 3
        assert {row["gpu"] for row in receiver_rows} == {gpu}
        assert {row["k_shot"] for row in receiver_rows} == {20}
        assert {row["seed"] for row in receiver_rows} == {713104}
        assert {row["state"] for row in receiver_rows} == {"DA1_REG1"}


def _write_prediction(path: Path) -> None:
    np.savez(
        path,
        query_ids=np.asarray(["q0", "q1", "q2", "q3"]),
        predicted_class_ids=np.asarray([0, 1, 2, 0], dtype=np.int64),
        scores=np.asarray(
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 1.0, 5.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def _write_truth(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"query_token": "q0", "true_class_index": 0},
                    {"query_token": "q1", "true_class_index": 1},
                    {"query_token": "q2", "true_class_index": 2},
                    {"query_token": "q3", "true_class_index": 2},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_stage2c_scorer_reports_old_new_macro_floor_after_exact_join(
    tmp_path: Path,
) -> None:
    """Catch a REG1 result being reduced to the old-only Stage2-B metric."""

    prediction = tmp_path / "predictions.npz"
    truth = tmp_path / "truth.json"
    output = tmp_path / "score.json"
    _write_prediction(prediction)
    _write_truth(truth)

    result = score_stage2c_predictions(
        prediction,
        truth,
        output_path=output,
        old_class_ids=[0, 1],
        class_names=["old-a", "old-b", "new-c"],
    )

    assert result["state"] == "DA1_REG1"
    assert result["overall_accuracy"] == pytest.approx(0.75)
    assert result["old_class_accuracy"] == pytest.approx(1.0)
    assert result["new_class_accuracy"] == pytest.approx(0.5)
    assert result["macro_accuracy"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert result["floor_accuracy"] == pytest.approx(0.5)
    assert result["per_class_accuracy"] == {
        "old-a": 1.0,
        "old-b": 1.0,
        "new-c": 0.5,
    }
    assert result["prediction_rows_verified_before_truth_open"] == 4


def test_stage2c_scorer_rejects_prediction_class_outside_registry_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch malformed predictions before the independent scorer opens truth."""

    prediction = tmp_path / "predictions.npz"
    np.savez(
        prediction,
        query_ids=np.asarray(["q0"]),
        predicted_class_ids=np.asarray([3], dtype=np.int64),
        scores=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    truth = tmp_path / "truth.json"
    truth.write_text("{}", encoding="utf-8")

    opened = False

    def fail_if_truth_is_opened(_path: str | Path):
        nonlocal opened
        opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(
        "cvsrffi.stage2_structured_late_block_scorer._load_truth_json",
        fail_if_truth_is_opened,
    )
    with pytest.raises(StructuredLateBlockScoringError, match="class registry"):
        score_stage2c_predictions(
            prediction,
            truth,
            output_path=tmp_path / "score.json",
            old_class_ids=[0, 1],
            class_names=["old-a", "old-b", "new-c"],
        )
    assert opened is False


def test_prototype_builder_accepts_frozen_fasttrust_checkpoint_and_any_matrix_row(
    tmp_path: Path,
) -> None:
    """Catch the historical 1-1/clear/CORE90 smoke lock in formal rows."""

    support = tmp_path / "support_leo_rain_weak_rx8-8_k20.npz"
    prototype = tmp_path / "prototypes_leo_rain_weak_rx8-8_k20.npz"
    config = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2",
        "split_id": "260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25",
        "checkpoint_path": FROZEN_CHECKPOINT_PATH,
        "support_path": str(support),
        "prototype_path": str(prototype),
        "candidate": "freq_f3_proj",
        "steps": 1,
        "learning_rate": 0.0005,
        "seed": 713104,
        "k_shot": 20,
    }

    resolved = prototype_builder._validate_config(config)  # noqa: SLF001
    prototype_builder._validate_row_binding(  # noqa: SLF001
        resolved, scene="leo_rain_weak", receiver="8-8"
    )


def test_runner_accepts_preregistered_trainable_fraction_bounds() -> None:
    """Catch the formal runner dropping the 3.4% FastTrust adapter fraction."""

    config = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "split_id": "split",
        "row_id": "RX8-8_leo_rain_weak_K20_S713104",
        "receiver": "8-8",
        "scenario": "leo_rain_weak",
        "seed": 713104,
        "k_shot": 20,
        "checkpoint_path": "/checkpoint.pth",
        "support_path": "/support.npz",
        "query_path": "/query.npz",
        "prototype_path": "/prototypes.npz",
        "candidate": "freq_f3_proj",
        "steps": 1,
        "learning_rate": 0.0005,
        "decision_rule": "frozen_prototype_cosine_v1",
        "min_trainable_fraction": 0.03,
        "max_trainable_fraction": 0.15,
    }

    resolved = runner._validate_config(config)  # noqa: SLF001
    assert resolved["min_trainable_fraction"] == 0.03
    assert resolved["max_trainable_fraction"] == 0.15


def test_staging_separates_truth_and_preserves_exact_k20(tmp_path: Path) -> None:
    """Catch labels/roles leaking into query payload or support/query overlap."""

    class_names = ["old-a", "new-b"]
    rows = []
    for class_index, class_name in enumerate(class_names):
        for role, count in (("support", 20), ("query", 3)):
            for rank in range(count):
                rows.append((class_index, class_name, role, rank))
    arrays = {
        "leo_weak_iq": np.asarray(
            [np.full((2, 4), index, dtype=np.float32) for index in range(len(rows))]
        ),
        "tx_ids": np.asarray([row[1] for row in rows]),
        "rx_ids": np.asarray(["8-8"] * len(rows)),
        "split_roles": np.asarray([row[2] for row in rows]),
        "split_ranks": np.asarray([row[3] for row in rows], dtype=np.int64),
        "canonical_physical_sample_ids": np.asarray(
            [f"physical-{index}" for index in range(len(rows))]
        ),
    }

    receipt = stage_receiver_arrays(
        {"leo_rain_weak": arrays},
        receiver="8-8",
        output_root=tmp_path / "predictor",
        truth_root=tmp_path / "truth",
        class_names=class_names,
        k_shot=20,
        token_salt="split-fixed",
    )

    with np.load(
        tmp_path / "predictor" / "support_leo_rain_weak_rx8-8_k20.npz",
        allow_pickle=False,
    ) as support:
        assert set(support.files) == {"received_iq", "support_labels"}
        assert support["received_iq"].shape[0] == 40
        assert np.bincount(support["support_labels"]).tolist() == [20, 20]
    with np.load(
        tmp_path / "predictor" / "query_leo_rain_weak_rx8-8_k20.npz",
        allow_pickle=False,
    ) as query:
        assert set(query.files) == {"received_iq", "query_ids"}
        assert query["received_iq"].shape[0] == 6
        query_ids = query["query_ids"].astype(str).tolist()
        assert len(set(query_ids)) == 6
        assert all("old-a" not in value and "new-b" not in value for value in query_ids)
    truth = json.loads(
        (tmp_path / "truth" / "truth_leo_rain_weak_rx8-8_k20.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(truth["rows"]) == 6
    assert {row["query_token"] for row in truth["rows"]} == set(query_ids)
    assert {row["true_class_index"] for row in truth["rows"]} == {0, 1}
    assert receipt["support_query_physical_disjoint"] is True


def test_score_gate_requires_all_21_predictions_before_any_truth_open(
    tmp_path: Path,
) -> None:
    """Catch early per-row scoring feeding target outcomes into unfinished rows."""

    matrix = build_receiver_matrix(
        run_root=str(tmp_path / "run"),
        checkpoint_path=FROZEN_CHECKPOINT_PATH,
        seed=713104,
    )
    with pytest.raises(ValueError, match="21/21"):
        confirmation.assert_all_predictions_complete(matrix)


def test_matrix_rejects_nonfrozen_checkpoint_identity() -> None:
    """Catch CONTROL/other-seed checkpoints being mislabeled as seed 713104."""

    with pytest.raises(ValueError, match="frozen FastTrust checkpoint"):
        build_receiver_matrix(
            run_root="/runs/formal",
            checkpoint_path="/runs/CONTROL/final_ssdg.pth",
            seed=713104,
        )


def test_prototype_builder_rejects_legacy_checkpoint_for_formal_seed(
    tmp_path: Path,
) -> None:
    """Catch the legacy CORE90 smoke checkpoint entering seed713104 rows."""

    config = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2",
        "split_id": "260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25",
        "checkpoint_path": prototype_builder.EXPECTED_CHECKPOINT,
        "support_path": str(tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"),
        "prototype_path": str(tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"),
        "candidate": "freq_f3_proj",
        "steps": 1,
        "learning_rate": 0.0005,
        "seed": 713104,
        "k_shot": 20,
    }
    with pytest.raises(ValueError, match="seed/checkpoint binding"):
        prototype_builder._validate_config(config)  # noqa: SLF001


def test_prediction_first_gate_fully_validates_all_rows_before_truth(tmp_path: Path) -> None:
    """Catch a present but incomplete prediction file passing the global gate."""

    matrix = build_receiver_matrix(
        run_root=str(tmp_path / "run"),
        checkpoint_path=FROZEN_CHECKPOINT_PATH,
        seed=713104,
    )
    for index, row in enumerate(matrix):
        destination = Path(row["prediction_output_dir"])
        destination.mkdir(parents=True)
        scores = np.zeros((1352, 26), dtype=np.float32)
        if index == len(matrix) - 1:
            scores[0, 0] = np.nan
        np.savez(
            destination / "predictions.npz",
            query_ids=np.asarray([f"q{index}-{q}" for q in range(1352)]),
            predicted_class_ids=np.zeros(1352, dtype=np.int64),
            scores=scores,
        )
    with pytest.raises(ValueError, match="prediction-first validation"):
        confirmation.assert_all_predictions_complete(matrix)
