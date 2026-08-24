from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import cvsrffi.stage2_meta_adapter_scorer as scorer  # noqa: E402


QUERY_IDS = np.asarray(["q-1", "q-2", "q-3", "q-4"])
DA0 = np.asarray([10, 10, 20, 20], dtype=np.int64)
DA1 = np.asarray([10, 20, 20, 20], dtype=np.int64)
SCORES = np.asarray(
    [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
    dtype=np.float32,
)
DA1_SCORES = np.asarray(
    [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]],
    dtype=np.float32,
)
REGISTERED_CLASS_IDS = np.asarray([10, 20], dtype=np.int64)
TRUTH = {
    "schema": "cvs.truth.v1",
    "rows": [
        {"query_token": "q-3", "true_class_index": 20},
        {"query_token": "q-1", "true_class_index": 10},
        {"query_token": "q-4", "true_class_index": 20},
        {"query_token": "q-2", "true_class_index": 10},
    ],
}


def _write_prediction(
    path: Path,
    *,
    query_ids: np.ndarray = QUERY_IDS,
    predicted: np.ndarray = DA0,
    scores: np.ndarray = SCORES,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    np.savez(
        path,
        query_ids=query_ids,
        predicted_class_ids=predicted,
        scores=scores,
        **(extra or {}),
    )


def _write_truth(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(payload or TRUTH),
        encoding="utf-8",
    )


def _write_receipt(
    root: Path,
    *,
    da0_name: str = "predictions_DA0_REG0.npz",
    da1_name: str = "predictions_DA1_REG0.npz",
    candidate_id: str = "tri-r4",
    bundle_id: str = "bundle-001",
) -> None:
    payload = {
        "status": "PREDICTIONS_COMPLETE",
        "states": ["DA0_REG0", "DA1_REG0"],
        "states_same_row": True,
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "operating_point": "K2/new2",
        "seed": 713101,
        "k_shot": 2,
        "candidate_id": candidate_id,
        "bundle_id": bundle_id,
        "registered_class_ids": REGISTERED_CLASS_IDS.tolist(),
        "prediction_paths": {
            "DA0_REG0": str((root / da0_name).resolve()),
            "DA1_REG0": str((root / da1_name).resolve()),
        },
    }
    (root / "receipt.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_scorer_requires_identical_query_ids_and_reports_reg0_na(tmp_path: Path) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    _write_receipt(tmp_path)

    score = scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)

    assert score.da0.state == "DA0_REG0"
    assert score.da1.state == "DA1_REG0"
    assert score.da0.seen_new_acc is None
    assert score.da1.h_old_new is None
    assert score.da0.mean_old_acc == pytest.approx(1.0)
    assert score.da1.mean_old_acc == pytest.approx(0.75)
    assert score.da0.old_class_floor == pytest.approx(1.0)
    assert score.da1.old_class_floor == pytest.approx(0.5)
    assert score.mean_delta_pp == pytest.approx(-25.0)
    assert score.floor_delta_pp == pytest.approx(-50.0)


def test_scorer_joins_full_cvs_sidecar_but_scores_target_old_only(
    tmp_path: Path,
) -> None:
    query_ids = np.asarray(["q-1", "q-2", "q-3", "q-4", "q-new"])
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth_sidecar.json"
    binding_path = tmp_path / "class_binding.json"
    scores = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.6, 0.4]],
        dtype=np.float32,
    )
    _write_prediction(
        da0_path,
        query_ids=query_ids,
        predicted=np.asarray([10, 10, 20, 20, 10], dtype=np.int64),
        scores=scores,
    )
    _write_prediction(
        da1_path,
        query_ids=query_ids,
        predicted=np.asarray([10, 20, 20, 20, 10], dtype=np.int64),
        scores=np.asarray(
            [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.6, 0.4]],
            dtype=np.float32,
        ),
    )
    _write_receipt(tmp_path)
    truth_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.query_truth_sidecar.v2",
                "rows": [
                    {
                        "query_token": "q-3",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_old",
                        "true_class_handle": "class-20",
                    },
                    {
                        "query_token": "q-new",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_new",
                        "true_class_handle": "unregistered-class",
                    },
                    {
                        "query_token": "q-1",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_old",
                        "true_class_handle": "class-10",
                    },
                    {
                        "query_token": "q-4",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_old",
                        "true_class_handle": "class-20",
                    },
                    {
                        "query_token": "q-2",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_old",
                        "true_class_handle": "class-10",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    binding_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.d20_adv3b02_class_binding.v2",
                "entries": [
                    {"class_index": 10, "registered_class_handle": "class-10"},
                    {"class_index": 20, "registered_class_handle": "class-20"},
                ],
            }
        ),
        encoding="utf-8",
    )

    score = scorer.score_meta_adapter_pair(
        da0_path,
        da1_path,
        truth_path,
        class_binding_path=binding_path,
    )

    assert score.da0.query_ids == ("q-1", "q-2", "q-3", "q-4")
    assert score.da0.mean_old_acc == pytest.approx(1.0)
    assert score.da1.mean_old_acc == pytest.approx(0.75)
    assert "q-new" not in score.da0.query_ids


def test_scorer_filters_numeric_cvs_truth_to_receipt_scenario(
    tmp_path: Path,
) -> None:
    query_ids = np.asarray(["q-old-10", "q-old-20", "q-new"])
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth_sidecar.json"
    scores = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]], dtype=np.float32)
    _write_prediction(
        da0_path,
        query_ids=query_ids,
        predicted=np.asarray([10, 20, 10], dtype=np.int64),
        scores=scores,
    )
    _write_prediction(
        da1_path,
        query_ids=query_ids,
        predicted=np.asarray([10, 10, 10], dtype=np.int64),
        scores=np.asarray([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4]], dtype=np.float32),
    )
    _write_receipt(tmp_path)
    truth_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.query_truth_sidecar.v2",
                "rows": [
                    {
                        "query_token": "q-old-10",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_old",
                        "true_class_index": 10,
                    },
                    {
                        "query_token": "q-old-20",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_old",
                        "true_class_index": 20,
                    },
                    {
                        "query_token": "q-new",
                        "scenario": "leo_clear_weak",
                        "evaluation_role": "target_new",
                        "true_class_index": 30,
                    },
                    {
                        "query_token": "other-old-10",
                        "scenario": "leo_rain_weak",
                        "evaluation_role": "target_old",
                        "true_class_index": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    score = scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)

    assert score.da0.query_ids == ("q-old-10", "q-old-20")
    assert score.da0.mean_old_acc == pytest.approx(1.0)
    assert score.da1.mean_old_acc == pytest.approx(0.5)
    assert "q-new" not in score.da0.query_ids


def test_scorer_rejects_itemwise_row_id_drift_before_truth_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(
        da1_path,
        query_ids=QUERY_IDS[[1, 0, 2, 3]],
        predicted=DA1,
        scores=DA1_SCORES,
    )
    _write_truth(truth_path)
    _write_receipt(tmp_path)

    truth_opened = False

    def forbidden_truth_open(_path: str | Path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth", forbidden_truth_open)
    with pytest.raises(ValueError, match="same ordered query IDs"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert truth_opened is False


def test_scorer_rejects_invalid_prediction_before_truth_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path, extra={"query_truth": np.asarray([10, 10, 20, 20])})
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    _write_receipt(tmp_path)

    truth_opened = False

    def forbidden_truth_open(_path: str | Path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth", forbidden_truth_open)
    with pytest.raises(ValueError, match="prediction artifact"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert truth_opened is False


def test_promotion_requires_both_mean_and_floor_thresholds() -> None:
    assert scorer.summarize_rows(mean_delta_pp=1.1, floor_delta_pp=0.4).promote is False
    assert scorer.summarize_rows(mean_delta_pp=0.9, floor_delta_pp=1.0).promote is False
    assert scorer.summarize_rows(mean_delta_pp=1.0, floor_delta_pp=0.5).promote is True
    assert scorer.summarize_rows(
        mean_delta_pp=1.0 - 1.0e-12,
        floor_delta_pp=0.5 - 1.0e-12,
    ).promote is True


def test_score_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    _write_receipt(tmp_path)
    output_path.write_text("keep", encoding="utf-8")

    score = scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    with pytest.raises(FileExistsError, match="already exists"):
        scorer.write_score_json(score, output_path)
    assert output_path.read_text(encoding="utf-8") == "keep"


def test_scorer_requires_task10_receipt_before_opening_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    truth_opened = False

    def forbidden_truth_open(_path: str | Path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth", forbidden_truth_open)
    with pytest.raises(ValueError, match="receipt"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert truth_opened is False


def test_receipt_binds_state_paths_and_complete_row_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    _write_receipt(tmp_path)
    truth_opened = False

    def forbidden_truth_open(_path: str | Path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth", forbidden_truth_open)
    with pytest.raises(ValueError, match="DA0_REG0.*path|state path"):
        scorer.score_meta_adapter_pair(da1_path, da0_path, truth_path)
    assert truth_opened is False

    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receiver")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receiver"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert truth_opened is False


def test_scorer_rejects_non_argmax_prediction_and_unknown_class(
    tmp_path: Path,
) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(
        da1_path,
        predicted=DA1,
        scores=SCORES,
    )
    _write_truth(truth_path)
    _write_receipt(tmp_path)
    with pytest.raises(ValueError, match="argmax"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)

    _write_prediction(da1_path, predicted=np.asarray([10, 30, 20, 20]), scores=DA1_SCORES)
    with pytest.raises(ValueError, match="registered class"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)


def test_scorer_rejects_score_width_and_nonregistered_truth(tmp_path: Path) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path, scores=np.ones((4, 3), dtype=np.float32))
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    _write_receipt(tmp_path)
    with pytest.raises(ValueError, match="score.*column|columns"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)

    _write_prediction(da0_path)
    truth = json.loads(json.dumps(TRUTH))
    truth["rows"][0]["true_class_index"] = 30
    _write_truth(truth_path, truth)
    with pytest.raises(ValueError, match="truth.*registered class"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)


def test_score_contains_bound_row_identity_and_matrix_rejects_mixed_json(
    tmp_path: Path,
) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1, scores=DA1_SCORES)
    _write_truth(truth_path)
    _write_receipt(tmp_path)
    score = scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    payload = score.to_dict()
    assert payload["schema"] == scorer.SCORE_SCHEMA
    assert payload["status"] == "ANALYZED"
    assert payload["candidate_id"] == "tri-r4"
    assert payload["bundle_id"] == "bundle-001"
    assert payload["row_id"]
    assert payload["row"]["receiver"] == "20-1"

    missing_same_row = dict(payload)
    missing_same_row.pop("same_row_ids")
    with pytest.raises(ValueError, match="same_row_ids"):
        scorer.summarize_meta_adapter_matrix([missing_same_row])

    mixed = json.loads(json.dumps(payload))
    mixed["candidate_id"] = "other-candidate"
    mixed["row"]["candidate_id"] = "other-candidate"
    mixed["row_id"] = "other-row"
    with pytest.raises(ValueError, match="candidate_id"):
        scorer.summarize_meta_adapter_matrix([payload, mixed])

    wrong_schema = json.loads(json.dumps(payload))
    wrong_schema["schema"] = "arbitrary"
    with pytest.raises(ValueError, match="schema"):
        scorer.summarize_meta_adapter_matrix([wrong_schema])

    wrong_status = json.loads(json.dumps(payload))
    wrong_status["status"] = "RUNNING"
    with pytest.raises(ValueError, match="status"):
        scorer.summarize_meta_adapter_matrix([wrong_status])

    mixed_bundle = json.loads(json.dumps(payload))
    mixed_bundle["bundle_id"] = "bundle-002"
    with pytest.raises(ValueError, match="bundle_id"):
        scorer.summarize_meta_adapter_matrix([payload, mixed_bundle])

    duplicate_row = json.loads(json.dumps(payload))
    with pytest.raises(ValueError, match="row_id"):
        scorer.summarize_meta_adapter_matrix([payload, duplicate_row])


def test_matrix_target_range_is_explicit() -> None:
    with pytest.raises(ValueError, match="Target5"):
        scorer.summarize_meta_adapter_matrix([], expected_target="Target5")


MATRIX_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
MATRIX_OPERATING_POINTS = (
    ("K10/new5", 10),
    ("K10/new10", 10),
    ("K10/new20", 10),
    ("K5/new20", 5),
    ("K1/new20", 1),
)
TARGET25_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")


def _matrix_score_payload(
    *,
    receiver: str,
    scenario: str,
    operating_point: str,
    k_shot: int,
    seed: int = 713101,
    split_suffix: str = "",
) -> dict[str, object]:
    state0 = scorer.StateScore(
        state="DA0_REG0",
        registration_state="REG0",
        query_ids=("q-1", "q-2"),
        mean_old_acc=0.50,
        old_class_floor=0.40,
        per_class_accuracy={"10": 0.5, "20": 0.5},
        per_class_correct={"10": 1, "20": 1},
        per_class_total={"10": 2, "20": 2},
        micro_old_acc=0.50,
    )
    state1 = scorer.StateScore(
        state="DA1_REG0",
        registration_state="REG0",
        query_ids=("q-1", "q-2"),
        mean_old_acc=0.52,
        old_class_floor=0.41,
        per_class_accuracy={"10": 0.52, "20": 0.52},
        per_class_correct={"10": 1, "20": 1},
        per_class_total={"10": 2, "20": 2},
        micro_old_acc=0.52,
    )
    row = {
        "candidate_id": "tri-r4",
        "bundle_id": "bundle-001",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": (
            f"split-{receiver}-{scenario}-{operating_point}{split_suffix}"
        ),
        "receiver": receiver,
        "scenario": scenario,
        "operating_point": operating_point,
        "seed": seed,
        "k_shot": k_shot,
    }
    return scorer.PairedStage2BScore(
        da0=state0,
        da1=state1,
        mean_delta_pp=2.0,
        floor_delta_pp=1.0,
        candidate_id="tri-r4",
        bundle_id="bundle-001",
        row_id=scorer._make_row_id(row),
        row=row,
        registered_class_ids=(10, 20),
    ).to_dict()


def _complete_matrix(receivers: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        _matrix_score_payload(
            receiver=receiver,
            scenario=scenario,
            operating_point=operating_point,
            k_shot=k_shot,
        )
        for receiver in receivers
        for operating_point, k_shot in MATRIX_OPERATING_POINTS
        for scenario in MATRIX_SCENARIOS
    ]


def test_target5_requires_complete_fifteen_row_cartesian_product() -> None:
    scores = _complete_matrix(("20-1",))

    decision = scorer.summarize_meta_adapter_matrix(
        scores,
        expected_target="Target5",
    )

    assert decision.target == "Target5"
    assert decision.row_count == 15
    assert decision.verdict == "PROMOTE_TO_TARGET25"


def test_target25_rejects_count_only_rows_without_required_cartesian_product() -> None:
    scores = [
        _matrix_score_payload(
            receiver="20-1",
            scenario="leo_clear_weak",
            operating_point="K10/new5",
            k_shot=10,
            split_suffix=f"-{index}",
        )
        for index in range(25)
    ]

    with pytest.raises(ValueError, match="Cartesian|combination|duplicate"):
        scorer.summarize_meta_adapter_matrix(scores, expected_target="Target25")


def test_target25_accepts_complete_seventy_five_row_cartesian_product() -> None:
    scores = _complete_matrix(TARGET25_RECEIVERS)

    decision = scorer.summarize_meta_adapter_matrix(
        scores,
        expected_target="Target25",
    )

    assert decision.target == "Target25"
    assert decision.row_count == 75


def test_matrix_rejects_missing_extra_and_mixed_seed_combinations() -> None:
    scores = _complete_matrix(("20-1",))
    missing_and_extra = json.loads(json.dumps(scores))
    missing_and_extra[-1]["row"]["scenario"] = "leo_extra_weak"
    missing_and_extra[-1]["row_id"] = scorer._make_row_id(
        missing_and_extra[-1]["row"]
    )
    with pytest.raises(ValueError, match="missing.*extra|extra.*missing"):
        scorer.summarize_meta_adapter_matrix(
            missing_and_extra,
            expected_target="Target5",
        )

    mixed_seed = json.loads(json.dumps(scores))
    mixed_seed[-1]["row"]["seed"] = 713102
    mixed_seed[-1]["row_id"] = scorer._make_row_id(mixed_seed[-1]["row"])
    with pytest.raises(ValueError, match="single seed"):
        scorer.summarize_meta_adapter_matrix(mixed_seed, expected_target="Target5")


def test_matrix_rejects_k_shot_that_disagrees_with_operating_point() -> None:
    scores = _complete_matrix(("20-1",))
    wrong_k = json.loads(json.dumps(scores))
    wrong_k[0]["row"]["k_shot"] = 5
    wrong_k[0]["row_id"] = scorer._make_row_id(wrong_k[0]["row"])

    with pytest.raises(ValueError, match="k_shot.*operating_point"):
        scorer.summarize_meta_adapter_matrix(wrong_k, expected_target="Target5")
