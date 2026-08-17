from __future__ import annotations

import inspect
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi.stage2_d92_continuous_session_prediction import (
    ContinuousSessionPredictionError,
    _after_apply_manifest_lock,
    _registration_resource,
    _original_d42_f0_prediction,
    prepare_continuous_session_support_deltas,
    run_continuous_session_prediction,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
QUERY_TOKENS = np.asarray(("qid_a", "qid_b", "qid_c"))


@dataclass(frozen=True)
class _State:
    classes: tuple[str, ...]
    old_class_count: int
    log_diag_fp32: np.ndarray
    coef1_qint8: np.ndarray
    coef2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    intercept_fp16: np.ndarray
    covariance_policy: str = "test_policy"


def _state(classes: tuple[str, ...]) -> _State:
    count = len(classes)
    return _State(
        classes=classes,
        old_class_count=2,
        log_diag_fp32=np.ones(4, dtype=np.float32),
        coef1_qint8=np.arange(count * 4, dtype=np.int8).reshape(count, 4),
        coef2_qint8=np.zeros((count, 4), dtype=np.int8),
        scale1_fp16=np.ones((count, 1), dtype=np.float16),
        scale2_fp16=np.ones((count, 1), dtype=np.float16),
        intercept_fp16=np.zeros(count, dtype=np.float16),
    )


def _base_kwargs(tmp_path: Path, builder: object) -> dict[str, object]:
    return {
        "before_enrollment_package_root": "before-enrollment-package",
        "before_enrollment_seal_path": "before-enrollment.seal.json",
        "before_enrollment_seal_sha256": SHA_A,
        "before_apply_package_root": "before-apply-package",
        "before_apply_seal_path": "before-apply.seal.json",
        "before_apply_seal_sha256": SHA_B,
        "after_apply_package_root": "after-apply-package",
        "after_apply_seal_path": "after-apply.seal.json",
        "after_apply_seal_sha256": SHA_A,
        "prepared_delta_root": tmp_path / "prepared-deltas",
        "ground_component_dir": "ground-component",
        "ground_manifest_sha256": SHA_B,
        "schedules": {
            "singleton_forward": {
                "increments": (1, 1),
                "arrival_order": (0, 1),
            }
        },
        "output_root": tmp_path / "prediction",
        "device": "cpu",
        "session_builder": builder,
    }


class _RecordingBuilder:
    def __init__(self, *, future_open_sentinel: int = 0, duplicate_second: bool = False, query_drift: bool = False, bad_prediction: bool = False) -> None:
        self.future_open_sentinel = future_open_sentinel
        self.duplicate_second = duplicate_second
        self.query_drift = query_drift
        self.bad_prediction = bad_prediction
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        index = int(kwargs["session_index"])
        classes = ("cls_old_a", "cls_old_b") + tuple(
            f"cls_new_{value}" for value in range(index)
        )
        arriving_tokens = () if index == 0 else (("sid_new_0",) if index == 1 else (
            ("sid_new_0",) if self.duplicate_second else ("sid_new_1",)
        ))
        query_tokens = QUERY_TOKENS.copy()
        if index == 2 and self.query_drift:
            query_tokens[-1] = "qid_drift"
        predictions = np.asarray((classes[0], classes[-1], classes[1]))
        if self.bad_prediction:
            predictions[-1] = "cls_not_registered"
        cumulative = () if index == 0 else (("sid_new_0",) if index == 1 else ("sid_new_0", "sid_new_1"))
        return {
            "state": _state(classes),
            "ledger": {"session_index": index, "arrived_class_handles": classes[2:]},
            "arriving_support_tokens": arriving_tokens,
            "cumulative_support_tokens": cumulative,
            "query_tokens": query_tokens,
            "scenarios": np.asarray(SCENARIOS),
            "predicted_class_handles": predictions,
            "audit": {
                "lifecycle_state": "DA1_REG0" if index == 0 else f"DA1_REG1_S{index}",
                "session_index": index,
                "future_support_open_sentinel": self.future_open_sentinel,
                "past_token_duplicate_count": 0,
                "full_solve_count": 1,
                "d42_codec_count": 1,
                "query_truth_access": False,
                "query_fit_access": False,
                "query_update_access": False,
                "query_selection_access": False,
                "query_role_oracle_access": False,
                "query_class_quota_access": False,
                "query_global_reassignment": False,
                "query_decision_policy": "per_sample_all_registered_classes",
            },
            "resource_audit": {
                "registration_wall_time_ns": 100 + index,
                "registration_incremental_peak_working_set_bytes": 200 + index,
                "support_bytes": 300 + index,
                "state_bytes": 400 + index,
                "query_macs": len(classes) * 288,
                "head_latency_ns": 50 + index,
            },
        }


def test_truth_free_prediction_opens_only_arriving_support_and_seals_every_session(
    tmp_path: Path,
) -> None:
    """Would fail if a future delta were passed to the builder or artifacts overwrote."""

    builder = _RecordingBuilder()
    result = run_continuous_session_prediction(**_base_kwargs(tmp_path, builder))

    assert [call["session_index"] for call in builder.calls] == [0, 1, 2]
    assert [call["increment"] for call in builder.calls] == [0, 1, 1]
    assert all(
        not any(forbidden in key.lower() for forbidden in ("truth", "score", "role", "quota"))
        for call in builder.calls
        for key in call
    )
    sessions = result["schedules"]["singleton_forward"]["sessions"]
    assert [row["lifecycle_state"] for row in sessions] == ["DA1_REG1_S1", "DA1_REG1_S2"]
    assert result["future_support_open_sentinel"] == 0
    assert sessions[-1]["cumulative_support_token_count"] == 2
    assert sessions[-1]["query_macs"] == 4 * 288
    for row in sessions:
        destination = Path(row["output_root"])
        assert {
            path.name for path in destination.iterdir()
        } == {
            "prediction_artifact.npz",
            "fit_audit.json",
            "resource_audit.json",
            "execution_receipt.json",
            "COMMIT.json",
        }
        for path in destination.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ) == 0
    with pytest.raises(FileExistsError, match="output"):
        run_continuous_session_prediction(**_base_kwargs(tmp_path, builder))
    assert [call["session_index"] for call in builder.calls] == [0, 1, 2]


def test_prediction_seals_one_shared_da1_reg0_baseline_before_all_sessions(
    tmp_path: Path,
) -> None:
    """Would fail if a schedule could start REG1 without the frozen old-only head."""

    builder = _RecordingBuilder()
    result = run_continuous_session_prediction(**_base_kwargs(tmp_path, builder))

    assert [call["session_index"] for call in builder.calls] == [0, 1, 2]
    baseline = result["DA1_REG0"]
    assert baseline["lifecycle_state"] == "DA1_REG0"
    assert baseline["registered_class_count"] == 2
    assert Path(baseline["output_root"]).joinpath("prediction_artifact.npz").is_file()


def test_prediction_runtime_accepts_registration_wall_at_v2_300ms_limit() -> None:
    """Would fail if the runtime still enforced the superseded 150ms gate."""

    resource = _registration_resource(
        {
            "registration_wall_time_ns": 300_000_000,
            "registration_incremental_peak_working_set_bytes": 4 * 1024 * 1024,
        },
        state=_state(("cls_old_a", "cls_old_b")),
        support_bytes=10,
        registered_class_count=2,
    )

    assert resource["registration_wall_time_ns"] == 300_000_000
    assert resource["registration_wall_hard_max_ns"] == 300_000_000


def test_prediction_runtime_rejects_registration_wall_above_v2_300ms_limit() -> None:
    """Would fail if a runtime session above 300ms could publish."""

    with pytest.raises(ContinuousSessionPredictionError, match="registration wall hard gate failed"):
        _registration_resource(
            {
                "registration_wall_time_ns": 300_000_001,
                "registration_incremental_peak_working_set_bytes": 4 * 1024 * 1024,
            },
            state=_state(("cls_old_a", "cls_old_b")),
            support_bytes=10,
            registered_class_count=2,
        )


def test_frozen_da1_reg0_baseline_records_resources_without_applying_session_gate() -> None:
    """Would fail if the old-only baseline were charged to the REG1 session budget."""

    resource = _registration_resource(
        {
            "registration_wall_time_ns": 300_000_001,
            "registration_incremental_peak_working_set_bytes": 4 * 1024 * 1024 + 1,
        },
        state=_state(("cls_old_a", "cls_old_b")),
        support_bytes=10,
        registered_class_count=2,
        enforce_hard_gate=False,
    )

    assert resource["registration_wall_time_ns"] == 300_000_001
    assert resource["registration_incremental_peak_working_set_bytes"] == 4 * 1024 * 1024 + 1
    assert resource["registration_hard_gate_enforced"] is False
    assert resource["registration_resource_scope"] == "frozen_da1_reg0_baseline_rebuild"


def test_after_apply_lock_accepts_the_required_stage2b_to_stage2c_transition() -> None:
    """Would fail if enrollment/apply stages had to be equal across registration."""

    shared = {
        "receiver": "20-1",
        "seed": 713106,
        "k_shot": 10,
        "phase1_checkpoint_sha256": SHA_A,
        "feature_runtime_sha256": SHA_B,
        "method_lock_sha256": SHA_C,
    }
    before = {**shared, "stage": "stage2b"}
    after = {
        **shared,
        "stage": "stage2c",
        "profile": "apply_only",
        "registration_state": "after",
        "registered_classes": [
            {"class_handle": "cls_old_a"},
            {"class_handle": "cls_new_a"},
        ],
    }

    _after_apply_manifest_lock(
        before,
        after,
        SimpleNamespace(old_handles=("cls_old_a",), new_handles=("cls_new_a",)),
    )


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (_RecordingBuilder(future_open_sentinel=1), "future"),
        (_RecordingBuilder(duplicate_second=True), "duplicate"),
        (_RecordingBuilder(query_drift=True), "query token"),
        (_RecordingBuilder(bad_prediction=True), "registered"),
    ],
)
def test_truth_free_prediction_rejects_protocol_drift_before_publication(
    tmp_path: Path, builder: _RecordingBuilder, message: str
) -> None:
    """Would fail if a malformed session could publish a usable prediction artifact."""

    with pytest.raises(ContinuousSessionPredictionError, match=message):
        run_continuous_session_prediction(**_base_kwargs(tmp_path, builder))
    assert not (tmp_path / "prediction").exists()


def test_prediction_entry_has_no_truth_or_scorer_input_surface() -> None:
    """Would fail if prediction could receive truth, scorer output, role, or quota."""

    signature = inspect.signature(run_continuous_session_prediction)
    joined = " ".join(signature.parameters).lower()
    for forbidden in ("truth", "scor", "role", "quota"):
        assert forbidden not in joined
    assert "prepared_delta_root" in signature.parameters
    assert not any("after_enrollment" in name for name in signature.parameters)


def test_continuous_intermediate_state_uses_original_d42_f0_scorer() -> None:
    """Would fail if S1 bypassed the original D42 scorer through a new query kernel."""

    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
    from cvsrffi import stage2_d92_continuous_session as continuous

    classes = tuple(f"cls_{index}" for index in range(7))
    state = continuous.ContinuousD42State(
        schema=continuous.SCHEMA,
        classes=classes,
        old_class_count=6,
        log_diag_fp32=np.zeros(288, dtype=np.float32),
        coef1_qint8=np.zeros((7, 288), dtype=np.int8),
        coef2_qint8=np.zeros((7, 288), dtype=np.int8),
        scale1_fp16=np.ones((7, 3), dtype=np.float16),
        scale2_fp16=np.ones((7, 3), dtype=np.float16),
        intercept_fp16=np.zeros(7, dtype=np.float16),
        covariance_policy="standard_scaler_ledoit_wolf_singleton",
        da_anchor_id="a" * 64,
        support_transform_identity="b" * 64,
    )
    features = np.ones((2, 288), dtype=np.float32)

    expected = d42.predict_d42_unified_shrinkage_lda(
        state.to_d42_unified_state(), features
    )
    observed = _original_d42_f0_prediction(d42, state, features)

    assert np.array_equal(observed, expected)


def _hash_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _support_payload(classes: tuple[str, ...], *, scenario: str, k_shot: int = 10) -> dict[str, np.ndarray]:
    labels = np.repeat(np.arange(len(classes), dtype=np.int64), k_shot)
    ranks = np.tile(np.arange(k_shot, dtype=np.int64), len(classes))
    tokens = np.asarray(
        [f"sid_{scenario}_{classes[label]}_{rank}" for label, rank in zip(labels, ranks)]
    )
    iq = np.arange(len(labels) * 8, dtype=np.float32).reshape(len(labels), 2, 4)
    return {
        "support_leo_weak_iq": iq,
        "support_class_indices": labels,
        "support_rank_within_class": ranks,
        "support_tokens": tokens,
        "support_post_channel_iq_sha256": np.asarray(
            [_hash_text(str(token)) for token in tokens]
        ),
    }


def _manifest(classes: tuple[str, ...]) -> dict[str, object]:
    return {
        "registered_classes": [{"class_handle": handle} for handle in classes],
        "k_shot": 10,
        "receiver": "20-1",
        "seed": 713106,
        "package_root_sha256": SHA_C,
    }


def test_prepare_deltas_reads_only_enrollment_support_and_seals_canonical_new_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if future support stayed in one candidate-visible after package."""

    from cvsrffi import stage2_d92_continuous_session_prediction as prediction

    old = tuple(f"cls_old_{index}" for index in range(6))
    after = old + ("cls_new_z", "cls_new_b", "cls_new_d", "cls_new_a", "cls_new_c")
    payloads = {
        "before": {
            scenario: _support_payload(old, scenario=scenario) for scenario in SCENARIOS
        },
        "after": {
            scenario: _support_payload(after, scenario=scenario) for scenario in SCENARIOS
        },
    }
    calls: list[str] = []

    def fake_loader(root: str, **_kwargs: object):
        calls.append(root)
        key = "before" if root == "before" else "after"
        return payloads[key], _manifest(old if key == "before" else after), {"preopen": key}

    monkeypatch.setattr(prediction, "load_verified_somph_predictor_bundle", fake_loader)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    receipt = prepare_continuous_session_support_deltas(
        before_enrollment_package_root="before",
        before_enrollment_seal_path="before.seal",
        before_enrollment_seal_sha256=SHA_A,
        after_enrollment_package_root="after",
        after_enrollment_seal_path="after.seal",
        after_enrollment_seal_sha256=SHA_B,
        prepared_delta_root=prepared_root,
    )

    assert calls == ["before", "after"]
    assert receipt["new_class_handles"] == [
        "cls_new_a", "cls_new_b", "cls_new_c", "cls_new_d", "cls_new_z"
    ]
    assert receipt["future_support_open_sentinel"] == 0
    for index, handle in enumerate(receipt["new_class_handles"], start=1):
        delta = Path(receipt["deltas"][index - 1]["root"])
        manifest = delta / "manifest.json"
        assert manifest.is_file()
        assert stat.S_IMODE(manifest.stat().st_mode) & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ) == 0
        for scenario in SCENARIOS:
            payload = delta / f"{scenario}.npz"
            assert payload.is_file()
            with np.load(payload, allow_pickle=False) as archive:
                assert archive["class_handle"].tolist() == [handle]
                assert archive["support_tokens"].shape == (10,)


def test_prepare_deltas_rejects_old_support_token_drift_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if a mismatched old support pool could seed a replay ledger."""

    from cvsrffi import stage2_d92_continuous_session_prediction as prediction

    old = tuple(f"cls_old_{index}" for index in range(6))
    after = old + tuple(f"cls_new_{index}" for index in range(5))
    before_payload = {scenario: _support_payload(old, scenario=scenario) for scenario in SCENARIOS}
    after_payload = {scenario: _support_payload(after, scenario=scenario) for scenario in SCENARIOS}
    after_payload[SCENARIOS[0]]["support_tokens"][0] = "sid_drift"

    def fake_loader(root: str, **_kwargs: object):
        return (
            before_payload if root == "before" else after_payload,
            _manifest(old if root == "before" else after),
            {},
        )

    monkeypatch.setattr(prediction, "load_verified_somph_predictor_bundle", fake_loader)
    with pytest.raises(ContinuousSessionPredictionError, match="old support"):
        prepare_continuous_session_support_deltas(
            before_enrollment_package_root="before",
            before_enrollment_seal_path="before.seal",
            before_enrollment_seal_sha256=SHA_A,
            after_enrollment_package_root="after",
            after_enrollment_seal_path="after.seal",
            after_enrollment_seal_sha256=SHA_B,
            prepared_delta_root=tmp_path / "prepared",
        )
    assert not (tmp_path / "prepared").exists()
