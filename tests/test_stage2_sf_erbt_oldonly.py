from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from cvsrffi.stage2_sf_erbt_oldonly import (
    OldOnlyERBTError,
    export_old_only_holdout,
    fit_old_only_erbt,
    make_fft96,
    run_old_only_prediction,
    score_old_only_predictions,
)
from cvsrffi.target_only_progressive_adapt import TargetPrototypeHead


def _source_pool(path, *, overlap: bool = False) -> str:
    labels = np.repeat(np.arange(6, dtype=np.int64), 20)
    ranks = np.tile(np.arange(20, dtype=np.int64), 6)
    iq = np.arange(120 * 2 * 256, dtype=np.float32).reshape(120, 2, 256) / 1000.0
    tokens = np.asarray([f"sid_{index:03d}" for index in range(120)])
    if overlap:
        tokens[10] = tokens[0]
    np.savez(
        path,
        support_pool_leo_weak_iq=iq,
        support_pool_class_indices=labels,
        support_pool_rank_within_class=ranks,
        support_pool_tokens=tokens,
    )
    return hashlib.sha256(iq[ranks < 10].tobytes()).hexdigest()


def test_export_creates_label_blind_holdout_and_truth_sidecar(tmp_path):
    source = tmp_path / "pool.npz"
    expected_sha = _source_pool(source)

    receipt = export_old_only_holdout(
        source,
        tmp_path / "export",
        k_shot=10,
        expected_support_iq_sha256=expected_sha,
        capsule_id="capsule-old-only",
        split_id="split-k10-k10holdout",
        adaptation_capsule_id="capsule-old-only",
        adaptation_split_id="split-k10-adaptation",
    )

    with np.load(tmp_path / "export" / "support.npz", allow_pickle=False) as support:
        assert set(support.files) == {"received_iq", "support_labels", "support_physical_ids"}
        assert support["received_iq"].shape == (60, 2, 256)
    with np.load(tmp_path / "export" / "query.npz", allow_pickle=False) as query:
        assert set(query.files) == {"received_iq", "query_ids"}
        assert query["received_iq"].shape == (60, 2, 256)
    with np.load(tmp_path / "export" / "truth.npz", allow_pickle=False) as truth:
        assert set(truth.files) == {"query_ids", "query_labels"}
        assert np.bincount(truth["query_labels"], minlength=6).tolist() == [10] * 6
    assert receipt["protocol_schema"] == "p2_min_v1"
    assert receipt["phase2_data_status"] == "VALIDATED_ONCE"
    assert receipt["support_query_physical_id_overlap"] == 0


def test_export_rejects_support_byte_drift_without_outputs(tmp_path):
    source = tmp_path / "pool.npz"
    _source_pool(source)

    with pytest.raises(OldOnlyERBTError, match="support IQ binding mismatch"):
        export_old_only_holdout(
            source,
            tmp_path / "export",
            k_shot=10,
            expected_support_iq_sha256="0" * 64,
            capsule_id="capsule-old-only",
            split_id="split-k10-k10holdout",
            adaptation_capsule_id="capsule-old-only",
            adaptation_split_id="split-k10-adaptation",
        )

    assert not (tmp_path / "export").exists()


def test_export_rejects_support_holdout_id_overlap(tmp_path):
    source = tmp_path / "pool.npz"
    expected_sha = _source_pool(source, overlap=True)

    with pytest.raises(OldOnlyERBTError, match="physical IDs overlap"):
        export_old_only_holdout(
            source,
            tmp_path / "export",
            k_shot=10,
            expected_support_iq_sha256=expected_sha,
            capsule_id="capsule-old-only",
            split_id="split-k10-k10holdout",
            adaptation_capsule_id="capsule-old-only",
            adaptation_split_id="split-k10-adaptation",
        )


def test_export_rejects_non_k10_matrix(tmp_path):
    source = tmp_path / "pool.npz"
    expected_sha = _source_pool(source)

    with pytest.raises(OldOnlyERBTError, match="locked to K10"):
        export_old_only_holdout(
            source,
            tmp_path / "export",
            k_shot=5,
            expected_support_iq_sha256=expected_sha,
            capsule_id="capsule-old-only",
            split_id="split-k5-invalid",
            adaptation_capsule_id="capsule-old-only",
            adaptation_split_id="split-k10-adaptation",
        )


def test_old_only_erbt_fits_balanced_support_and_predicts_all_classes():
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(6), 10)
    identity = np.zeros((60, 160), dtype=np.float32)
    fft = np.zeros((60, 96), dtype=np.float32)
    for class_id in range(6):
        mask = labels == class_id
        identity[mask, class_id] = 5.0
        fft[mask, class_id] = 2.0
    identity += rng.normal(0.0, 0.01, identity.shape).astype(np.float32)
    fft += rng.normal(0.0, 0.01, fft.shape).astype(np.float32)

    state = fit_old_only_erbt(identity, fft, labels, class_ids=tuple(range(6)), seed=713101)
    prediction = state.predict(identity, fft)

    assert np.array_equal(prediction, labels)
    assert state.audit["arm"] == "M29-FFT96-A4"
    assert state.audit["method_lock"] == "D92-E0-NORF32"
    assert state.audit["rf32_used"] is False
    assert state.audit["registration_state"] == "REG0"
    assert state.audit["d92_registration_balanced_active"] is False


def test_old_only_erbt_scores_zero_identity_row_from_valid_fft_block():
    labels = np.repeat(np.arange(6), 10)
    identity = np.zeros((60, 160), dtype=np.float32)
    fft = np.zeros((60, 96), dtype=np.float32)
    for class_id in range(6):
        mask = labels == class_id
        identity[mask, class_id] = 1.0
        fft[mask, class_id] = 1.0
    state = fit_old_only_erbt(
        identity,
        fft,
        labels,
        class_ids=tuple(range(6)),
        seed=713101,
    )

    query_identity = np.zeros((1, 160), dtype=np.float32)
    query_fft = np.zeros((1, 96), dtype=np.float32)
    query_fft[0, 2] = 1.0
    logits = state.score(query_identity, query_fft)

    assert logits.shape == (1, 6)
    assert np.isfinite(logits).all()


def test_old_only_erbt_rejects_query_with_both_feature_blocks_degenerate():
    labels = np.repeat(np.arange(6), 10)
    identity = np.zeros((60, 160), dtype=np.float32)
    fft = np.zeros((60, 96), dtype=np.float32)
    for class_id in range(6):
        mask = labels == class_id
        identity[mask, class_id] = 1.0
        fft[mask, class_id] = 1.0
    state = fit_old_only_erbt(
        identity,
        fft,
        labels,
        class_ids=tuple(range(6)),
        seed=713101,
    )

    with pytest.raises(OldOnlyERBTError, match="feature row is degenerate"):
        state.score(
            np.zeros((1, 160), dtype=np.float32),
            np.zeros((1, 96), dtype=np.float32),
        )


def test_fft96_is_deterministic_and_has_locked_dimension():
    rows = np.ones((2, 2, 256), dtype=np.float32)
    first = make_fft96(rows)
    second = make_fft96(rows.copy())
    assert first.shape == (2, 96)
    assert np.array_equal(first, second)


class _TinyEmbeddingModel(torch.nn.Module):
    def forward(self, rows, return_aux=False):
        embedding = rows.flatten(1)[:, :160]
        return {"z_id": embedding}


def _tiny_bundle_loader(path, *, device):
    weight = torch.eye(6, 160)
    model = _TinyEmbeddingModel().to(device).eval()
    head = TargetPrototypeHead(weight, tuple(range(6)), scale=16.0).to(device).eval()
    return model, head, {
        "schema": "cvs.sf_tapft.v1",
        "query_input_capability": False,
        "capsule_id": "capsule-old-only",
        "split_id": "split-k10-adaptation",
    }


def test_prediction_is_truth_blind_and_emits_two_reg0_arms(tmp_path, monkeypatch):
    support_iq = np.zeros((60, 2, 256), dtype=np.float32)
    query_iq = np.zeros((12, 2, 256), dtype=np.float32)
    labels = np.repeat(np.arange(6), 10)
    for class_id in range(6):
        support_iq[labels == class_id, 0, class_id] = 5.0
        query_iq[2 * class_id : 2 * class_id + 2, 0, class_id] = 5.0
    np.savez(
        tmp_path / "support.npz",
        received_iq=support_iq,
        support_labels=labels,
        support_physical_ids=np.asarray([f"s{index}" for index in range(60)]),
    )
    np.savez(
        tmp_path / "query.npz",
        received_iq=query_iq,
        query_ids=np.asarray([f"q{index}" for index in range(12)]),
    )
    (tmp_path / "handle.json").write_text(
        json.dumps(
            {
                "schema": "cvs.sf_erbt_oldonly_export.v1",
                "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE",
                "capsule_id": "capsule-old-only",
                "split_id": "split-k10-k10holdout",
                "adaptation_capsule_id": "capsule-old-only",
                "adaptation_split_id": "split-k10-adaptation",
                "k_shot": 10,
                "class_count": 6,
                "support_rows": 60,
                "query_rows": 12,
                "registration_state": "REG0",
                "query_truth_in_predictor": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        torch,
        "frombuffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("NumPy buffer bridge used")),
    )
    monkeypatch.setattr(
        torch.Tensor,
        "numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Torch NumPy bridge used")),
    )
    receipt = run_old_only_prediction(
        bundle_path=tmp_path / "bundle.pt",
        support_path=tmp_path / "support.npz",
        query_path=tmp_path / "query.npz",
        data_handle_path=tmp_path / "handle.json",
        output_root=tmp_path / "prediction",
        seed=713101,
        device="cpu",
        bundle_loader=_tiny_bundle_loader,
    )

    with np.load(tmp_path / "prediction" / "predictions.npz", allow_pickle=False) as data:
        assert set(data.files) == {"query_ids", "sf_head_predictions", "sf_erbt_predictions"}
        assert data["sf_head_predictions"].shape == (12,)
        assert data["sf_erbt_predictions"].shape == (12,)
    assert receipt["query_truth_opened"] is False
    assert receipt["registration_state"] == "REG0"
    assert receipt["erbt_d92_registration_balanced_active"] is False
    assert receipt["method_lock"] == "D92-E0-NORF32"
    assert receipt["rf32_used"] is False


def test_prediction_rejects_query_labels_before_bundle_load(tmp_path):
    np.savez(
        tmp_path / "query.npz",
        received_iq=np.zeros((1, 2, 256), dtype=np.float32),
        query_ids=np.asarray(["q0"]),
        query_labels=np.asarray([0]),
    )

    with pytest.raises(OldOnlyERBTError, match="query allowlist mismatch"):
        run_old_only_prediction(
            bundle_path=tmp_path / "bundle.pt",
            support_path=tmp_path / "missing-support.npz",
            query_path=tmp_path / "query.npz",
            data_handle_path=tmp_path / "missing-handle.json",
            output_root=tmp_path / "prediction",
            seed=713101,
            device="cpu",
            bundle_loader=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loader opened")),
        )


def test_prediction_rejects_cross_capsule_bundle(tmp_path):
    support_iq = np.zeros((60, 2, 256), dtype=np.float32)
    labels = np.repeat(np.arange(6), 10)
    np.savez(
        tmp_path / "support.npz",
        received_iq=support_iq,
        support_labels=labels,
        support_physical_ids=np.asarray([f"s{index}" for index in range(60)]),
    )
    np.savez(
        tmp_path / "query.npz",
        received_iq=np.zeros((6, 2, 256), dtype=np.float32),
        query_ids=np.asarray([f"q{index}" for index in range(6)]),
    )
    (tmp_path / "handle.json").write_text(
        json.dumps(
            {
                "schema": "cvs.sf_erbt_oldonly_export.v1",
                "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE",
                "capsule_id": "capsule-old-only",
                "split_id": "split-k10-k10holdout",
                "adaptation_capsule_id": "capsule-old-only",
                "adaptation_split_id": "split-k10-adaptation",
                "k_shot": 10,
                "class_count": 6,
                "support_rows": 60,
                "query_rows": 6,
                "registration_state": "REG0",
                "query_truth_in_predictor": False,
            }
        ),
        encoding="utf-8",
    )

    def wrong_bundle(path, *, device):
        model, head, audit = _tiny_bundle_loader(path, device=device)
        return model, head, {**audit, "capsule_id": "other-capsule"}

    with pytest.raises(OldOnlyERBTError, match="bundle/data binding mismatch"):
        run_old_only_prediction(
            bundle_path=tmp_path / "bundle.pt",
            support_path=tmp_path / "support.npz",
            query_path=tmp_path / "query.npz",
            data_handle_path=tmp_path / "handle.json",
            output_root=tmp_path / "prediction",
            seed=713101,
            device="cpu",
            bundle_loader=wrong_bundle,
        )


def test_truth_last_scorer_reports_mean_floor_and_delta(tmp_path):
    query_ids = np.asarray([f"q{index}" for index in range(60)])
    truth = np.repeat(np.arange(6), 10)
    head = truth.copy()
    head[50:] = 0
    np.savez(
        tmp_path / "predictions.npz",
        query_ids=query_ids,
        sf_head_predictions=head,
        sf_erbt_predictions=truth,
    )
    np.savez(
        tmp_path / "truth.npz",
        query_ids=query_ids,
        query_labels=truth,
    )
    binding = {
        "capsule_id": "capsule-old-only",
        "split_id": "split-k10-k10holdout",
        "k_shot": 10,
        "support_rows": 60,
        "query_rows": 60,
        "registration_state": "REG0",
    }
    (tmp_path / "prediction_receipt.json").write_text(
        json.dumps({**binding, "schema": "cvs.sf_erbt_oldonly_prediction.v1"}),
        encoding="utf-8",
    )
    (tmp_path / "data_handle.json").write_text(
        json.dumps({**binding, "schema": "cvs.sf_erbt_oldonly_export.v1"}),
        encoding="utf-8",
    )

    result = score_old_only_predictions(
        tmp_path / "predictions.npz",
        tmp_path / "truth.npz",
        tmp_path / "prediction_receipt.json",
        tmp_path / "data_handle.json",
        tmp_path / "score.json",
    )

    assert result["sf_head"]["accuracy"] == pytest.approx(5 / 6)
    assert result["sf_head"]["class_floor"] == 0.0
    assert result["sf_erbt"]["accuracy"] == 1.0
    assert result["sf_erbt"]["class_floor"] == 1.0
    assert result["sf_erbt_minus_sf_head"]["accuracy"] == pytest.approx(1 / 6)


def test_cli_help_runs_from_repo_root_without_pythonpath():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(root / "code" / "scripts" / "run_sf_erbt_oldonly.py"), "--help"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
