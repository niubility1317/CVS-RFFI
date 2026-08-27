from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cvsrffi.stage2_sf_tapft_query_closure as query_module
from cvsrffi.stage2_sf_tapft_query_closure import (
    QueryClosureError,
    load_clean_single_pair_strict,
    run_clean_query_prediction,
    score_clean_query_prediction,
)
from cvsrffi.target_only_progressive_adapt import SFTAPFTConfig, TargetPrototypeHead


class _TinyModel(torch.nn.Module):
    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = int(offset)

    def forward(self, rows, return_aux=False):
        embedding = rows.flatten(1)[:, :160].roll(self.offset, dims=1)
        return {"z_id": embedding}


class _ParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


def _pair_loader(path, support_path, *, device, expected_target_binding):
    weight = torch.eye(6, 160, device=device)
    da0 = _TinyModel(offset=1).to(device).eval()
    da1 = _TinyModel(offset=0).to(device).eval()
    head0 = TargetPrototypeHead(weight, tuple(range(6)), scale=8.0).to(device).eval()
    head1 = TargetPrototypeHead(weight, tuple(range(6)), scale=8.0).to(device).eval()
    return da0, head0, da1, head1, {
        "capsule_id": expected_target_binding["capsule_id"],
        "split_id": expected_target_binding["split_id"],
        "support_count": 60,
        "class_ids": list(range(6)),
    }


def _case(tmp_path):
    labels = np.repeat(np.arange(6), 10)
    support_iq = np.zeros((60, 2, 256), dtype=np.float32)
    query_iq = np.zeros((60, 2, 256), dtype=np.float32)
    for class_id in range(6):
        support_iq[labels == class_id, 0, class_id] = 5.0
        query_iq[labels == class_id, 0, class_id] = 5.0
    support_ids = np.asarray([f"s{index}" for index in range(60)])
    query_ids = np.asarray([f"q{index}" for index in range(60)])
    np.savez(
        tmp_path / "support.npz",
        received_iq=support_iq,
        support_labels=labels,
        support_physical_ids=support_ids,
    )
    np.savez(tmp_path / "query.npz", received_iq=query_iq, query_ids=query_ids)
    np.savez(tmp_path / "truth.npz", query_ids=query_ids, query_labels=labels)
    handle = {
        "schema": "cvs.sf_erbt_oldonly_export.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "query-capsule",
        "split_id": "query-split",
        "adaptation_capsule_id": "adapt-capsule",
        "adaptation_split_id": "adapt-split",
        "k_shot": 10,
        "class_count": 6,
        "support_rows": 60,
        "query_rows": 60,
        "registration_state": "REG0",
        "query_truth_in_predictor": False,
        "support_query_physical_id_overlap": 0,
        "support_iq_sha256": hashlib.sha256(support_iq.tobytes(order="C")).hexdigest(),
    }
    (tmp_path / "handle.json").write_text(json.dumps(handle), encoding="utf-8")
    return labels


def test_prediction_emits_truth_free_da0_da1_artifacts(tmp_path):
    _case(tmp_path)

    receipt = run_clean_query_prediction(
        bundle_path=tmp_path / "bundle.pt",
        support_path=tmp_path / "support.npz",
        query_path=tmp_path / "query.npz",
        data_handle_path=tmp_path / "handle.json",
        output_root=tmp_path / "prediction",
        device="cpu",
        pair_loader=_pair_loader,
    )

    assert receipt["status"] == "PREDICTIONS_COMPLETE"
    assert receipt["states"] == ["DA0_REG0", "DA1_REG0"]
    assert receipt["query_truth_opened"] is False
    assert receipt["query_role_opened"] is False
    for name in ("da0_reg0.npz", "da1_reg0.npz"):
        with np.load(tmp_path / "prediction" / name, allow_pickle=False) as payload:
            assert set(payload.files) == {"query_ids", "predicted_class_ids", "scores"}
            assert payload["scores"].shape == (60, 6)


def test_pair_loader_accepts_historical_research_validation_schedule(
    tmp_path, monkeypatch
):
    config = asdict(SFTAPFTConfig(phase_steps=(1, 0, 0), validation_steps=()))
    config["validation_steps"] = (1, 2, 3)
    bundle = tmp_path / "bundle.pt"
    torch.save(
        {"config": config, "base_checkpoint_path": str(tmp_path / "base.pth")},
        bundle,
    )
    labels = torch.repeat_interleave(torch.arange(6), 10)
    support = SimpleNamespace(
        physical_ids=tuple(f"p{index}" for index in range(60)),
        class_ids=tuple(range(6)),
        received_iq=torch.zeros(60, 2, 256),
        labels=labels,
    )
    weight = torch.eye(6, 160)
    monkeypatch.setattr(query_module, "_load_target_support", lambda _path: support)
    monkeypatch.setattr(
        query_module,
        "load_sf_tapft_clean_single_bundle_strict",
        lambda *_args, **_kwargs: (
            _ParamModel(),
            TargetPrototypeHead(weight, tuple(range(6)), scale=4.0),
            {"capsule_id": "adapt", "split_id": "adapt-split"},
        ),
    )
    monkeypatch.setattr(
        query_module, "_default_checkpoint_loader", lambda *_args, **_kwargs: _ParamModel()
    )
    monkeypatch.setattr(query_module, "_source_classifier_weight", lambda _model: weight)
    monkeypatch.setattr(query_module, "ensure_time_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        query_module,
        "_forward_aux",
        lambda _model, rows: {"z_id": torch.zeros(len(rows), 160)},
    )
    monkeypatch.setattr(
        query_module,
        "_extract_joint_embedding",
        lambda outputs, _rows: outputs["z_id"],
    )
    monkeypatch.setattr(query_module, "_target_prototypes", lambda *_args: weight)

    _, da0_head, _, da1_head, audit = load_clean_single_pair_strict(
        bundle,
        tmp_path / "support.npz",
        device="cpu",
        expected_target_binding={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "adapt",
            "split_id": "adapt-split",
            "support_count": 60,
            "per_class_counts": [
                {"class_id": class_id, "count": 10} for class_id in range(6)
            ],
        },
    )

    assert da0_head.class_ids == tuple(range(6))
    assert da1_head.class_ids == tuple(range(6))
    assert audit["class_ids"] == list(range(6))


def test_prediction_rejects_truth_member_before_bundle_load(tmp_path):
    _case(tmp_path)
    with np.load(tmp_path / "query.npz", allow_pickle=False) as source:
        np.savez(
            tmp_path / "bad_query.npz",
            received_iq=source["received_iq"],
            query_ids=source["query_ids"],
            query_labels=np.zeros(60, dtype=np.int64),
        )

    with pytest.raises(QueryClosureError, match="query allowlist mismatch"):
        run_clean_query_prediction(
            bundle_path=tmp_path / "bundle.pt",
            support_path=tmp_path / "support.npz",
            query_path=tmp_path / "bad_query.npz",
            data_handle_path=tmp_path / "handle.json",
            output_root=tmp_path / "prediction",
            device="cpu",
            pair_loader=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("bundle loader must not run")
            ),
        )


def test_truth_last_score_reports_per_class_nll_and_da_delta(tmp_path):
    _case(tmp_path)
    run_clean_query_prediction(
        bundle_path=tmp_path / "bundle.pt",
        support_path=tmp_path / "support.npz",
        query_path=tmp_path / "query.npz",
        data_handle_path=tmp_path / "handle.json",
        output_root=tmp_path / "prediction",
        device="cpu",
        pair_loader=_pair_loader,
    )

    score = score_clean_query_prediction(
        prediction_root=tmp_path / "prediction",
        truth_path=tmp_path / "truth.npz",
        data_handle_path=tmp_path / "handle.json",
        output_path=tmp_path / "score.json",
    )

    assert score["status"] == "ANALYZED"
    assert score["DA1_REG0"]["balanced_accuracy"] == 1.0
    assert score["DA0_REG0"]["balanced_accuracy"] == 0.0
    assert score["da_effect"]["balanced_accuracy_pp"] == 100.0
    assert score["DA1_REG0"]["class_floor"] == 1.0
    assert len(score["DA1_REG0"]["per_class_accuracy"]) == 6
    assert score["DA1_REG0"]["nll"] < score["DA0_REG0"]["nll"]


def test_scorer_rejects_missing_prediction_id(tmp_path):
    _case(tmp_path)
    run_clean_query_prediction(
        bundle_path=tmp_path / "bundle.pt",
        support_path=tmp_path / "support.npz",
        query_path=tmp_path / "query.npz",
        data_handle_path=tmp_path / "handle.json",
        output_root=tmp_path / "prediction",
        device="cpu",
        pair_loader=_pair_loader,
    )
    path = tmp_path / "prediction" / "da1_reg0.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {name: payload[name][:-1] for name in payload.files}
    path.unlink()
    np.savez(path, **values)

    with pytest.raises(QueryClosureError, match="row count|query ID"):
        score_clean_query_prediction(
            prediction_root=tmp_path / "prediction",
            truth_path=tmp_path / "truth.npz",
            data_handle_path=tmp_path / "handle.json",
            output_path=tmp_path / "score.json",
        )
