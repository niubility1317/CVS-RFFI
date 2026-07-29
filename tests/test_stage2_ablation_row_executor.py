from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.stage2_ablation_row_executor as row_executor
from cvsrffi.stage2_ablation_row_executor import (
    Stage2AblationRowExecutionError,
    execute_feature_row,
)
from cvsrffi.stage2_prediction_artifact import verify_prediction_artifact
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


def _handle(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture(k_shot: int = 2):
    rng = np.random.default_rng(4401)
    old_classes = tuple(_handle("cls_", f"old-{index}") for index in range(6))
    new_classes = tuple(_handle("cls_", f"new-{index}") for index in range(5))
    means = rng.normal(size=(11, 288)).astype(np.float32)
    old_targets = np.repeat(np.arange(6), k_shot)
    new_targets = np.repeat(np.arange(5), k_shot)
    payloads = {}
    prototypes = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        query_tokens = np.asarray(
            [
                _handle("qid_", f"{scenario}-query-{index}")
                for index in range(8)
            ]
        )
        offset = np.float32(0.005 * scenario_index)
        old_rows = (
            means[old_targets]
            + offset
            + 0.02 * rng.normal(size=(6 * k_shot, 288))
        ).astype(np.float32)
        new_rows = (
            means[6 + new_targets]
            + offset
            + 0.02 * rng.normal(size=(5 * k_shot, 288))
        ).astype(np.float32)
        query = np.concatenate([old_rows[:4], new_rows[:4]])
        payloads[scenario] = {
            "old_support_features": old_rows,
            "old_support_labels": np.asarray(old_classes)[old_targets],
            "new_support_features": new_rows,
            "new_support_labels": np.asarray(new_classes)[new_targets],
            "query_features": query,
            "query_tokens": query_tokens,
        }
        labels = payloads[scenario]["old_support_labels"]
        prototypes[scenario] = np.stack(
            [old_rows[labels == value].mean(axis=0) for value in old_classes]
        )
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    return {
        "old_classes": old_classes,
        "new_classes": new_classes,
        "scenario_payloads": payloads,
        "deployment_prototypes_by_scenario": prototypes,
        "ground_basis": basis,
        "ground_spectral_weights": np.asarray([0.5, 0.3, 0.2]),
        "ground_audit": {
            "d81_basis_sha256": "a" * 64,
            "d81_spectral_weight_sha256": "b" * 64,
            "d81_participation_ratio_effective_rank": 2.6,
            "d81_retained_rank": 3,
            "d81_rank_policy": "ceil_participation_ratio_effective_rank",
            "ground_component_input_count": 84,
            "ground_statistic_semantics": (
                "class_centered_cross_domain_centroid_drift_eigenspectrum"
            ),
        },
    }


def _run(
    tmp_path: Path,
    *,
    ablation_id: str,
    device: str = "cpu",
):
    fixture = _fixture()
    if ablation_id == "P2-S2A":
        fixture["new_classes"] = ()
    stage = "stage2a" if ablation_id == "P2-S2A" else "stage2c"
    k_shot = 0 if stage == "stage2a" else 2
    return execute_feature_row(
        ablation_id=ablation_id,
        row_id="row_" + "c" * 64,
        receiver="20-1",
        candidate_lock_sha256="d" * 64,
        package_root_sha256="e" * 64,
        package_seal_sha256="f" * 64,
        input_identity={
            "stage_scope": stage,
            "k_shot": k_shot,
            "new_class_count": (
                0 if stage == "stage2a" else 5
            ),
            "method_seed": 840001,
            "support_seed": 0 if stage == "stage2a" else 840002,
            "query_seed": 840003,
            "new_class_draw_seed": (
                0 if stage == "stage2a" else 840004
            ),
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "capsule-test",
            "split_id": "split-test",
            "phase1_bundle_sha256": "1" * 64,
            "phase1_prototype_sha256": "2" * 64,
            "feature_cache_payload_sha256": "3" * 64,
            "feature_cache_manifest_sha256": "4" * 64,
        },
        output_root=tmp_path / ablation_id,
        seed=840001,
        device=device,
        feature_cache_bytes=4096,
        deployment_state_bytes=8192,
        peak_rss_bytes=8192,
        peak_vram_bytes=0,
        **fixture,
    )


def test_cuda_memory_audit_initializes_context_before_peak_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    device = row_executor.torch.device("cuda:0")
    monkeypatch.setattr(
        row_executor.torch.cuda,
        "set_device",
        lambda value: calls.append(("set_device", value)),
    )
    monkeypatch.setattr(
        row_executor.torch,
        "empty",
        lambda *args, **kwargs: calls.append(
            ("empty", kwargs.get("device"))
        ),
    )
    monkeypatch.setattr(
        row_executor.torch.cuda,
        "reset_peak_memory_stats",
        lambda value: calls.append(("reset", value)),
    )

    row_executor._prepare_cuda_memory_audit(device)

    assert calls == [
        ("set_device", device),
        ("empty", device),
        ("reset", device),
    ]


def test_cuda_initialization_failure_precedes_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "P2-S2A"
    monkeypatch.setattr(
        row_executor.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        row_executor,
        "_prepare_cuda_memory_audit",
        lambda _device: (_ for _ in ()).throw(
            RuntimeError("synthetic CUDA initialization failure")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic CUDA initialization failure",
    ):
        _run(
            tmp_path,
            ablation_id="P2-S2A",
            device="cuda:0",
        )
    assert not output.exists()


def test_feature_row_executor_has_no_truth_or_dataset_surface() -> None:
    parameters = inspect.signature(execute_feature_row).parameters
    assert not any(
        name in parameters
        for name in (
            "query_truth",
            "truth",
            "truth_sidecar",
            "dataset",
            "clean_samples",
            "source_samples",
        )
    )


def test_stage2a_publishes_zero_support_immutable_predictions(
    tmp_path: Path,
) -> None:
    receipt = _run(tmp_path, ablation_id="P2-S2A")
    assert receipt["stage"] == "stage2a"
    assert receipt["k_shot"] == 0
    assert receipt["fit_query_rows_used"] == 0
    assert receipt["query_truth_opened"] is False
    assert receipt["input_identity"]["query_seed"] == 840003
    verified = verify_prediction_artifact(
        receipt["prediction"]["path"],
        expected_artifact_sha256=receipt["prediction"]["artifact_sha256"],
        expected_seal_sha256=receipt["prediction"]["seal_sha256"],
    )
    assert verified["manifest"]["stage"] == "Stage2-A"
    assert verified["manifest"]["k_shot"] == 0


def test_stage2c_f2_publishes_complete_receipts_without_fp32_sidecar(
    tmp_path: Path,
) -> None:
    receipt = _run(tmp_path, ablation_id="P2-F2")
    assert receipt["stage"] == "stage2c"
    assert receipt["k_shot"] == 2
    assert receipt["prediction"]["row_count"] == 3 * 8
    assert receipt["quantization"]["max_logit_abs_error"] >= 0.0
    assert receipt["quantization"]["prediction_agreement_rate"] <= 1.0
    assert receipt["resource"]["state_bytes"] > 0
    with pytest.raises(
        Stage2AblationRowExecutionError,
        match="output root",
    ):
        _run(tmp_path, ablation_id="P2-F2")
