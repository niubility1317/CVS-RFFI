from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from cvsrffi.stage2_d42_unified_shrinkage_lda import (
    fit_d42_old_only,
    fit_d42_unified_shrinkage_lda,
    score_d42_unified_shrinkage_lda,
)
from cvsrffi.stage2_d92_da0_phase1_old_state import (
    JOINT288_TRANSFORM_SCHEMA,
    build_source_only_joint288,
    fit_source_only_old_state,
    load_sealed_source_only_old_state,
    seal_source_only_old_state,
)


def _hash(character: str) -> str:
    return character * 64


def _normalized_rows(
    classes: tuple[str, ...], *, k_shot: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(classes):
        center = rng.normal(size=288).astype(np.float32)
        center[(37 * class_index) % 288] += np.float32(6.0)
        for _ in range(k_shot):
            row = center + np.float32(0.20) * rng.normal(size=288).astype(np.float32)
            rows.append((row / np.linalg.norm(row)).astype(np.float32))
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels, dtype=str)


def _state_bytes(state) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(getattr(state, name)).tobytes()
        for name in (
            "log_diag_fp32",
            "coef1_qint8",
            "coef2_qint8",
            "scale1_fp16",
            "scale2_fp16",
            "intercept_fp16",
            "coef_fp32",
            "intercept_fp32",
        )
    )


class _FrozenRuntime(torch.nn.Module):
    def forward(self, iq: torch.Tensor):
        flattened = iq.flatten(start_dim=1)
        repeats = (160 + int(flattened.shape[1]) - 1) // int(flattened.shape[1])
        features = flattened.repeat(1, repeats)[:, :160].contiguous().float()
        return features, torch.zeros((len(iq), 2), dtype=torch.float32, device=iq.device)


def test_public_old_only_fit_is_bitwise_equal_to_full_before_state() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    old_features, old_labels = _normalized_rows(old_classes, k_shot=5, seed=11)
    new_features, new_labels = _normalized_rows(new_classes, k_shot=5, seed=12)

    assert "new" not in inspect.signature(fit_d42_old_only).parameters
    old_only = fit_d42_old_only(
        old_features, old_labels, old_classes, seed=29, device="cpu"
    )
    full = fit_d42_unified_shrinkage_lda(
        old_features,
        old_labels,
        old_classes,
        new_features,
        new_labels,
        new_classes,
        seed=29,
        device="cpu",
    )

    assert _state_bytes(old_only.state) == _state_bytes(full.before_state)
    assert _state_bytes(old_only.matched_fp32_state) == _state_bytes(
        full.matched_fp32_before_state
    )
    assert old_only.training_trace == full.training_trace
    assert old_only.old_state_sha256 == old_only.state_sha256


def test_source_only_iq_state_is_sealed_and_scores_all_old_classes(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(23)
    iq = rng.normal(size=(6, 2, 64)).astype(np.float32)
    labels = np.asarray(("old_a", "old_a", "old_a", "old_b", "old_b", "old_b"))
    old_classes = ("old_a", "old_b")
    runtime = _FrozenRuntime().eval()

    features = build_source_only_joint288(
        runtime, iq, device=torch.device("cpu"), batch_size=2
    )
    old_fit = fit_source_only_old_state(
        features, labels, old_classes, seed=31, device="cpu"
    )
    source_cache_sha = {
        "leo_clear_weak": _hash("a"),
        "leo_low_elev_weak": _hash("b"),
        "leo_rain_weak": _hash("c"),
    }
    receipt = seal_source_only_old_state(
        tmp_path / "sealed",
        states_by_scenario={scenario: old_fit for scenario in source_cache_sha},
        provenance={
            "checkpoint_sha256": _hash("d"),
            "runtime_sha256": _hash("e"),
            "source_cache_set_manifest_sha256": _hash("f"),
            "source_dataset_sha256": _hash("0"),
            "class_handle_binding_sha256": _hash("1"),
            "source_cache_member_sha256_by_scenario": source_cache_sha,
            "source_cache_physical_id_root_by_scenario": {
                "leo_clear_weak": _hash("2"),
                "leo_low_elev_weak": _hash("3"),
                "leo_rain_weak": _hash("4"),
            },
        },
    )

    manifest = json.loads((tmp_path / "sealed" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "cvs.phase1.d92_da0_old_state.v1"
    assert manifest["joint288_transform_schema"] == JOINT288_TRANSFORM_SCHEMA
    assert manifest["checkpoint_sha256"] == _hash("d")
    assert manifest["source_dataset_sha256"] == _hash("0")
    assert manifest["old_registry_binding_sha256"] == _hash("1")
    assert manifest["target_receiver_old_support_opened"] is False
    assert manifest["target_receiver_new_support_opened"] is False
    assert manifest["target_query_opened"] is False
    assert manifest["target_package_somph_head_opened"] is False
    assert manifest["query_truth_opened"] is False
    assert receipt["content_root_sha256"] == manifest["content_root_sha256"]

    loaded = load_sealed_source_only_old_state(tmp_path / "sealed")
    state = loaded["states_by_scenario"]["leo_clear_weak"]
    assert state.classes == old_classes
    assert state.old_class_count == len(old_classes)
    assert _state_bytes(state) == _state_bytes(old_fit.state)
    assert score_d42_unified_shrinkage_lda(state, features).shape == (len(features), 2)
    with np.load(tmp_path / "sealed" / "states" / "leo_clear_weak.npz", allow_pickle=False) as archive:
        assert set(archive.files).isdisjoint({"source_iq", "features", "labels"})
