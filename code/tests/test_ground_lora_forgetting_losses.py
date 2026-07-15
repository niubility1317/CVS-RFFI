from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_apply_phase1_iq_preadapter_20260703 import (
    FORMAL_LEO_WEAK_SCENARIOS,
    PHASE2_SAMPLE_VIEW_POLICY,
    SealedLeoWeakSourceDataset,
    _lazy_formal_training_view_pair,
    _make_source_loader,
    _validate_source_only_ground_lora_mode,
    nested_k_worst_prototype_risk,
    prototype_gram_deconfusion_loss,
    relation_gram_preservation_loss,
)


def test_formal_training_receive_views_are_lazily_materialized(monkeypatch) -> None:
    rows = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)

    def forbidden_cfo(*_args, **_kwargs):
        raise AssertionError("shift-only step must not materialize CFO views")

    monkeypatch.setattr(
        "train_apply_phase1_iq_preadapter_20260703._satellite_tta_views",
        forbidden_cfo,
    )
    views = _lazy_formal_training_view_pair(rows, 1)
    assert [name for name, _value in views] == ["rx_base", "rx_shift_m2"]
    assert torch.equal(views[1][1], torch.roll(rows, shifts=-2, dims=-1))


def test_effective_ground_lora_must_stop_before_target_export(tmp_path) -> None:
    cache_set = tmp_path / "source_cache_set.json"
    cache_set.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="requires --source_only_ground_lora"):
        _validate_source_only_ground_lora_mode(
            argparse.Namespace(
                model_adapter_mode="lora_effective_feature",
                source_only_ground_lora=False,
                source_leo_weak_cache_set_manifest=cache_set,
            )
        )
    _validate_source_only_ground_lora_mode(
        argparse.Namespace(
            model_adapter_mode="lora_effective_feature",
            source_only_ground_lora=True,
            source_leo_weak_cache_set_manifest=cache_set,
            leo_reference_identity_weight=22.0,
            leo_reference_cos_weight=1.0,
            leo_reference_margin_weight=7.5,
        )
    )


def test_effective_ground_lora_requires_sealed_source_cache(tmp_path) -> None:
    with pytest.raises(ValueError, match="source_leo_weak_cache_set_manifest"):
        _validate_source_only_ground_lora_mode(
            argparse.Namespace(
                model_adapter_mode="lora_effective_feature",
                source_only_ground_lora=True,
                source_leo_weak_cache_set_manifest=tmp_path / "missing.json",
            )
        )


def test_formal_source_loader_consumes_only_verified_postchannel_arrays(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays_by_scenario = {}
    for offset, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        arrays_by_scenario[scenario] = {
            "leo_weak_iq": np.full((2, 2, 8), offset + 1, dtype=np.float32),
            "tx_ids": np.asarray(["a", "b"]),
            "rx_ids": np.asarray(["r", "r"]),
            "sat_scenarios": np.asarray([scenario, scenario]),
            "sample_ids": np.asarray(["source|a|r|d|1|0", "source|b|r|d|1|1"]),
        }
    observed = {}

    def fake_loader(path, *, expected_scope, allowed_roles):
        observed.update(
            path=str(path),
            expected_scope=expected_scope,
            allowed_roles=set(allowed_roles),
        )
        return (
            arrays_by_scenario,
            {"cache_scope": "source_train"},
            {"sha256": "a" * 64, "cache_audits": {}},
        )

    monkeypatch.setattr(
        "train_apply_phase1_iq_preadapter_20260703.load_verified_leo_weak_cache_set",
        fake_loader,
    )
    manifest = tmp_path / "source_set.json"
    manifest.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        model_adapter_mode="lora_effective_feature",
        source_leo_weak_cache_set_manifest=manifest,
        source_tx_ids="a,b",
        num_old_classes=2,
        source_rxs="r",
        wisig_out_len=8,
        batch_size=3,
    )
    loader, dataset, info = _make_source_loader(args)
    assert isinstance(dataset, SealedLeoWeakSourceDataset)
    assert len(dataset) == 6
    assert observed == {
        "path": str(manifest),
        "expected_scope": "source_train",
        "allowed_roles": {"source"},
    }
    batch = next(iter(loader))
    assert batch[0].shape[1:] == (2, 8)
    assert set(batch[3]["phase2_sample_view_policy"]) == {
        PHASE2_SAMPLE_VIEW_POLICY
    }
    assert info["clean_sample_access"] is False


def test_sealed_source_dataset_supports_numpy2_torch21_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays_by_scenario = {}
    for offset, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        arrays_by_scenario[scenario] = {
            "leo_weak_iq": np.full((1, 2, 8), offset + 1, dtype=np.float32),
            "tx_ids": np.asarray(["a"]),
            "rx_ids": np.asarray(["r"]),
            "sat_scenarios": np.asarray([scenario]),
            "sample_ids": np.asarray([f"source|a|r|d|1|{offset}"]),
        }
    dataset = SealedLeoWeakSourceDataset(arrays_by_scenario, tx_labels=("a",))

    def incompatible_from_numpy(_value):
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    monkeypatch.setattr(torch, "from_numpy", incompatible_from_numpy)
    iq, class_id, domain_id, _meta = dataset[0]
    assert iq.dtype == torch.float32
    assert iq.shape == (2, 8)
    assert torch.equal(iq, torch.ones((2, 8), dtype=torch.float32))
    assert class_id.item() == 0
    assert domain_id.item() == 0


def test_source_only_switch_cannot_hide_a_nonformal_adapter_export() -> None:
    with pytest.raises(ValueError, match="reserved for lora_effective_feature"):
        _validate_source_only_ground_lora_mode(
            argparse.Namespace(
                model_adapter_mode="lora_full_feature",
                source_only_ground_lora=True,
            )
        )


def test_non_source_only_path_still_requires_target_cells() -> None:
    with pytest.raises(ValueError, match="nonempty --cells"):
        _validate_source_only_ground_lora_mode(
            argparse.Namespace(
                model_adapter_mode="none",
                source_only_ground_lora=False,
                cells="",
            )
        )


def test_relation_loss_is_zero_for_identical_geometry_and_positive_after_drift() -> None:
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    same = relation_gram_preservation_loss(reference.clone(), reference)
    drifted = reference.clone()
    drifted[1] = torch.tensor([1.0, 1.0])
    changed = relation_gram_preservation_loss(drifted, reference)
    assert float(same) == 0.0
    assert float(changed) > 0.0


def test_prototype_gram_loss_targets_only_crowded_classes() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    crowded = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.95, 0.05], [0.95, 0.05]]
    )
    assert float(prototype_gram_deconfusion_loss(separated, labels)) == 0.0
    assert float(prototype_gram_deconfusion_loss(crowded, labels)) > 0.0


def test_nested_k_risk_is_query_free_differentiable_and_tracks_k() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, -0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-0.1, 1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    risk, by_k = nested_k_worst_prototype_risk(
        features, labels, k_values=(1, 2, 5), risk_tau=0.2
    )
    assert set(by_k) == {1, 2}
    assert torch.isfinite(risk)
    risk.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
