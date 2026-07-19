from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "code"
    / "scripts"
    / "probe_d72_physical_rank_leave_one_head_bagging.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d72_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_build_wraps_d62_fit_and_changes_only_probe_audit(monkeypatch) -> None:
    component_records = []

    def build(_d42):
        def fit(rows, labels, class_count, k_shot):
            del labels, k_shot
            return (
                np.zeros((class_count, rows.shape[1]), dtype=np.float32),
                np.zeros(class_count, dtype=np.float32),
                {"d62_probe_arm": probe.d62.ARM},
            )

        return fit, component_records

    monkeypatch.setattr(probe.d62, "build_d62_fit", build)
    fit, records = probe.build_d72_fit(object())
    coefficient, intercept, audit = fit(
        np.zeros((8, 5), dtype=np.float32),
        np.repeat(np.arange(2), 4),
        2,
        4,
    )
    assert coefficient.shape == (2, 5)
    assert intercept.shape == (2,)
    assert audit["d62_probe_arm"] == probe.d62.ARM
    assert audit["d43_probe_arm"] == probe.ARM
    assert audit["d72_probe_arm"] == probe.ARM
    assert audit["d72_formula"] == probe.FORMULA
    assert records is component_records


def test_added_resource_counts_all_leave_one_d62_components(monkeypatch) -> None:
    fake = SimpleNamespace(
        FEATURE_DIM=288,
        _lda_fit_macs=lambda rows, classes: rows * 1000 + classes,
    )
    monkeypatch.setattr(
        probe.d62.d61,
        "_fisher_dense_macs",
        lambda dimension, fits: dimension * fits,
    )
    evidence = probe._stage_added_resource(fake, 8, 11)
    assert evidence["d62_fit_count"] == 8
    assert evidence["component_fit_count"] == 256
    assert evidence["lda_fit_macs"] > 0
    assert evidence["fisher_dense_macs"] == 288 * 128
    assert evidence["gate_scalar_macs"] == 8 * 7 * 11 * 11 * 8
    assert evidence["mean_scalar_macs"] == 8 * 11 * 289


def test_k1_resource_is_exact_zero_fallback() -> None:
    fake = SimpleNamespace(FEATURE_DIM=288)
    assert probe._stage_added_resource(fake, 1, 6) == {
        "d62_fit_count": 0,
        "component_fit_count": 0,
        "lda_fit_macs": 0,
        "fisher_dense_macs": 0,
        "gate_scalar_macs": 0,
        "mean_scalar_macs": 0,
    }


def test_probe_source_closes_protocol_resource_and_call_counts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "registry.top_fit_count != 30" in source
    assert "registry.inner_base_fit_count != 480" in source
    assert "len(component_records) != 8760" in source
    assert '"d72_ground_component_input_count": 0' in source
    assert '"d72_query_extra_mac_equivalents": 0' in source
    assert '"d72_single_affine_state_only": True' in source


def test_probe_has_no_role_scene_class_or_query_fit_branch() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert '"d72_class_id_specific_formula": false' in source
    assert '"d72_old_new_role_specific_branch": false' in source
    assert '"d72_scene_receiver_handle_specific_branch": false' in source
    assert '"d72_uses_outer_held_or_query_for_fit": false' in source
    assert "temperature" not in source
    assert "bootstrap_sample_indices" not in source
