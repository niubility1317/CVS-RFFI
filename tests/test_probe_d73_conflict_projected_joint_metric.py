from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d73_conflict_projected_joint_metric.py"
SPEC = importlib.util.spec_from_file_location("probe_d73_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_build_wraps_d62_fit_and_adds_only_d73_audit(monkeypatch) -> None:
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
    fit, records = probe.build_d73_fit(object())
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
    assert audit["d73_probe_arm"] == probe.ARM
    assert audit["d73_formula"] == probe.FORMULA
    assert records is component_records


def test_added_refit_resource_counts_one_complete_d62_fit(monkeypatch) -> None:
    fake = SimpleNamespace(
        FEATURE_DIM=288,
        _lda_fit_macs=lambda rows, classes: rows * 1000 + classes,
    )
    monkeypatch.setattr(
        probe.d62.d61,
        "_fisher_dense_macs",
        lambda dimension, fits: dimension * fits,
    )
    evidence = probe._added_refit_resource(fake, 8, 11)
    assert evidence["component_fit_count"] == 36
    assert evidence["lda_fit_macs"] > 0
    assert evidence["fisher_dense_macs"] == 288 * 18
    assert evidence["gate_scalar_macs"] == 8 * 11 * 11 * 8


def test_k1_refit_resource_is_zero() -> None:
    fake = SimpleNamespace(FEATURE_DIM=288)
    assert probe._added_refit_resource(fake, 1, 11) == {
        "component_fit_count": 0,
        "lda_fit_macs": 0,
        "fisher_dense_macs": 0,
        "gate_scalar_macs": 0,
    }


def test_gradient_resource_formula_is_positive_and_exact() -> None:
    assert probe._gradient_mac_upper_bound(288, 11, 8) == (
        8 * 88 * 11 * 288 + 6 * 88 * 11 + 16 * 288
    )


def test_probe_source_closes_protocol_resource_and_call_counts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "registry.top_fit_count != 30" in source
    assert "registry.extra_d62_fit_count != 30" in source
    assert "len(component_records) != 1620" in source
    assert '"d73_ground_component_input_count": 0' in source
    assert '"d73_query_extra_mac_equivalents": 0' in source
    assert '"d73_single_affine_state_only": True' in source
    assert '"d73_query_role_specific_branch": False' in source
    assert '"d73_uses_outer_held_or_query_for_fit": False' in source
    assert "training_trace=tuple(trace)" in source
    increment_block = source[source.index('for field in (\n                "adaptation_epochs"'):]
    increment_block = increment_block[: increment_block.index("resource[\"adaptation_epoch_cap_pass\"]")]
    assert '"total_optimizer_steps"' not in increment_block
