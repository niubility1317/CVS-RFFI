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
    / "probe_d71_crossfitted_top2_centroid_reranker.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d71_test", SCRIPT)
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
    fit, records = probe.build_d71_fit(object())
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
    assert audit["d71_probe_arm"] == probe.ARM
    assert audit["d71_formula"] == probe.FORMULA
    assert records is component_records


def test_registry_score_only_reorders_registered_top2() -> None:
    fake_d42 = SimpleNamespace(
        _transform=lambda features, _log_diag: np.asarray(features, dtype=np.float32)
    )
    base = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    registry = probe.RerankerRegistry(
        fake_d42,
        base_fit=None,
        original_score=lambda _state, _features: base,
    )
    state = SimpleNamespace(log_diag_fp32=np.zeros(2, dtype=np.float32))
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    directions = np.asarray([[1.0, -1.0]], dtype=np.float32)
    biases = np.asarray([0.0], dtype=np.float32)
    int8, _, _ = probe.core.compile_pair_states(
        pairs, directions, biases, np.asarray([True])
    )
    registry.states[id(state)] = (state, int8)
    scores = registry.score(state, np.asarray([[0.0, 1.0]], dtype=np.float32))
    assert scores.tolist() == [[2.0, 3.0, 1.0]]
    assert registry.reranked_prediction_count == 1
    assert scores.flags.writeable is False


def test_probe_source_closes_protocol_resource_and_call_counts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "registry.top_fit_count != 30" in source
    assert "registry.inner_base_fit_count != 120" in source
    assert "len(component_records) != 2280" in source
    assert '"d71_ground_component_input_count": 0' in source
    assert '"d71_dense_query_graph_bytes": 0' in source
    assert '"d71_top2_only": True' in source
    assert '"d71_single_affine_state_only": False' in source


def test_probe_has_no_role_scene_class_or_query_fit_branch() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert '"d71_class_id_specific_formula": false' in source
    assert '"d71_old_new_role_specific_branch": false' in source
    assert '"d71_scene_receiver_handle_specific_branch": false' in source
    assert '"d71_uses_outer_held_or_query_for_fit": false' in source
    assert "threshold" not in source
    assert "temperature" not in source
