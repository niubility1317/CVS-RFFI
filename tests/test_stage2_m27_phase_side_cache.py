from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_m27_phase_side_cache import (
    M27PhaseSideCacheError,
    load_phase_side_cache,
    phase_coherence32,
    publish_phase_side_cache,
)


def _iq_rows() -> np.ndarray:
    rng = np.random.default_rng(82710)
    time = np.arange(256, dtype=np.float64)
    rows = []
    for index in range(5):
        phase = 0.03 * time + 0.0002 * (index + 1) * time**2
        value = (1.0 + 0.2 * np.cos(0.07 * time + index)) * np.exp(1j * phase)
        value += 0.01 * (rng.normal(size=len(time)) + 1j * rng.normal(size=len(time)))
        rows.append(np.stack([value.real, value.imag]))
    return np.asarray(rows, dtype=np.float32)


def test_phase32_is_finite_deterministic_and_invariant_to_gain_and_global_phase() -> None:
    iq = _iq_rows()
    first = phase_coherence32(iq)
    repeated = phase_coherence32(iq.copy())
    rotated_complex = (iq[:, 0] + 1j * iq[:, 1]) * (3.2 * np.exp(1j * 0.73))
    rotated = np.stack([rotated_complex.real, rotated_complex.imag], axis=1)
    transformed = phase_coherence32(rotated)
    assert first.shape == (5, 32)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(repeated, first)
    np.testing.assert_allclose(transformed, first, atol=2.0e-5)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1.0e-6)


def _scenario_payloads() -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(82711)
    old = np.asarray(["cls_" + "a" * 32, "cls_" + "b" * 32])
    new = np.asarray(["cls_" + "c" * 32])
    result = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        query_count = 4
        result[scenario] = {
            "old_support_phase32": rng.normal(size=(4, 32)).astype(np.float32),
            "old_support_labels": np.repeat(old, 2),
            "new_support_phase32": rng.normal(size=(2, 32)).astype(np.float32),
            "new_support_labels": np.repeat(new, 2),
            "query_phase32": rng.normal(size=(query_count, 32)).astype(np.float32),
            "query_tokens": np.asarray(
                [f"qid_{scenario_index:02x}{item:02x}" + "d" * 60 for item in range(query_count)]
            ),
        }
    return result


def test_phase_side_cache_is_truth_free_immutable_and_bound_to_base_cache(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "phase32.npz"
    manifest_path = tmp_path / "phase32.manifest.json"
    scenario_payloads = _scenario_payloads()
    receipt = publish_phase_side_cache(
        payload_path,
        manifest_path,
        base_manifest_sha256="a" * 64,
        capsule_id="capsule-test",
        split_id="split-test",
        receiver="3-19",
        method_seed=7282101,
        k_shot=2,
        old_classes=("cls_" + "a" * 32, "cls_" + "b" * 32),
        new_classes=("cls_" + "c" * 32,),
        scenario_payloads=scenario_payloads,
    )
    loaded = load_phase_side_cache(
        payload_path,
        manifest_path,
        expected_payload_sha256=receipt["payload_sha256"],
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_base_manifest_sha256="a" * 64,
        expected_capsule_id="capsule-test",
        expected_split_id="split-test",
        expected_query_tokens_by_scenario={
            scenario: payload["query_tokens"]
            for scenario, payload in scenario_payloads.items()
        },
    )
    assert loaded["manifest"]["query_truth_present"] is False
    assert loaded["manifest"]["query_role_present"] is False
    assert loaded["manifest"]["clean_source_samples_present"] is False
    assert loaded["scenario_payloads"]["leo_rain_weak"]["query_phase32"].shape == (4, 32)

    with pytest.raises(M27PhaseSideCacheError, match="base feature-cache"):
        load_phase_side_cache(
            payload_path,
            manifest_path,
            expected_payload_sha256=receipt["payload_sha256"],
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_base_manifest_sha256="b" * 64,
            expected_capsule_id="capsule-test",
            expected_split_id="split-test",
            expected_query_tokens_by_scenario={
                scenario: payload["query_tokens"]
                for scenario, payload in scenario_payloads.items()
            },
        )


def test_phase_side_cache_rejects_query_token_drift(tmp_path: Path) -> None:
    payload_path = tmp_path / "phase32.npz"
    manifest_path = tmp_path / "phase32.manifest.json"
    scenario_payloads = _scenario_payloads()
    receipt = publish_phase_side_cache(
        payload_path,
        manifest_path,
        base_manifest_sha256="a" * 64,
        capsule_id="capsule-test",
        split_id="split-test",
        receiver="8-8",
        method_seed=7282101,
        k_shot=2,
        old_classes=("cls_" + "a" * 32, "cls_" + "b" * 32),
        new_classes=("cls_" + "c" * 32,),
        scenario_payloads=scenario_payloads,
    )
    expected = {
        scenario: payload["query_tokens"].copy()
        for scenario, payload in scenario_payloads.items()
    }
    expected["leo_clear_weak"][0] = "qid_" + "e" * 64
    with pytest.raises(M27PhaseSideCacheError, match="query-token"):
        load_phase_side_cache(
            payload_path,
            manifest_path,
            expected_payload_sha256=receipt["payload_sha256"],
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_base_manifest_sha256="a" * 64,
            expected_capsule_id="capsule-test",
            expected_split_id="split-test",
            expected_query_tokens_by_scenario=expected,
        )
