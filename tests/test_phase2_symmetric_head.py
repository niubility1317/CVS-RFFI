from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.phase2_symmetric_head import (
    fit_locked_symmetric_head,
    score_locked_symmetric_head,
    score_locked_symmetric_head_with_raw,
)
from paper_reproduction.cvs_aligned.k1_symmetric_head import (
    fit_locked_symmetric_support_head,
    quantize_symmetric_head_fp16,
    score_symmetric_head,
)


@pytest.mark.parametrize(
    "selected",
    [
        {
            "use_alignment": True,
            "prototype_rule": "mean",
            "ridge": 0.1,
            "gram_mix": 0.5,
            "uncertainty_penalty": 0.25,
        },
        {
            "use_alignment": False,
            "prototype_rule": "medoid",
            "ridge": None,
            "gram_mix": 0.0,
            "uncertainty_penalty": 0.0,
        },
    ],
)
def test_minimal_runtime_head_matches_locked_reference(selected: dict[str, object]) -> None:
    generator = np.random.default_rng(20260715)
    observations = generator.normal(size=(15, 7, 32)).astype(np.float32)
    source_mean = generator.normal(size=32).astype(np.float32)
    source_std = np.maximum(generator.random(size=32), 0.1).astype(np.float32)
    queries = generator.normal(size=(23, 32)).astype(np.float32)

    reference = quantize_symmetric_head_fp16(
        fit_locked_symmetric_support_head(
            observations,
            physical_shots_per_class=5,
            selected=selected,
            source_mean=source_mean,
            source_std=source_std,
        )
    )
    strict = fit_locked_symmetric_head(
        observations,
        physical_shots_per_class=5,
        selected=selected,
        source_mean=source_mean,
        source_std=source_std,
    )
    np.testing.assert_allclose(
        score_locked_symmetric_head(queries, strict),
        score_symmetric_head(queries, reference),
        rtol=0.0,
        atol=2.0e-6,
    )


def test_strict_head_rejects_non_three_scenario_support_shape() -> None:
    with pytest.raises(ValueError, match="three scenario views"):
        fit_locked_symmetric_head(
            np.zeros((4, 3, 8), dtype=np.float32),
            physical_shots_per_class=1,
            selected={
                "use_alignment": False,
                "prototype_rule": "mean",
                "ridge": None,
                "gram_mix": 0.0,
                "uncertainty_penalty": 0.0,
            },
            source_mean=np.zeros(8, dtype=np.float32),
            source_std=np.ones(8, dtype=np.float32),
        )


def _evidence_selected() -> dict[str, object]:
    return {
        "use_alignment": False,
        "prototype_rule": "mean",
        "ridge": None,
        "gram_mix": 0.0,
        "uncertainty_penalty": 0.0,
        "evidence_calibration": {
            "mode": "robust_lopo_class_symmetric",
            "negative_quantile": 0.95,
            "prior_physical_shots": 8.0,
            "scale_floor": 0.05,
            "inverse_scale_cap": 10.0,
        },
    }


@pytest.mark.parametrize(
    ("shots", "expected_fold"),
    ((1, "leave_one_view_out"), (2, "leave_one_physical_out")),
)
def test_evidence_head_uses_physical_or_view_folds_and_exact_state_bytes(
    shots: int, expected_fold: str
) -> None:
    generator = np.random.default_rng(20260716 + shots)
    observations = generator.normal(size=(3 * shots, 5, 12)).astype(np.float32)
    head = fit_locked_symmetric_head(
        observations,
        physical_shots_per_class=shots,
        selected=_evidence_selected(),
        source_mean=np.zeros(12, dtype=np.float32),
        source_std=np.ones(12, dtype=np.float32),
    )

    diagnostics = head["evidence_diagnostics"]
    assert diagnostics["adaptation_type"] == "EVAL_ONLY_CLOSED_FORM_ADAPTATION"
    assert diagnostics["fold_mode"] == expected_fold
    assert diagnostics["trainable_parameters"] == 0
    assert diagnostics["adapt_epochs"] == 0
    assert diagnostics["optimizer_steps"] == 0
    assert diagnostics["additional_backbone_forwards"] == 0
    assert diagnostics["state_bytes_fp16"] == 2 * 5 * 2
    assert head["evidence_bias"].shape == (5,)
    assert head["evidence_scale"].shape == (5,)
    assert np.all(head["evidence_scale"] > 0.0)


def test_evidence_head_is_class_permutation_equivariant() -> None:
    generator = np.random.default_rng(20260716)
    observations = generator.normal(size=(6, 4, 16)).astype(np.float32)
    queries = generator.normal(size=(17, 16)).astype(np.float32)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    kwargs = {
        "physical_shots_per_class": 2,
        "selected": _evidence_selected(),
        "source_mean": np.zeros(16, dtype=np.float32),
        "source_std": np.ones(16, dtype=np.float32),
    }
    original = fit_locked_symmetric_head(observations, **kwargs)
    permuted = fit_locked_symmetric_head(observations[:, permutation, :], **kwargs)

    np.testing.assert_allclose(
        score_locked_symmetric_head(queries, permuted),
        score_locked_symmetric_head(queries, original)[:, permutation],
        rtol=0.0,
        atol=2.0e-6,
    )


def test_evidence_head_rejects_interacting_gram_or_uncertainty_controls() -> None:
    selected = _evidence_selected()
    selected["ridge"] = 0.1
    selected["gram_mix"] = 0.5
    with pytest.raises(ValueError, match="requires no alignment"):
        fit_locked_symmetric_head(
            np.ones((3, 2, 8), dtype=np.float32),
            physical_shots_per_class=1,
            selected=selected,
            source_mean=np.zeros(8, dtype=np.float32),
            source_std=np.ones(8, dtype=np.float32),
        )


def test_evidence_head_k2_excludes_all_three_views_of_one_physical_shot() -> None:
    observations = np.zeros((6, 2, 2), dtype=np.float32)
    # Scenario-major layout: physical shot 0 is {0,2,4}; shot 1 is {1,3,5}.
    observations[[0, 2, 4], 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    observations[[1, 3, 5], 0] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations[:, 1] = np.asarray([-1.0, 0.0], dtype=np.float32)
    head = fit_locked_symmetric_head(
        observations,
        physical_shots_per_class=2,
        selected=_evidence_selected(),
        source_mean=np.zeros(2, dtype=np.float32),
        source_std=np.ones(2, dtype=np.float32),
    )

    diagnostics = head["evidence_diagnostics"]
    assert diagnostics["fold_mode"] == "leave_one_physical_out"
    assert diagnostics["raw_positive_by_class"][0] == pytest.approx(0.0, abs=1.0e-7)


def test_evidence_head_raw_gate_stream_matches_v1_and_inverse_scale_is_capped() -> None:
    observations = np.ones((3, 3, 4), dtype=np.float32)
    queries = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    common = {
        "physical_shots_per_class": 1,
        "source_mean": np.zeros(4, dtype=np.float32),
        "source_std": np.ones(4, dtype=np.float32),
    }
    v1_selected = {
        key: value
        for key, value in _evidence_selected().items()
        if key != "evidence_calibration"
    }
    v1 = fit_locked_symmetric_head(observations, selected=v1_selected, **common)
    v2 = fit_locked_symmetric_head(observations, selected=_evidence_selected(), **common)
    normalized, raw_gate = score_locked_symmetric_head_with_raw(queries, v2)

    np.testing.assert_allclose(
        raw_gate,
        score_locked_symmetric_head(queries, v1),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert np.max(1.0 / v2["evidence_scale"]) <= 10.0
    assert v2["evidence_diagnostics"]["quantized_inverse_scale_max"] <= 10.0
    assert np.isfinite(normalized).all()


def test_evidence_negative_q95_uses_higher_rule() -> None:
    observations = np.zeros((6, 2, 2), dtype=np.float32)
    observations[:, 0, 0] = 1.0
    cosine_values = np.asarray([-0.8, -0.6, -0.4, 0.2, 0.4, 0.9], dtype=np.float32)
    observations[:, 1, 0] = cosine_values
    observations[:, 1, 1] = np.sqrt(1.0 - cosine_values**2)
    head = fit_locked_symmetric_head(
        observations,
        physical_shots_per_class=2,
        selected=_evidence_selected(),
        source_mean=np.zeros(2, dtype=np.float32),
        source_std=np.ones(2, dtype=np.float32),
    )

    assert head["evidence_diagnostics"]["raw_negative_by_class"][0] == pytest.approx(
        0.9, abs=1.0e-6
    )


@pytest.mark.parametrize(
    "field",
    (
        "negative_quantile",
        "prior_physical_shots",
        "scale_floor",
        "inverse_scale_cap",
    ),
)
def test_evidence_head_rejects_boolean_numeric_controls(field: str) -> None:
    selected = _evidence_selected()
    selected["evidence_calibration"][field] = True
    with pytest.raises(ValueError, match="value is invalid"):
        fit_locked_symmetric_head(
            np.ones((3, 2, 4), dtype=np.float32),
            physical_shots_per_class=1,
            selected=selected,
            source_mean=np.zeros(4, dtype=np.float32),
            source_std=np.ones(4, dtype=np.float32),
        )


def test_evidence_head_rejects_nonfinite_support() -> None:
    observations = np.ones((3, 2, 4), dtype=np.float32)
    observations[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_locked_symmetric_head(
            observations,
            physical_shots_per_class=1,
            selected=_evidence_selected(),
            source_mean=np.zeros(4, dtype=np.float32),
            source_std=np.ones(4, dtype=np.float32),
        )
