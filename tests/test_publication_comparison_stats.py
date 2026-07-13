from __future__ import annotations

import math

import pytest

from tools.analyze_publication_comparison import analyze_rows, holm_adjust, paired_sign_flip_pvalue


def _rows():
    rows = []
    for seed, cvs, baseline in [(1, 80.0, 70.0), (2, 82.0, 72.0), (3, 84.0, 74.0), (4, 86.0, 76.0), (5, 88.0, 78.0)]:
        rows.append({"method": "CVS", "seed": seed, "metric": "strict_udu", "value": cvs})
        rows.append({"method": "CVCNN-CE", "seed": seed, "metric": "strict_udu", "value": baseline})
    return rows


def test_analyze_rows_reports_paired_delta_ci_effect_and_holm_p() -> None:
    result = analyze_rows(_rows(), reference_method="CVS", bootstrap_samples=1000)
    assert result["schema"] == "cvs_publication_comparison_stats_v1"
    assert len(result["summary"]) == 2
    comparison = result["paired_comparisons"][0]
    assert comparison["paired_n"] == 5
    assert comparison["mean_delta_reference_minus_comparison"] == pytest.approx(10.0)
    assert comparison["delta_ci95_low"] == pytest.approx(10.0)
    assert comparison["delta_ci95_high"] == pytest.approx(10.0)
    assert math.isinf(comparison["paired_effect_size_dz"])
    assert comparison["holm_p_adjusted"] == comparison["sign_flip_p_raw"]


def test_sign_flip_exact_pvalue_uses_all_sign_patterns() -> None:
    assert paired_sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(0.25)


def test_holm_adjust_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.06), "c": pytest.approx(0.2)}


def test_duplicate_seed_is_rejected() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_rows(rows, reference_method="CVS", bootstrap_samples=100)
