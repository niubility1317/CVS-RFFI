from __future__ import annotations

from scripts.run_d112_seam_g0_one_shot import _decision


def _row(
    k: int,
    *,
    rho: int = 0,
    anchor: int = 0,
    score: int = 0,
    margin: int = 0,
    functional: bool = False,
) -> dict[str, object]:
    return {
        "K": k,
        "positive_rho_count": rho,
        "anchor_changed_count": anchor,
        "score_changed_count": score,
        "margin_changed_count": margin,
        "functional_nonzero": functional,
    }


def test_g0_decision_rejects_only_complete_structural_zero() -> None:
    result = _decision([_row(1), _row(5), _row(10)])
    assert result == {
        "functional_nonzero_k_values": [],
        "all_k_structurally_zero": True,
        "functional_status": "REJECT_NO_FUNCTION_STRUCTURAL_ALL_K_ZERO",
        "g1_entry_allowed": False,
    }


def test_g0_decision_requires_rho_anchor_and_score_or_margin() -> None:
    result = _decision(
        [
            _row(1, rho=1, anchor=2, score=3, functional=True),
            _row(5, rho=1, anchor=2),
            _row(10),
        ]
    )
    assert result["functional_nonzero_k_values"] == [1]
    assert result["functional_status"] == "G0_FUNCTION_PRESENT_PROCEED_SINGLE_G1"
    assert result["g1_entry_allowed"] is True


def test_partial_mechanical_change_is_not_promoted() -> None:
    result = _decision([_row(1, rho=1), _row(5), _row(10)])
    assert result["all_k_structurally_zero"] is False
    assert result["functional_status"] == "G0_INCONCLUSIVE_PARTIAL_FUNCTION_REVIEW_IMPLEMENTATION"
    assert result["g1_entry_allowed"] is False
