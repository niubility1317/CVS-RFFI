from __future__ import annotations

import pytest

from scripts import summarize_m24_d1_refit_full125 as summary


def test_matrix_without_compile_parity_arm_does_not_require_parity_payload() -> None:
    parity_for_identity = getattr(summary, "_parity_for_identity", None)
    assert callable(parity_for_identity), "full-125 summaries need an explicit optional-parity contract"

    assert parity_for_identity(
        {"arm": "M24-G1-FROZEN-BALANCED-PROTOTYPE"},
        parity_arm=None,
    ) is None


def test_compile_parity_arm_still_requires_its_parity_payload() -> None:
    parity_for_identity = getattr(summary, "_parity_for_identity", None)
    assert callable(parity_for_identity), "full-125 summaries need an explicit optional-parity contract"

    with pytest.raises(ValueError, match="missing d1_historical_parity"):
        parity_for_identity(
            {"arm": "M24-D1-COMPILE-PARITY"},
            parity_arm="M24-D1-COMPILE-PARITY",
        )
