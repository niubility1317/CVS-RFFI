from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.phase2_symmetric_head import (
    fit_locked_symmetric_head,
    score_locked_symmetric_head,
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
