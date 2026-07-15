from __future__ import annotations

import numpy as np
import pytest

from paper_reproduction.cvs_aligned.extreme_light_adapter import (
    concatenate_registered_features,
    normalized_auxiliary_energy_share,
)


def test_fft_weight_maps_to_expected_normalized_block_energy() -> None:
    assert normalized_auxiliary_energy_share(2.0) == pytest.approx(0.8)
    assert normalized_auxiliary_energy_share(1.0) == pytest.approx(0.5)
    assert normalized_auxiliary_energy_share(0.7) == pytest.approx(
        0.49 / 1.49
    )


def test_joint_feature_block_norms_match_reported_energy_share() -> None:
    primary = np.asarray([[3.0, 4.0]], dtype=np.float32)
    fft = np.asarray([[5.0, 12.0]], dtype=np.float32)
    joint = concatenate_registered_features(
        primary, fft, auxiliary_weight=0.7
    )
    observed = float(np.sum(joint[:, 2:] ** 2) / np.sum(joint**2))
    assert observed == pytest.approx(normalized_auxiliary_energy_share(0.7))
