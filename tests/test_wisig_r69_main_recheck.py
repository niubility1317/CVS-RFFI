import numpy as np
import pytest

from experiments.pamf_unrolled_restore.wisig_r69_main_recheck import matching_raw


def test_fresh_r69_comparison_preserves_archived_candidate_bypass_identity():
    raw = np.zeros((3, 7, 4), dtype=np.uint8)
    candidate_r69 = raw.copy()
    fresh_r69 = raw.copy()
    fresh_r69[2, 4, 1] = 1
    assert matching_raw({"raw": raw, "R69": raw.copy()},
                        {"raw": raw, "R69": fresh_r69}) == 1
    mismatched_raw = raw.copy()
    mismatched_raw[1, 0, 2] = 4
    with pytest.raises(ValueError, match="raw"):
        matching_raw({"raw": raw, "R69": candidate_r69},
                     {"raw": mismatched_raw, "R69": fresh_r69})
