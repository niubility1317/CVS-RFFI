import numpy as np

from scripts.phase2_open_set_geometry_audit import (
    _auc_unknown_higher,
    _fpr_at_tpr,
    _oracle_at_far,
    _threshold_eval,
)


def test_auc_unknown_higher_handles_ties_and_ordering():
    known = [0.1, 0.2, 0.3]
    unknown = [0.2, 0.4]

    # Unknown 0.2 ties one known, beats one known, loses to one known.
    # Unknown 0.4 beats all known: (1.5 + 3) / 6 = 0.75.
    assert _auc_unknown_higher(known, unknown) == 0.75


def test_fpr_at_tpr_reports_known_false_positive_rate():
    known = [0.1, 0.2, 0.3, 0.4]
    unknown = [0.35, 0.45]

    assert _fpr_at_tpr(known, unknown, 1.0) == 0.25


def test_threshold_eval_uses_high_score_as_unknown_reject():
    out = _threshold_eval(
        threshold=0.5,
        known_scores=np.asarray([0.1, 0.6]),
        old_scores=np.asarray([0.2, 0.7]),
        seen_scores=np.asarray([0.3, 0.8]),
        unknown_scores=np.asarray([0.4, 0.9]),
    )

    assert out["known_reject_rate"] == 0.5
    assert out["old_reject_rate"] == 0.5
    assert out["seen_new_reject_rate"] == 0.5
    assert out["unknown_reject_rate"] == 0.5
    assert out["unknown_FAR"] == 0.5


def test_oracle_at_far_is_marked_diagnostic_only():
    out = _oracle_at_far(
        known_scores=np.asarray([0.1, 0.2, 0.3]),
        old_scores=np.asarray([0.1, 0.2]),
        seen_scores=np.asarray([0.2, 0.3]),
        unknown_scores=np.asarray([0.9, 1.0]),
        far_limit=0.05,
    )

    assert out is not None
    assert out["unknown_FAR"] == 0.0
    assert out["uses_target_unknown_labels"] is True
    assert out["diagnostic_only"] is True
