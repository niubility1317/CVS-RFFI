import numpy as np
import pytest

from scripts.phase2_aware_ci_eval import _profile_names, _query_indices, _target_pass, _topk_mean
from scripts.phase2_proxy_adapter_ci_eval import AdapterTrainingPlan


def test_aware_profile_parser_rejects_unknown_profile():
    assert _profile_names("aware_old_safe") == ["aware_old_safe"]
    with pytest.raises(Exception):
        _profile_names("missing_profile")


def test_aware_target_pass_requires_all_metric_gates_and_resources():
    row = {
        "old_acc": 0.99,
        "min_old": 0.95,
        "seen_new_acc": 0.97,
        "min_seen": 0.93,
        "unknown_reject": 0.99,
        "resource_pass": True,
        "target_old_acc": 0.99,
        "target_min_old": 0.95,
        "target_seen_new_acc": 0.97,
        "target_min_seen": 0.93,
        "target_unknown_reject": 0.99,
    }
    assert _target_pass(row) is True

    row["old_acc"] = 0.98
    assert _target_pass(row) is False


def test_aware_query_indices_excludes_support_and_keeps_unknown_eval_rows():
    payload = {
        "dataset_role": np.asarray(
            [
                "target_old",
                "target_old",
                "target_new",
                "target_new",
                "target_unknown",
                "target_unknown",
            ],
            dtype=object,
        ),
        "tx_ids": np.asarray(["old-a", "old-a", "new-a", "new-a", "unk-a", "unk-a"], dtype=object),
        "rx_ids": np.asarray(["rx-t", "rx-t", "rx-t", "rx-t", "rx-t", "rx-t"], dtype=object),
        "day_ids": np.asarray(["d0", "d1", "d0", "d1", "d0", "d1"], dtype=object),
        "sig_ids": np.asarray(["s0", "s1", "s0", "s1", "s0", "s1"], dtype=object),
    }
    plan = AdapterTrainingPlan(
        source_old_indices=[],
        proxy_unknown_indices=[],
        support_indices=[0, 2],
        support_labels=["old-a", "new-a"],
        target_unknown_indices=[4, 5],
        target_receivers=["rx-t"],
        old_labels=["old-a"],
        seen_new_labels=["new-a"],
        unknown_labels=["unk-a"],
    )

    idx = _query_indices(payload, plan, query_per_class=1, seed=7)

    assert 0 not in idx
    assert 2 not in idx
    assert set(idx) == {1, 3, 4}


def test_aware_topk_mean_uses_available_width():
    values = np.asarray([[0.1, 0.9], [0.5, 0.7]], dtype=np.float32)

    np.testing.assert_allclose(_topk_mean(values, 8), np.asarray([0.5, 0.6], dtype=np.float32))
    np.testing.assert_allclose(_topk_mean(values, 1), np.asarray([0.9, 0.7], dtype=np.float32))
