import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "scripts"))

from eval_phase1_multiview_reject import evaluate  # noqa: E402


def test_multiview_reject_groups_repeated_rows(tmp_path):
    features = []
    logits = []
    roles = []
    tx_ids = []
    sig_ids = []
    # Two old source groups, two proxy groups, two target-old groups, two unknown groups.
    groups = [
        ("source", "old-a", "s0", [1.0, 0.0], [[5.0, 0.0], [4.8, 0.1]]),
        ("source", "old-b", "s1", [0.0, 1.0], [[0.0, 5.0], [0.1, 4.8]]),
        ("proxy_unknown", "proxy-a", "s2", [-1.0, 0.0], [[0.5, 0.4], [0.4, 0.5]]),
        ("proxy_unknown", "proxy-b", "s3", [0.0, -1.0], [[0.4, 0.5], [0.5, 0.4]]),
        ("target_old", "old-a", "s4", [1.0, 0.0], [[5.0, 0.0], [4.8, 0.1]]),
        ("target_old", "old-b", "s5", [0.0, 1.0], [[0.0, 5.0], [0.1, 4.8]]),
        ("target_unknown", "new-a", "s6", [-1.0, 0.0], [[0.5, 0.4], [0.4, 0.5]]),
        ("target_unknown", "new-b", "s7", [0.0, -1.0], [[0.4, 0.5], [0.5, 0.4]]),
    ]
    for role, tx, sig, feat, los in groups:
        for view_i, logit in enumerate(los):
            features.append([feat[0] + 0.01 * view_i, feat[1] - 0.01 * view_i])
            logits.append(logit)
            roles.append(role)
            tx_ids.append(tx)
            sig_ids.append(sig)
    npz = tmp_path / "features.npz"
    np.savez(
        npz,
        features=np.asarray(features, dtype=np.float32),
        tx_logits=np.asarray(logits, dtype=np.float32),
        dataset_role=np.asarray(roles),
        tx_ids=np.asarray(tx_ids),
        rx_ids=np.asarray(["rx"] * len(tx_ids)),
        day_ids=np.asarray(["day"] * len(tx_ids)),
        eq_ids=np.asarray(["1"] * len(tx_ids)),
        sig_ids=np.asarray(sig_ids),
        channel_views=np.asarray(["clean", "satellite"] * len(groups)),
        sat_scenarios=np.asarray(["", "leo_clear_weak"] * len(groups)),
        manifest_json=np.asarray(json.dumps({"source_tx_ids": ["old-a", "old-b"]})),
    )
    metrics = evaluate(
        Namespace(
            feature_npz=str(npz),
            source_tx_ids="old-a,old-b",
            unknown_tx_ids="new-a,new-b",
            train_known_roles="source",
            proxy_unknown_roles="proxy_unknown",
            known_query_roles="target_old",
            unknown_query_roles="target_unknown",
            threshold_policy="source_accept",
            source_accept_quantile=1.0,
            proxy_far_quantile=0.05,
            head_type="linear",
            hidden_dim=8,
            epochs=120,
            lr=0.03,
            l2=1e-4,
            seed=11,
            unknown_far_target=0.05,
            max_old_drop_pp=3.0,
            output_json="",
            score_table_csv="",
        )
    )
    assert metrics["group_count"] == 8
    assert metrics["known_closed_accuracy_no_reject"] == 1.0
    assert metrics["known_full_accuracy_after_reject"] == 1.0
    assert metrics["unknown_FAR"] == 0.0
    assert metrics["passes_dual_target"] is True
