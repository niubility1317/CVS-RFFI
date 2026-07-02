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

from eval_phase1_proxy_unknown_reject import evaluate  # noqa: E402


def test_proxy_unknown_reject_reports_dual_target(tmp_path):
    features = np.asarray(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 1.0],
            [0.0, 1.1],
            [-1.0, 0.0],
            [-1.1, 0.0],
            [0.0, -1.0],
            [0.0, -1.1],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    logits = np.asarray(
        [
            [5.0, 0.0],
            [5.1, 0.0],
            [0.0, 5.0],
            [0.0, 5.1],
            [0.5, 0.4],
            [0.6, 0.4],
            [0.4, 0.5],
            [0.4, 0.6],
            [5.0, 0.0],
            [0.0, 5.0],
            [0.5, 0.4],
            [0.4, 0.5],
        ],
        dtype=np.float32,
    )
    roles = np.asarray(
        [
            "source",
            "source",
            "source",
            "source",
            "proxy_unknown",
            "proxy_unknown",
            "proxy_unknown",
            "proxy_unknown",
            "target_old",
            "target_old",
            "target_unknown",
            "target_unknown",
        ]
    )
    tx_ids = np.asarray(
        [
            "old-a",
            "old-a",
            "old-b",
            "old-b",
            "proxy-a",
            "proxy-a",
            "proxy-b",
            "proxy-b",
            "old-a",
            "old-b",
            "new-a",
            "new-b",
        ]
    )
    npz = tmp_path / "features.npz"
    np.savez(
        npz,
        features=features,
        tx_logits=logits,
        dataset_role=roles,
        tx_ids=tx_ids,
        rx_ids=np.asarray(["rx"] * len(tx_ids)),
        day_ids=np.asarray(["day"] * len(tx_ids)),
        sat_scenarios=np.asarray([""] * len(tx_ids)),
        channel_views=np.asarray(["clean"] * len(tx_ids)),
        manifest_json=np.asarray(json.dumps({"source_tx_ids": ["old-a", "old-b"]})),
    )
    out_json = tmp_path / "metrics.json"
    score_csv = tmp_path / "scores.csv"
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
            source_accept_quantile=0.995,
            proxy_far_quantile=0.05,
            epochs=200,
            lr=0.03,
            l2=1e-4,
            seed=7,
            unknown_far_target=0.05,
            max_old_drop_pp=3.0,
            output_json=str(out_json),
            score_table_csv=str(score_csv),
        )
    )
    assert metrics["known_closed_accuracy_no_reject"] == 1.0
    assert metrics["known_full_accuracy_after_reject"] == 1.0
    assert metrics["unknown_FAR"] == 0.0
    assert metrics["passes_dual_target"] is True
    assert metrics["threshold_scope"].startswith("source_old_and_source_proxy_unknown")
    assert out_json.exists()
    assert score_csv.exists()
