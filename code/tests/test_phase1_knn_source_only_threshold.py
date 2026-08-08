from __future__ import annotations

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

from eval_phase1_knn_reject import evaluate  # noqa: E402


def _write_payload(path: Path, unknown_features: np.ndarray) -> None:
    source_a = np.asarray(
        [[1.00, 0.00, 0.00], [1.00, 0.02, 0.01], [0.98, -0.01, 0.03], [1.02, 0.01, -0.02], [0.97, 0.03, -0.01], [1.01, -0.03, 0.02], [0.99, 0.04, 0.03]],
        dtype=np.float32,
    )
    source_b = np.asarray(
        [[0.00, 1.00, 0.00], [0.02, 0.98, 0.01], [-0.01, 1.02, 0.03], [0.01, 0.97, -0.02], [-0.02, 1.01, 0.02], [0.03, 0.99, -0.03], [-0.03, 1.03, 0.01]],
        dtype=np.float32,
    )
    features = np.concatenate([source_a, source_b, unknown_features], axis=0)
    n = len(features)
    np.savez(
        path,
        features=features,
        tx_logits=np.asarray(
            [[5.0, 0.0]] * (len(source_a) - 1)
            + [[0.0, 5.0]]
            + [[0.0, 5.0]] * len(source_b)
            + [[5.0, 0.0]] * len(unknown_features),
            dtype=np.float32,
        ),
        dataset_role=np.asarray(["source"] * (len(source_a) + len(source_b)) + ["proxy_unknown"] * len(unknown_features)),
        tx_ids=np.asarray(["old-a"] * len(source_a) + ["old-b"] * len(source_b) + ["new-a"] * len(unknown_features)),
        rx_ids=np.asarray([f"rx-{i}" for i in range(n)]),
        day_ids=np.asarray(["day-0"] * n),
        eq_ids=np.asarray(["1"] * n),
        sig_ids=np.asarray([f"sig-{i}" for i in range(n)]),
    )


def _args(npz: Path) -> Namespace:
    return Namespace(
        feature_npz=str(npz),
        source_tx_ids="old-a,old-b",
        unknown_tx_ids="new-a",
        train_known_roles="source",
        proxy_unknown_roles="__disabled__",
        known_query_roles="source",
        unknown_query_roles="proxy_unknown",
        train_known_correct_only=True,
        source_incorrect_as_proxy=True,
        feature_reduce="mean",
        distance="cosine",
        knn_k=5,
        exclude_self=True,
        class_conditional_threshold=True,
        threshold_policy="source_accept",
        source_accept_quantile=0.98,
        proxy_far_quantile=0.05,
        unknown_far_target=0.05,
        max_old_drop_pp=2.0,
        output_json="",
        score_table_csv="",
    )


def test_knn_q98_threshold_is_source_only_when_unknown_rows_change(tmp_path: Path) -> None:
    near = tmp_path / "near.npz"
    far = tmp_path / "far.npz"
    _write_payload(near, np.asarray([[0.99, 0.01, 0.02], [0.97, 0.03, -0.01]], dtype=np.float32))
    _write_payload(far, np.asarray([[-1.0, 0.0, 0.0], [-0.9, -0.1, 0.02]], dtype=np.float32))

    near_metrics = evaluate(_args(near))
    far_metrics = evaluate(_args(far))

    assert near_metrics["proxy_unknown_count"] == 1
    assert far_metrics["proxy_unknown_count"] == 1
    assert near_metrics["threshold"]["threshold_policy"] == "source_accept"
    assert near_metrics["threshold"]["class_thresholds"] == far_metrics["threshold"]["class_thresholds"]
    assert near_metrics["known_coverage"] == far_metrics["known_coverage"]
