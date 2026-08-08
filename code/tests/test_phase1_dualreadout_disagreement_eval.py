import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "scripts"))

from eval_phase1_dualreadout_disagreement import evaluate  # noqa: E402


def _write_payload(path: Path, logits: np.ndarray, *, roles=None, tx_ids=None, sig_ids=None) -> None:
    n = int(logits.shape[0])
    roles = roles or ["source", "source", "source", "source", "proxy_unknown", "proxy_unknown"]
    tx_ids = tx_ids or ["old-a", "old-b", "old-a", "old-b", "new-a", "new-b"]
    sig_ids = sig_ids or [f"sig-{i}" for i in range(n)]
    np.savez(
        path,
        tx_logits=np.asarray(logits, dtype=np.float32),
        dataset_role=np.asarray(roles),
        tx_ids=np.asarray(tx_ids),
        rx_ids=np.asarray(["rx"] * n),
        day_ids=np.asarray(["day"] * n),
        eq_ids=np.asarray(["eq"] * n),
        sig_ids=np.asarray(sig_ids),
        sat_scenarios=np.asarray([""] * n),
        channel_views=np.asarray(["clean"] * n),
        manifest_json=np.asarray(json.dumps({"source_tx_ids": ["old-a", "old-b"]})),
    )


def _args(tmp_path: Path, angular: Path, robust: Path) -> Namespace:
    return Namespace(
        angular_npz=str(angular),
        robust_npz=str(robust),
        source_tx_ids="old-a,old-b",
        unknown_tx_ids="new-a,new-b",
        known_query_roles="source",
        unknown_query_roles="proxy_unknown",
        calibration_roles="source",
        conf_quantile=0.05,
        margin_quantile=0.05,
        energy_quantile=0.95,
        js_quantile=0.95,
        unknown_far_target=0.05,
        output_json=str(tmp_path / "metrics.json"),
        score_table_csv=str(tmp_path / "scores.csv"),
    )


def test_dualreadout_uses_robust_class_and_angular_rejection(tmp_path):
    angular = tmp_path / "angular.npz"
    robust = tmp_path / "robust.npz"
    source_and_known = [[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 7.0]]
    _write_payload(angular, np.asarray(source_and_known + [[0.1, 0.0], [0.0, 0.1]]))
    _write_payload(robust, np.asarray(source_and_known + [[4.0, 0.0], [0.0, 4.0]]))

    metrics = evaluate(_args(tmp_path, angular, robust))

    assert metrics["known_closed_accuracy_no_reject"] == 1.0
    assert metrics["unknown_FAR"] == 0.0
    assert metrics["passes_unknown_far_target"] is True
    assert metrics["gate_policy"]["registered_prediction_from"] == "robust_readout"
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "scores.csv").exists()


def test_dualreadout_calibration_does_not_change_with_unknown_logits(tmp_path):
    angular_a = tmp_path / "angular_a.npz"
    angular_b = tmp_path / "angular_b.npz"
    robust_a = tmp_path / "robust_a.npz"
    robust_b = tmp_path / "robust_b.npz"
    known = [[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 7.0]]
    _write_payload(angular_a, np.asarray(known + [[0.1, 0.0], [0.0, 0.1]]))
    _write_payload(angular_b, np.asarray(known + [[100.0, 0.0], [0.0, 100.0]]))
    _write_payload(robust_a, np.asarray(known + [[4.0, 0.0], [0.0, 4.0]]))
    _write_payload(robust_b, np.asarray(known + [[0.0, 4.0], [4.0, 0.0]]))

    first = evaluate(_args(tmp_path, angular_a, robust_a))
    second = evaluate(_args(tmp_path, angular_b, robust_b))

    assert first["angular_gate_calibration"] == second["angular_gate_calibration"]
    assert first["disagreement_calibration"] == second["disagreement_calibration"]


def test_dualreadout_rejects_metadata_mismatch(tmp_path):
    angular = tmp_path / "angular.npz"
    robust = tmp_path / "robust.npz"
    logits = np.asarray([[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 7.0], [0.1, 0.0], [0.0, 0.1]])
    _write_payload(angular, logits)
    _write_payload(robust, logits, tx_ids=["old-a", "old-b", "old-a", "old-b", "new-a", "different"])

    with pytest.raises(ValueError, match="metadata differ for tx_ids"):
        evaluate(_args(tmp_path, angular, robust))


def test_dualreadout_rejects_same_bucket_physical_row_permutation(tmp_path):
    angular = tmp_path / "angular.npz"
    robust = tmp_path / "robust.npz"
    logits = np.asarray([[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 7.0], [0.1, 0.0], [0.0, 0.1]])
    sig_ids = [f"sig-{i}" for i in range(len(logits))]
    permuted = sig_ids.copy()
    permuted[0], permuted[2] = permuted[2], permuted[0]
    _write_payload(angular, logits, sig_ids=sig_ids)
    _write_payload(robust, logits, sig_ids=permuted)

    with pytest.raises(ValueError, match="metadata differ for sig_ids"):
        evaluate(_args(tmp_path, angular, robust))


def test_dualreadout_rejects_missing_physical_row_ids(tmp_path):
    angular = tmp_path / "angular.npz"
    robust = tmp_path / "robust.npz"
    logits = np.asarray([[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 7.0], [0.1, 0.0], [0.0, 0.1]])
    _write_payload(angular, logits, sig_ids=[""] * len(logits))
    _write_payload(robust, logits, sig_ids=[""] * len(logits))

    with pytest.raises(ValueError, match="require non-empty sig_ids"):
        evaluate(_args(tmp_path, angular, robust))
