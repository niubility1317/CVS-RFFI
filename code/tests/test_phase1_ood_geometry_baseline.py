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

from eval_phase1_ood_geometry_baseline import evaluate  # noqa: E402


def _write_tiny_feature_npz(path: Path, *, include_logits: bool = True) -> None:
    roles = np.asarray(
        ["source", "source", "source", "source", "target_old", "target_old", "target_unknown", "target_unknown"]
    )
    tx_ids = np.asarray(["old-a", "old-a", "old-b", "old-b", "old-a", "old-b", "unk-a", "unk-b"])
    features = np.asarray(
        [
            [1.00, 0.00, 0.00],
            [0.98, 0.04, 0.00],
            [0.00, 1.00, 0.00],
            [0.03, 0.97, 0.00],
            [0.99, 0.01, 0.00],
            [0.01, 0.99, 0.00],
            [0.00, 0.00, 1.00],
            [-0.70, -0.70, 0.10],
        ],
        dtype=np.float32,
    )
    arrays = {
        "features": features,
        "dataset_role": roles,
        "tx_ids": tx_ids,
        "rx_ids": np.asarray(["rx"] * len(tx_ids)),
        "day_ids": np.asarray(["day"] * len(tx_ids)),
        "sat_scenarios": np.asarray(["leo_clear_weak"] * len(tx_ids)),
        "channel_views": np.asarray(["leo"] * len(tx_ids)),
        "manifest_json": np.asarray(json.dumps({"source_tx_ids": ["old-a", "old-b"]})),
    }
    if include_logits:
        arrays["tx_logits"] = np.asarray(
            [
                [5.0, 0.0],
                [4.8, 0.1],
                [0.0, 5.0],
                [0.1, 4.9],
                [4.9, 0.0],
                [0.0, 4.9],
                [0.2, 0.1],
                [0.1, 0.2],
            ],
            dtype=np.float32,
        )
    np.savez(path, **arrays)


def _args(npz: Path, tmp_path: Path, **overrides):
    values = dict(
        feature_npz=str(npz),
        source_tx_ids="old-a,old-b",
        unknown_tx_ids="unk-a,unk-b",
        known_query_roles="target_old",
        unknown_query_roles="target_unknown",
        calibration_roles="source",
        distance_quantile=0.95,
        energy_quantile=0.95,
        unknown_far_target=0.05,
        knn_k=1,
        var_floor=1.0e-3,
        use_energy_gate=True,
        output_json=str(tmp_path / "metrics.json"),
        score_table_csv=str(tmp_path / "scores.csv"),
    )
    values.update(overrides)
    return Namespace(**values)


def test_source_calibrated_geometry_baseline_rejects_unknown_without_unknown_threshold_fit(tmp_path):
    npz = tmp_path / "features.npz"
    _write_tiny_feature_npz(npz, include_logits=True)

    metrics = evaluate(_args(npz, tmp_path))

    assert metrics["phase"] == "phase1_source_calibrated_ood_geometry_baseline"
    assert metrics["threshold_scope"] == "source_calibrated_only_no_target_support_no_unknown_query_tuning"
    assert metrics["target_unknown_training_count"] == 0
    assert metrics["uses_unknown_query_for_threshold"] is False
    assert metrics["known_closed_accuracy_no_reject"] == 1.0
    assert metrics["known_full_accuracy_after_reject"] == 1.0
    assert metrics["unknown_FAR"] == 0.0
    assert metrics["passes_unknown_far_target"] is True
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "scores.csv").exists()


def test_geometry_baseline_fails_closed_if_unknown_role_used_for_calibration(tmp_path):
    npz = tmp_path / "features.npz"
    _write_tiny_feature_npz(npz, include_logits=True)

    try:
        evaluate(_args(npz, tmp_path, calibration_roles="source,target_unknown"))
    except RuntimeError as exc:
        assert "LOCAL_PROTOCOL_REPAIR_REQUIRED" in str(exc)
    else:
        raise AssertionError("expected target_unknown calibration role to fail closed")


def test_geometry_baseline_runs_without_logits_as_geometry_only(tmp_path):
    npz = tmp_path / "features.npz"
    _write_tiny_feature_npz(npz, include_logits=False)

    metrics = evaluate(_args(npz, tmp_path, use_energy_gate=False))

    assert metrics["gate_policy"]["energy"] is False
    assert "energy_max" not in metrics["thresholds"]
    assert metrics["unknown_FAR"] == 0.0
