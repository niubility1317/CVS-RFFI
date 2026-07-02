import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "scripts"))

from eval_phase1_open_set_reject import evaluate, parse_args  # noqa: E402


def _write_prototypes(path: Path) -> None:
    package = {
        "feature_key": "z_id",
        "fusion_components": [
            [
                {
                    "component_id": 0,
                    "source_domains": [0],
                    "n_samples": 12,
                    "mu": [1.0, 0.0],
                    "r_core_deg": 12.0,
                    "r_accept_deg": 12.0,
                    "density_p05": None,
                    "density_p10": None,
                    "nll_p95": None,
                    "accept_enabled": True,
                }
            ],
            [
                {
                    "component_id": 0,
                    "source_domains": [0],
                    "n_samples": 12,
                    "mu": [0.0, 1.0],
                    "r_core_deg": 12.0,
                    "r_accept_deg": 12.0,
                    "density_p05": None,
                    "density_p10": None,
                    "nll_p95": None,
                    "accept_enabled": True,
                }
            ],
        ],
    }
    path.write_text(json.dumps(package), encoding="utf-8")


def _write_features(path: Path) -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.0, 1.0],
            [0.087, 0.996],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.707, 0.707],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        features=features,
        dataset_role=np.asarray(["source", "source", "source", "source", "target_old", "target_old", "target_unknown", "target_unknown"]),
        tx_ids=np.asarray(["old_a", "old_a", "old_b", "old_b", "old_a", "old_b", "unk_x", "unk_y"]),
        rx_ids=np.asarray(["rx0"] * 8),
        day_ids=np.asarray(["d0"] * 8),
        channel_views=np.asarray(["clean"] * 8),
        sat_scenarios=np.asarray([""] * 8),
        manifest_json=np.asarray(json.dumps({"payload_source": "unit"})),
    )


def test_phase1_open_set_reject_eval_uses_source_calibrated_gate(tmp_path):
    proto = tmp_path / "proto.json"
    feats = tmp_path / "features.npz"
    metrics_path = tmp_path / "metrics.json"
    score_path = tmp_path / "scores.csv"
    _write_prototypes(proto)
    _write_features(feats)

    args = parse_args(
        [
            "--feature_npz",
            str(feats),
            "--prototype_package",
            str(proto),
            "--source_tx_ids",
            "old_a,old_b",
            "--unknown_tx_ids",
            "unk_x,unk_y",
            "--core_quantile",
            "0.90",
            "--max_core_radius_deg",
            "8.0",
            "--min_geo_margin_deg",
            "6.0",
            "--output_json",
            str(metrics_path),
            "--score_table_csv",
            str(score_path),
        ]
    )

    metrics = evaluate(args)

    assert metrics["threshold_scope"] == "source_calibrated_only_no_target_support_no_unknown_query_tuning"
    assert metrics["known_query_count"] == 2
    assert metrics["known_full_accuracy"] == 1.0
    assert metrics["unknown_query_count"] == 2
    assert metrics["unknown_FAR"] == 0.0
    assert metrics["passes_unknown_far_target"] is True
    assert metrics_path.exists()
    assert score_path.exists()
