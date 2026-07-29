import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "scripts"))

from eval_phase1_open_set_reject import evaluate, parse_args  # noqa: E402
from cvsrffi.phase2_prototypes import attach_endpoint_accept_v1_manifest  # noqa: E402


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
            "--endpoint_mode",
            "diagnostic_dynamic_gate_v0",
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


def test_phase1_open_set_reject_eval_uses_strict_offline_endpoint_identity(tmp_path):
    proto = tmp_path / "strict.pt"
    feats = tmp_path / "strict_features.npz"
    components = []
    for mu in ([1.0, 0.0], [0.0, 1.0]):
        components.append([{
            "component_id": 0,
            "source_domains": [0],
            "n_samples": 20,
            "mu": mu,
            "r_core_deg": 6.0,
            "r_accept_deg": 12.0,
            "r_tail_deg": 18.0,
            "r_vac_deg": 24.0,
            "density_p05": 0.0,
            "density_p10": 0.0,
            "nll_p95": 10.0,
            "nll_tail_p95": 10.0,
            "accept_enabled": True,
        }])
    package = attach_endpoint_accept_v1_manifest({
        "feature_key": "z_id",
        "fused_tx_prototypes": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        "fused_tx_mask": torch.ones(2, 1, dtype=torch.bool),
        "fusion_components": components,
        "fusion_accept_policy": "local_component",
        "global_fused_radius_is_accept_region": False,
        "endpoint_gate_thresholds": {
            "energy_max_by_class": {"0": 0.0, "1": 0.0},
            "energy_temperature": 1.0,
            "energy_formula_id": "negative_logsumexp_temperature_v1",
            "density_formula_id": "exp_neg_sq_normalized_angle_v1",
            "nll_formula_id": "half_sq_normalized_angle_v1",
            "logit_margin_core_min": 0.5,
            "logit_margin_tail_min": 1.0,
            "geo_margin_core_min_deg": 2.0,
            "geo_margin_tail_min_deg": 4.0,
            "allow_tail_auto_accept": False,
            "use_density_gate": True,
            "use_nll_gate": True,
            "use_energy_gate": True,
            "use_geo_margin_gate": True,
            "reject_nan": True,
            "reject_zero_direction": True,
            "max_radius_to_inter_ratio": 0.5,
        },
        "endpoint_calibration": {
            "threshold_source": "source_val_only",
            "calibration_split": "source_val",
            "input_num_samples": 40,
            "directional_num_samples": 40,
            "num_samples": 40,
            "zero_direction_excluded_samples": 0,
            "zero_direction_excluded_fraction": 0.0,
            "zero_direction_excluded_by_class": {"0": 0, "1": 0},
            "zero_direction_excluded_fraction_by_class": {"0": 0.0, "1": 0.0},
            "zero_direction_policy": "force_reject_exclude_from_angular_calibration_v1",
            "max_zero_direction_fraction": 0.001,
            "class_sample_counts": {"0": 20, "1": 20},
        },
        "metadata": {
            "source_checkpoint_sha256": "a" * 64,
            "run_id": "unit",
            "candidate_id": "strict",
            "known_class_count": 2,
            "class_id_to_tx": ["old_a", "old_b"],
            "logit_class_order": [0, 1],
            "checkpoint_load_strict": True,
            "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
            "endpoint_runtime_entry_parity_digest": "b" * 64,
            "endpoint_runtime_entry_parity_sample_count": 8,
        },
    })
    torch.save(package, proto)
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    logits = np.asarray([[4.0, 1.0], [1.0, 4.0], [4.0, 1.0]], dtype=np.float32)
    np.savez(
        feats,
        features=features,
        logits=logits,
        dataset_role=np.asarray(["target_old", "target_old", "target_unknown"]),
        tx_ids=np.asarray(["old_a", "old_b", "unk_x"]),
        manifest_json=np.asarray(json.dumps({
            "feature_key": "z_id",
            "source_checkpoint_sha256": "a" * 64,
            "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
            "class_id_to_tx": ["old_a", "old_b"],
            "logit_class_order": [0, 1],
            "checkpoint_load_strict": True,
        })),
    )
    args = parse_args([
        "--feature_npz", str(feats),
        "--prototype_package", str(proto),
        "--source_tx_ids", "old_a,old_b",
        "--unknown_tx_ids", "unk_x",
    ])

    metrics = evaluate(args)

    assert metrics["endpoint_mode"] == "endpoint_accept_v1"
    assert metrics["known_query_count"] == 2
    assert metrics["unknown_query_count"] == 1
