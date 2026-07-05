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

from eval_target_old_mlp_adapter_upper_bound import evaluate  # noqa: E402


def _write_npz(path: Path) -> None:
    rows = [
        ("target_old", "old-a", "rx-a", "d0", "a0", [1.00, 0.10, 0.00]),
        ("target_old", "old-a", "rx-b", "d1", "a1", [0.95, 0.15, 0.00]),
        ("target_old", "old-a", "rx-c", "d2", "a2", [0.90, 0.12, 0.00]),
        ("target_old", "old-a", "rx-d", "d3", "a3", [0.92, 0.10, 0.00]),
        ("target_old", "old-b", "rx-a", "d0", "b0", [0.10, 1.00, 0.00]),
        ("target_old", "old-b", "rx-b", "d1", "b1", [0.15, 0.95, 0.00]),
        ("target_old", "old-b", "rx-c", "d2", "b2", [0.12, 0.90, 0.00]),
        ("target_old", "old-b", "rx-d", "d3", "b3", [0.10, 0.92, 0.00]),
        ("target_new", "new-a", "rx-a", "d0", "n0", [0.70, 0.70, 0.00]),
        ("target_unknown", "unk-a", "rx-a", "d0", "u0", [0.00, 0.00, 1.00]),
    ]
    np.savez(
        path,
        features=np.asarray([r[5] for r in rows], dtype=np.float32),
        dataset_role=np.asarray([r[0] for r in rows]),
        tx_ids=np.asarray([r[1] for r in rows]),
        rx_ids=np.asarray([r[2] for r in rows]),
        day_ids=np.asarray([r[3] for r in rows]),
        sig_ids=np.asarray([r[4] for r in rows]),
        manifest_json=np.asarray(json.dumps({"target_old_tx_ids": ["old-a", "old-b"]})),
    )


def test_mlp_adapter_uses_only_target_old_and_reports_protocol_fields(tmp_path):
    npz = tmp_path / "features.npz"
    _write_npz(npz)
    out = tmp_path / "metrics.json"
    csv = tmp_path / "summary.csv"

    metrics = evaluate(
        [
            "--feature_npz",
            str(npz),
            "--target_old_tx_ids",
            "old-a,old-b",
            "--k_values",
            "2",
            "--seeds",
            "7",
            "--epochs",
            "40",
            "--hidden_dim",
            "8",
            "--lr",
            "0.05",
            "--weight_decay",
            "0.0",
            "--output_json",
            str(out),
            "--summary_csv",
            str(csv),
        ]
    )

    assert metrics["phase"] == "target_old_only_mlp_adapter_upper_bound_diagnostic"
    assert metrics["verdict_scope"] == "non_deployment_target_old_only_diagnostic"
    assert metrics["target_unknown_training_count"] == 0
    assert metrics["target_new_training_count"] == 0
    assert metrics["uses_unknown_query_for_threshold"] is False
    assert metrics["uses_unknown_for_model_selection"] is False
    assert metrics["early_stopping_source"] == "fixed_epoch_no_query_early_stopping"
    row = metrics["results"][0]
    assert row["support_query_overlap_count"] == 0
    assert row["valid_row"] is True
    assert row["invalid_reason"] == ""
    assert row["ignored_non_target_old_rows"] == 2
    assert len(row["support_index_sha256"]) == 64
    assert len(row["query_index_sha256"]) == 64
    assert row["old_acc"] == 1.0
    assert row["macro_old_acc"] == 1.0
    assert row["min_old_class_acc"] == 1.0
    assert row["invalid_classes"] == []
    assert out.exists()
    assert csv.exists()


def test_mlp_adapter_fails_closed_without_target_old(tmp_path):
    npz = tmp_path / "features.npz"
    np.savez(
        npz,
        features=np.asarray([[0.0, 1.0]], dtype=np.float32),
        dataset_role=np.asarray(["target_unknown"]),
        tx_ids=np.asarray(["unk-a"]),
    )

    try:
        evaluate(["--feature_npz", str(npz), "--target_old_tx_ids", "old-a", "--k_values", "1"])
    except ValueError as exc:
        assert "no target_old rows" in str(exc)
    else:
        raise AssertionError("expected missing target_old rows to fail")


def test_mlp_adapter_marks_insufficient_class_row_invalid(tmp_path):
    npz = tmp_path / "features.npz"
    rows = [
        ("target_old", "old-a", "rx-a", "d0", "a0", [1.00, 0.00]),
        ("target_old", "old-b", "rx-a", "d0", "b0", [0.00, 1.00]),
        ("target_old", "old-b", "rx-b", "d1", "b1", [0.10, 0.95]),
        ("target_old", "old-b", "rx-c", "d2", "b2", [0.05, 0.90]),
    ]
    np.savez(
        npz,
        features=np.asarray([r[5] for r in rows], dtype=np.float32),
        dataset_role=np.asarray([r[0] for r in rows]),
        tx_ids=np.asarray([r[1] for r in rows]),
        rx_ids=np.asarray([r[2] for r in rows]),
        day_ids=np.asarray([r[3] for r in rows]),
        sig_ids=np.asarray([r[4] for r in rows]),
        manifest_json=np.asarray(
            json.dumps(
                {
                    "target_old_tx_ids": ["old-a", "old-b"],
                    "source_receiver_ids": ["rx-source"],
                    "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
                    "target_channel_view": "leo_clear_weak",
                }
            )
        ),
    )

    metrics = evaluate(
        [
            "--feature_npz",
            str(npz),
            "--target_old_tx_ids",
            "old-a,old-b",
            "--k_values",
            "2",
            "--seeds",
            "7",
            "--epochs",
            "1",
        ]
    )

    assert metrics["receiver_split_disjoint"] is True
    assert metrics["target_channel_view"] == "leo_clear_weak"
    assert metrics["observed_target_old_rx_ids"] == ["rx-a", "rx-b", "rx-c"]
    assert metrics["observed_target_old_rx_within_target_receiver_ids"] is True
    row = metrics["results"][0]
    assert row["valid_row"] is False
    assert row["invalid_reason"] == "class_has_insufficient_support_or_empty_query"
    assert row["invalid_classes"] == ["old-a"]
    assert row["old_acc"] is None
    assert row["macro_old_acc"] is None
    assert row["min_old_class_acc"] is None
