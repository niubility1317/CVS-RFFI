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

from eval_target_old_linear_probe_upper_bound import evaluate  # noqa: E402


def _write_npz(path: Path) -> None:
    rows = [
        ("target_old", "old-a", "rx-a", "d0", "a0", [1.00, 0.10, 0.00]),
        ("target_old", "old-a", "rx-b", "d1", "a1", [0.95, 0.15, 0.00]),
        ("target_old", "old-a", "rx-c", "d2", "a2", [0.90, 0.12, 0.00]),
        ("target_old", "old-b", "rx-a", "d0", "b0", [0.10, 1.00, 0.00]),
        ("target_old", "old-b", "rx-b", "d1", "b1", [0.15, 0.95, 0.00]),
        ("target_old", "old-b", "rx-c", "d2", "b2", [0.12, 0.90, 0.00]),
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


def test_linear_probe_upper_bound_uses_only_target_old_rows(tmp_path):
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
            "1,2",
            "--ridge_lambdas",
            "0.01,1.0",
            "--output_json",
            str(out),
            "--summary_csv",
            str(csv),
        ]
    )

    assert metrics["phase"] == "target_old_only_linear_probe_upper_bound_diagnostic"
    assert metrics["verdict_scope"] == "non_deployment_target_old_only_diagnostic"
    assert metrics["target_unknown_training_count"] == 0
    assert metrics["uses_unknown_query_for_threshold"] is False
    assert metrics["uses_unknown_for_model_selection"] is False
    assert metrics["lambda_selection_scope"] == "reported_grid_diagnostic_not_deployment_selection"
    assert len(metrics["results"]) == 4
    first = metrics["results"][0]
    assert first["support_query_overlap_count"] == 0
    assert len(first["support_index_sha256"]) == 64
    assert len(first["query_index_sha256"]) == 64
    assert first["ignored_non_target_old_rows"] == 1
    assert first["old_acc"] == 1.0
    assert first["macro_old_acc"] == 1.0
    assert first["min_old_class_acc"] == 1.0
    assert first["invalid_classes"] == []
    assert first["per_tx_query_count"] == {"old-a": 2, "old-b": 2}
    assert first["confusion"]["old-a"]["old-a"] == 2
    assert first["confusion"]["old-b"]["old-b"] == 2
    assert out.exists()
    assert csv.exists()


def test_linear_probe_fails_closed_when_no_target_old_rows(tmp_path):
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
