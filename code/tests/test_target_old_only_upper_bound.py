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

from eval_target_old_only_upper_bound import evaluate  # noqa: E402


def _write_npz(path: Path) -> None:
    rows = [
        ("target_old", "old-a", "rx-a", "d0", "s0", [1.00, 0.00, 0.00]),
        ("target_old", "old-a", "rx-a", "d1", "s1", [0.98, 0.02, 0.00]),
        ("target_old", "old-a", "rx-b", "d2", "s2", [0.97, 0.03, 0.00]),
        ("target_old", "old-b", "rx-a", "d0", "s3", [0.00, 1.00, 0.00]),
        ("target_old", "old-b", "rx-a", "d1", "s4", [0.02, 0.98, 0.00]),
        ("target_old", "old-b", "rx-b", "d2", "s5", [0.03, 0.97, 0.00]),
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


def test_target_old_only_upper_bound_uses_only_target_old_support_and_query(tmp_path):
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
            "--output_json",
            str(out),
            "--summary_csv",
            str(csv),
        ]
    )

    assert metrics["phase"] == "target_old_only_upper_bound_diagnostic"
    assert metrics["target_unknown_training_count"] == 0
    assert metrics["uses_unknown_query_for_threshold"] is False
    assert metrics["verdict_scope"] == "non_deployment_target_old_only_diagnostic"
    assert metrics["results"][0]["k"] == 1
    assert metrics["results"][0]["old_acc"] == 1.0
    assert metrics["results"][0]["min_old_class_acc"] == 1.0
    assert metrics["results"][0]["support_query_overlap_count"] == 0
    assert metrics["results"][0]["ignored_non_target_old_rows"] == 1
    assert out.exists()
    assert csv.exists()


def test_target_old_only_upper_bound_fails_without_target_old_rows(tmp_path):
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
