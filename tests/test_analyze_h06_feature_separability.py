import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _unit(x):
    arr = np.asarray(x, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def test_h06_feature_separability_uses_existing_manifest_splits(tmp_path: Path):
    features = []
    tx_ids = []
    roles = []

    def add(tx: str, role: str, rows):
        for row in rows:
            features.append(row)
            tx_ids.append(tx)
            roles.append(role)

    add("old_a", "source", [[1.0, 0.0], [0.98, 0.04], [0.96, -0.04], [0.95, 0.02]])
    add("old_b", "source", [[0.0, 1.0], [0.05, 0.98], [-0.04, 0.96], [0.02, 0.95]])
    add("old_a", "target_old", [[0.90, 0.08], [0.88, 0.10], [0.86, 0.12], [0.84, 0.14]])
    add("old_b", "target_old", [[0.08, 0.90], [0.10, 0.88], [0.12, 0.86], [0.14, 0.84]])
    add("unk_x", "target_unknown", [[-1.0, 0.0], [-0.9, -0.1], [-0.8, -0.2], [-0.7, -0.3]])
    arr = _unit(np.asarray(features, dtype=np.float32))
    npz = tmp_path / "features.npz"
    np.savez(
        npz,
        features=arr,
        tx_ids=np.asarray(tx_ids, dtype=str),
        dataset_role=np.asarray(roles, dtype=str),
        manifest_json=json.dumps(
            {
                "source_tx_ids": ["old_a", "old_b"],
                "target_old_tx_ids": ["old_a", "old_b"],
                "unknown_tx_ids": ["unk_x"],
                "star_ground_channel_impl": "simplified_leo_residual",
            }
        ),
    )
    manifest = {
        "source_label_map": {"old_a": 0, "old_b": 1},
        "split_indices_by_role": {
            "source_prototype": [0, 1, 4, 5],
            "target_old_support": [8, 12],
            "target_old_query": [9, 10, 13, 14],
            "unknown_query": [16, 17, 18, 19],
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out = tmp_path / "diag.json"
    cmd = [
        sys.executable,
        "tools/analyze_h06_feature_separability.py",
        "--feature-npz",
        str(npz),
        "--output-json",
        str(out),
        "--source-proto-per-tx",
        "2",
        "--target-old-support-per-tx",
        "1",
        "--target-old-query-per-tx",
        "2",
        "--unknown-query-per-tx",
        "4",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["candidate_count"] == 1
    detail = data["details"][0]
    assert detail["manifest"]["split_source"] == "existing_manifest_json"
    assert detail["counts"]["support"] == 2
    assert detail["counts"]["target_old_query"] == 4
    assert detail["variants"]["rho_0"]["old_query_acc_no_reject"] == 1.0
    assert detail["variants"]["rho_0"]["old_vs_unknown_best_score_auroc"] > 0.9
