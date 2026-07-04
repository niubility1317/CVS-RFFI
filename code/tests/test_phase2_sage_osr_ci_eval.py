import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import argparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_npz(path: Path) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for n in range(4):
        add("source", "old-a", "src-a", "d0", f"src-old-a-{n}", "", [1.0, 0.0, 0.0])
        add("source", "old-b", "src-b", "d0", f"src-old-b-{n}", "", [0.0, 1.0, 0.0])
        add("proxy_unknown", "proxy-a", "src-p", "d0", f"proxy-{n}", "leo_clear_weak", [0.0, 0.0, 1.0])
    for rx in ["rx-a", "rx-b"]:
        add("target_old", "old-a", rx, "d1", f"old-a-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_old", "old-b", rx, "d1", f"old-b-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.7, 0.7, 0.0])
        for q in range(2):
            add("target_old", "old-a", rx, "d2", f"old-a-query-{q}", "leo_clear_weak", [0.98, 0.02, 0.0])
            add("target_old", "old-b", rx, "d2", f"old-b-query-{q}", "leo_clear_weak", [0.02, 0.98, 0.0])
            add("target_new", "new-a", rx, "d2", f"new-query-{q}", "leo_clear_weak", [0.68, 0.70, 0.0])
            add("target_unknown", "unk-a", rx, "d2", f"unk-query-{q}", "leo_clear_weak", [0.0, 0.0, 1.0])
    manifest = {
        "source_tx_ids": ["old-a", "old-b"],
        "target_old_tx_ids": ["old-a", "old-b"],
        "new_tx_ids": ["new-a"],
        "unknown_tx_ids": ["unk-a"],
        "proxy_unknown_tx_ids": ["proxy-a"],
        "target_channel_view": "satellite/LEO",
    }
    np.savez(
        path,
        features=np.stack([r[6] for r in rows]).astype(np.float32),
        dataset_role=np.asarray([r[0] for r in rows], dtype=object),
        tx_ids=np.asarray([r[1] for r in rows], dtype=object),
        rx_ids=np.asarray([r[2] for r in rows], dtype=object),
        day_ids=np.asarray([r[3] for r in rows], dtype=object),
        sig_ids=np.asarray([r[4] for r in rows], dtype=object),
        sat_scenarios=np.asarray([r[5] for r in rows], dtype=object),
        channel_views=np.asarray(["satellite" if r[5] else "clean" for r in rows], dtype=object),
        manifest_json=np.asarray(json.dumps(manifest)),
    )


def test_sage_osr_runs_cpu_smoke_and_keeps_unknown_eval_only():
    from phase2_sage_osr_ci_eval import main

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        npz = root / "features.npz"
        out = root / "out"
        _write_npz(npz)

        rc = main(
            [
                "--feature_npz",
                str(npz),
                "--output_dir",
                str(out),
                "--device",
                "cpu",
                "--adapter_epochs",
                "2",
                "--adapter_rank",
                "2",
                "--batch_size",
                "4",
                "--k_shot",
                "1",
                "--query_per_class",
                "2",
                "--collab_counts",
                "all",
            ]
        )

        assert rc == 0
        summary = json.loads((out / "sage_osr_summary.json").read_text(encoding="utf-8"))
        aware = json.loads((out / "sage_osr_aware.json").read_text(encoding="utf-8"))
        assert summary["target_unknown_eval_only"] is True
        assert summary["training_counts"]["target_unknown_training_count"] == 0
        assert summary["training_counts"]["proxy_unknown"] > 0
        assert {str(row["collab_count"]) for row in aware["summary_rows"]} == {"1", "2"}
        assert Path(summary["adapted_feature_npz"]).exists()
        assert summary["adapter_metadata"]["train_metrics"]["curriculum"] == "two_stage"
        assert summary["adapter_metadata"]["train_metrics"]["alignment_epochs"] == 1
        assert summary["adapter_metadata"]["train_metrics"]["negative_epochs"] == 1


def test_sage_osr_two_stage_epoch_split_and_tx_balanced_proxy_sampling():
    from phase2_sage_osr_ci_eval import _sample_proxy_indices, _split_stage_epochs

    args = argparse.Namespace(
        curriculum="two_stage",
        adapter_epochs=5,
        alignment_epochs=-1,
        negative_epochs=-1,
        alignment_fraction=0.6,
    )
    assert _split_stage_epochs(args) == (3, 2)

    rng = np.random.default_rng(7)
    out = _sample_proxy_indices(
        rng=rng,
        proxy_indices=[0, 1, 2, 3, 4, 5],
        proxy_groups=[np.asarray([0, 1, 2]), np.asarray([3, 4, 5])],
        take=4,
        policy="tx_balanced",
    )

    assert len(out) == 4
    assert any(int(v) in {0, 1, 2} for v in out)
    assert any(int(v) in {3, 4, 5} for v in out)
