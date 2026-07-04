import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_npz(path: Path) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b"]:
        add("source", "old-a", "src-a", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query", "leo_clear_weak", [0.98, 0.02, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query-2", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query", "leo_clear_weak", [0.02, 0.98, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query-2", "leo_clear_weak", [0.01, 0.99, 0.0])
        add("target_unknown", "unk-a", rx, "d2", "unk-query", "leo_clear_weak", [0.0, 0.0, 1.0])
        add("target_unknown", "unk-a", rx, "d2", "unk-query-2", "leo_clear_weak", [0.0, 0.01, 0.99])
    manifest = {
        "source_tx_ids": ["old-a"],
        "target_old_tx_ids": ["old-a"],
        "new_tx_ids": ["new-a"],
        "unknown_tx_ids": ["unk-a"],
        "target_channel_view": "satellite/LEO",
    }
    np.savez(
        path,
        features=np.stack([row[6] for row in rows]).astype(np.float32),
        dataset_role=np.asarray([row[0] for row in rows], dtype=object),
        tx_ids=np.asarray([row[1] for row in rows], dtype=object),
        rx_ids=np.asarray([row[2] for row in rows], dtype=object),
        day_ids=np.asarray([row[3] for row in rows], dtype=object),
        sig_ids=np.asarray([row[4] for row in rows], dtype=object),
        sat_scenarios=np.asarray([row[5] for row in rows], dtype=object),
        channel_views=np.asarray(["satellite" if row[5] else "clean" for row in rows], dtype=object),
        manifest_json=np.asarray(json.dumps(manifest)),
    )


class Phase2OrbitC3RGuardEvalTest(unittest.TestCase):
    def test_orbit_c3r_outputs_profiles_and_resource_rows(self):
        from phase2_orbit_c3r_guard_eval import main

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            output = Path(td) / "orbit.json"
            summary = Path(td) / "summary.csv"
            _write_npz(npz)
            rc = main(
                [
                    "--feature_npz",
                    str(npz),
                    "--output_json",
                    str(output),
                    "--output_summary_csv",
                    str(summary),
                    "--profiles",
                    "old_guarded,balanced",
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "2",
                    "--qknn_k",
                    "1",
                    "--virtual_unknown_samples_per_class",
                    "1",
                    "--class_negative_samples_per_class",
                    "1",
                    "--collab_counts",
                    "all",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            summary_exists = summary.exists()

        self.assertEqual(rc, 0)
        self.assertEqual(payload["algorithm"], "ORBIT-C3R Guard")
        self.assertEqual(set(payload["profile_results"]), {"old_guarded", "balanced"})
        self.assertEqual(payload["qknn_metadata"]["qknn_k"], 1)
        self.assertTrue(payload["qknn_metadata"]["unknown_query_eval_only"])
        self.assertEqual({row["collab_count"] for row in payload["summary_rows"]}, {1, 2})
        self.assertTrue(all("resource_pass" in row for row in payload["summary_rows"]))
        self.assertTrue(summary_exists)

    def test_rejects_non_positive_arguments(self):
        from phase2_orbit_c3r_guard_eval import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--feature_npz", "features.npz", "--output_json", "out.json", "--k_shot", "0"])


if __name__ == "__main__":
    unittest.main()
