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
        add("target_old", "old-a", rx, "d2", "old-query", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query", "leo_clear_weak", [0.01, 0.99, 0.0])
        add("target_unknown", "unk-a", rx, "d1", f"unk-support-{rx}", "leo_clear_weak", [0.0, 0.0, 1.0])
        add("target_unknown", "unk-a", rx, "d2", "unk-query", "leo_clear_weak", [0.0, 0.0, 0.99])
    manifest = {
        "source_tx_ids": ["old-a"],
        "target_old_tx_ids": ["old-a"],
        "new_tx_ids": ["new-a"],
        "unknown_tx_ids": ["unk-a"],
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


class Phase2OracleUnknownHoldoutEvalTest(unittest.TestCase):
    def test_marks_oracle_holdout_as_non_deployment(self):
        from phase2_oracle_unknown_holdout_eval import main

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            output = Path(td) / "out.json"
            rows = Path(td) / "rows.csv"
            evidence = Path(td) / "evidence.csv"
            _write_npz(npz)
            rc = main(
                [
                    "--feature_npz",
                    str(npz),
                    "--output_json",
                    str(output),
                    "--output_rows_csv",
                    str(rows),
                    "--output_evidence_csv",
                    str(evidence),
                    "--k_shot",
                    "1",
                    "--oracle_unknown_shot",
                    "1",
                    "--query_per_class",
                    "1",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertTrue(payload["non_deployment_diagnostic"])
        self.assertTrue(payload["labeled_unknown_support_used_for_boundary_fit"])
        self.assertFalse(payload["unknown_query_used_for_threshold_fit"])
        self.assertTrue(payload["oracle_metadata"]["labeled_unknown_support_used_for_boundary_fit"])
        self.assertTrue(payload["oracle_metadata"]["unknown_query_eval_only"])
        self.assertEqual(payload["oracle_metadata"]["threshold_scope"], "oracle_unknown_holdout")
        self.assertIn("not_deployable_reason", payload["oracle_metadata"])
        self.assertEqual(set(payload["counts"]), {"1", "2"})

    def test_rejects_non_positive_shot_arguments(self):
        from phase2_oracle_unknown_holdout_eval import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--feature_npz", "features.npz", "--output_json", "out.json", "--k_shot", "0"])


if __name__ == "__main__":
    unittest.main()
