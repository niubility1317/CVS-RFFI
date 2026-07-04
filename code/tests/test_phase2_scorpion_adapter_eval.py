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
        add("target_unknown", "unk-a", rx, "d2", "unk-query", "leo_clear_weak", [0.45, 0.45, 0.8])
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


class Phase2ScorpionAdapterEvalTest(unittest.TestCase):
    def test_runs_support_only_adapter_without_unknown_fit(self):
        from phase2_scorpion_adapter_eval import run_scorpion_adapter
        from phase2_scorpion_cvs_eval import _parse_weighted_components

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            result = run_scorpion_adapter(
                feature_npz=npz,
                collab_counts=None,
                k_shot=1,
                query_per_class=1,
                seed=7,
                ridge_lambda=0.1,
                boundary_ridge_lambda=0.1,
                class_temperature=0.05,
                boundary_temperature=0.25,
                support_threshold_quantile=0.05,
                virtual_negative_policy="shell_mix",
                virtual_negative_shell_scale=1.5,
                virtual_negative_mix_pairs_per_class=2,
                event_alignment_policy="receiver_domain_ranked",
                support_selection_policy="stable_first",
                evidence_packet_bytes=112.0,
                risk_components=_parse_weighted_components("virtual_unknown_risk:1"),
                unknown_gate=0.52,
                old_shield_gate=0.68,
                min_margin=0.0,
                min_pvalue=0.0,
                min_quality=0.0,
            )

        self.assertEqual(result["algorithm"], "SCORPION-CVS-support-virtual-negative-adapter")
        self.assertFalse(result["unknown_query_used_for_threshold_fit"])
        self.assertTrue(result["adapter_metadata"]["unknown_query_eval_only"])
        self.assertEqual(set(result["counts"]), {"1", "2"})
        self.assertGreater(result["adapter_metadata"]["state_size_bytes"], 0)

    def test_cli_writes_outputs(self):
        from phase2_scorpion_adapter_eval import main

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            output_json = Path(td) / "out.json"
            output_rows = Path(td) / "rows.csv"
            output_evidence = Path(td) / "evidence.csv"
            _write_npz(npz)
            rc = main(
                [
                    "--feature_npz",
                    str(npz),
                    "--output_json",
                    str(output_json),
                    "--output_rows_csv",
                    str(output_rows),
                    "--output_evidence_csv",
                    str(output_evidence),
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(output_json.exists())
            self.assertTrue(output_rows.exists())
            self.assertTrue(output_evidence.exists())


if __name__ == "__main__":
    unittest.main()
