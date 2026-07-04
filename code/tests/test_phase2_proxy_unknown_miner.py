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


def _arr(n):
    return np.zeros((n, 256, 2), dtype=np.float32)


def _make_ds():
    tx = ["old-a", "new-a", "unk-a", "10-1", "10-2", "11-1"]
    rx = ["src-a", "src-b", "src-c", "target-a"]
    days = ["d0", "d1"]
    eq = [0, 1]
    data = []
    counts = {
        "old-a": [10, 10, 10, 10],
        "new-a": [10, 10, 10, 10],
        "unk-a": [10, 10, 10, 10],
        "10-1": [70, 70, 70, 90],
        "10-2": [80, 80, 0, 90],
        "11-1": [72, 72, 72, 90],
    }
    for tx_id in tx:
        tx_rows = []
        for rx_count in counts[tx_id]:
            rx_rows = []
            for _day in days:
                rx_rows.append([_arr(0), _arr(rx_count // len(days))])
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": tx,
        "rx_list": rx,
        "capture_date_list": days,
        "equalized_list": eq,
    }


class Phase2ProxyUnknownMinerTest(unittest.TestCase):
    def test_excludes_protocol_tx_and_target_receiver(self):
        from phase2_proxy_unknown_miner import build_candidate_table, select_candidates

        rows, audit = build_candidate_table(
            _make_ds(),
            source_tx_ids=["old-a"],
            target_new_tx_ids=["new-a"],
            target_unknown_tx_ids=["unk-a"],
            proxy_source_rxs=["src-a", "src-b", "src-c", "target-a"],
            target_rxs=["target-a"],
            min_source_rx_coverage=3,
            min_samples_per_tx=100,
        )
        by_tx = {row["tx_id"]: row for row in rows}

        self.assertFalse(by_tx["old-a"]["eligible"])
        self.assertEqual(by_tx["old-a"]["excluded_reason"], "reserved_protocol_tx")
        self.assertEqual(by_tx["10-1"]["rx_coverage"], 3)
        self.assertNotIn("target-a", audit["proxy_source_rxs_used"])
        self.assertFalse(audit["target_unknown_used_for_scoring"])

        selected = select_candidates(rows, top_k=2, family_repeat_penalty=0.5)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["tx_id"] not in {"old-a", "new-a", "unk-a"} for row in selected))

    def test_min_coverage_filters_weak_candidate(self):
        from phase2_proxy_unknown_miner import build_candidate_table

        rows, _audit = build_candidate_table(
            _make_ds(),
            source_tx_ids=["old-a"],
            target_new_tx_ids=["new-a"],
            target_unknown_tx_ids=["unk-a"],
            proxy_source_rxs=["src-a", "src-b", "src-c"],
            target_rxs=["target-a"],
            min_source_rx_coverage=3,
            min_samples_per_tx=100,
        )
        by_tx = {row["tx_id"]: row for row in rows}
        self.assertFalse(by_tx["10-2"]["eligible"])
        self.assertEqual(by_tx["10-2"]["excluded_reason"], "insufficient_source_rx_coverage")

    def test_run_miner_writes_json_csv_and_printable_ids(self):
        import pickle
        from phase2_proxy_unknown_miner import build_parser, run_miner

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pkl_path = tmp_path / "mini.pkl"
            json_path = tmp_path / "manifest.json"
            csv_path = tmp_path / "candidates.csv"
            with pkl_path.open("wb") as f:
                pickle.dump(_make_ds(), f)

            args = build_parser().parse_args(
                [
                    "--wisig_pkl",
                    str(pkl_path),
                    "--source_tx_ids",
                    "old-a",
                    "--target_new_tx_ids",
                    "new-a",
                    "--target_unknown_tx_ids",
                    "unk-a",
                    "--proxy_source_rxs",
                    "src-a,src-b,src-c,target-a",
                    "--target_rxs",
                    "target-a",
                    "--top_k",
                    "2",
                    "--min_source_rx_coverage",
                    "3",
                    "--min_samples_per_tx",
                    "100",
                    "--output_json",
                    str(json_path),
                    "--output_csv",
                    str(csv_path),
                ]
            )

            manifest = run_miner(args)

            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(len(manifest["selected_proxy_unknown_tx_ids"]), 2)
            self.assertEqual(manifest["audit"]["selection_basis"], "source_rx_metadata_counts_only")


if __name__ == "__main__":
    unittest.main()
