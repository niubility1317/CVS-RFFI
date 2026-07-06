import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class TrainApplyPhase1IqPreadapterCellTest(unittest.TestCase):
    def test_parse_export_cell_keeps_legacy_unknown_only_format(self):
        from train_apply_phase1_iq_preadapter_20260703 import _parse_export_cell

        parsed = _parse_export_cell("CASE:7-14:10-10,11-10")

        self.assertEqual(
            parsed,
            {
                "name": "CASE",
                "target_rx": "7-14",
                "target_new_tx": "",
                "target_unknown_tx": "10-10,11-10",
            },
        )

    def test_parse_export_cell_accepts_explicit_stage2c_new_and_unknown_sets(self):
        from train_apply_phase1_iq_preadapter_20260703 import _parse_export_cell

        parsed = _parse_export_cell("CASE:7-14:1-10,1-12:10-7,11-1")

        self.assertEqual(parsed["target_new_tx"], "1-10,1-12")
        self.assertEqual(parsed["target_unknown_tx"], "10-7,11-1")

    def test_parse_export_cell_rejects_new_unknown_overlap(self):
        from train_apply_phase1_iq_preadapter_20260703 import _parse_export_cell

        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            _parse_export_cell("CASE:7-14:1-10,1-12:1-12,10-7")


if __name__ == "__main__":
    unittest.main()
