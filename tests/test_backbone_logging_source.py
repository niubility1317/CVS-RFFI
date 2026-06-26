from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "code" / "train.py"
DATASET_WISIG_PY = ROOT / "code" / "dataset_wisig.py"


class BackboneTrainingLoggingSourceTest(unittest.TestCase):
    def test_backbone_train_prints_fjmp_style_config_sections(self):
        text = TRAIN_PY.read_text(encoding="utf-8")

        for marker in [
            "[CONFIG-BEGIN]",
            "[CONFIG-RUN]",
            "[CONFIG-DATA]",
            "[CONFIG-MODEL]",
            "[CONFIG-OPT]",
            "[CONFIG-LOSS]",
            "[CONFIG-SAT]",
            "[CONFIG-CKPT]",
            "[CONFIG-END]",
        ]:
            self.assertIn(marker, text)

    def test_backbone_epoch_block_prints_raw_and_weighted_loss_lines(self):
        text = TRAIN_PY.read_text(encoding="utf-8")

        for marker in [
            "[EPOCH-BEGIN]",
            "[LOSS-CORE-RAW]",
            "[LOSS-CORE-W]",
            "[LOSS-AUX-RAW]",
            "[LOSS-AUX-W]",
            "[LOSS-SAT-RAW]",
            "[LOSS-SAT-W]",
            "[LOSS-DG-RAW]",
            "[LOSS-DG-W]",
            "[LOSS-WEIGHT]",
            "[LOSS-TOP]",
            "[EPOCH-END]",
        ]:
            self.assertIn(marker, text)

        for weighted_meter in [
            '"w_cls"',
            '"w_dom"',
            '"w_adv"',
            '"w_cls_pa"',
            '"w_sat_cls"',
            '"w_proto"',
            '"w_fishr"',
        ]:
            self.assertIn(weighted_meter, text)

    def test_backbone_train_labels_unseen_day_per_receiver_splits(self):
        text = TRAIN_PY.read_text(encoding="utf-8")

        self.assertIn('name.startswith("test_unseen_day_rx_")', text)
        self.assertIn("on unseen_days", text)

    def test_wisig_dataset_builds_unseen_day_per_receiver_named_tests(self):
        text = DATASET_WISIG_PY.read_text(encoding="utf-8")

        self.assertIn('key = f"test_unseen_day_rx_{r_idx}"', text)
        self.assertIn('split_source=f"full_test_unseen_day_rx_{r_idx}"', text)


if __name__ == "__main__":
    unittest.main()
