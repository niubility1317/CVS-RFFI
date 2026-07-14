import sys
import unittest
from argparse import Namespace
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

    def test_multi_policy_export_reuses_one_adapter_with_safe_subdirs(self):
        from train_apply_phase1_iq_preadapter_20260703 import (
            _resolve_export_tta_policies,
            _resolve_tta_export_subdirs,
            _tta_export_subdir,
        )

        args = Namespace(
            satellite_tta_policy="rx_light5",
            export_tta_policies="none,rx_shift3,rx_cfo3,rx_light5",
        )
        self.assertEqual(
            _resolve_export_tta_policies(args),
            ["none", "rx_shift3", "rx_cfo3", "rx_light5"],
        )
        self.assertEqual(
            _tta_export_subdir("ADAPTER60_FFT96", "rx_shift3", "{base}_{policy}_{view_count}v"),
            "ADAPTER60_FFT96_rx_shift3_3v",
        )
        args.out_subdir = "ADAPTER60_FFT96"
        args.export_tta_subdir_template = "{base}_{policy}"
        self.assertEqual(
            _resolve_tta_export_subdirs(args, _resolve_export_tta_policies(args)),
            [
                "ADAPTER60_FFT96_none",
                "ADAPTER60_FFT96_rx_shift3",
                "ADAPTER60_FFT96_rx_cfo3",
                "ADAPTER60_FFT96_rx_light5",
            ],
        )

    def test_multi_policy_export_rejects_duplicates(self):
        from train_apply_phase1_iq_preadapter_20260703 import (
            _resolve_export_tta_policies,
            _resolve_tta_export_subdirs,
        )

        args = Namespace(satellite_tta_policy="none", export_tta_policies="none,none")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _resolve_export_tta_policies(args)
        args.export_tta_policies = "none,rx_shift3"
        args.out_subdir = "ADAPTER60_FFT96"
        args.export_tta_subdir_template = "{base}"
        with self.assertRaisesRegex(ValueError, "unique directory"):
            _resolve_tta_export_subdirs(args, _resolve_export_tta_policies(args))

    def test_light_tta_policy_view_counts_match_generated_views(self):
        import torch
        from export_spaceborne_features import _satellite_tta_view_count, _satellite_tta_views

        x = torch.zeros((2, 2, 16), dtype=torch.float32)
        for policy, expected in (("none", 1), ("rx_shift3", 3), ("rx_cfo3", 3), ("rx_light5", 5)):
            views = _satellite_tta_views(x, policy)
            self.assertEqual(len(views), expected)
            self.assertEqual(_satellite_tta_view_count(policy), expected)
            self.assertEqual(len({name for name, _ in views}), expected)

    def test_frozen_export_skips_all_adv3b02_adapter_updates(self):
        import torch
        from train_apply_phase1_iq_preadapter_20260703 import (
            build_frozen_backbone_export_adapter,
        )

        args = Namespace(
            input_adapter_enabled=False,
            model_adapter_mode="none",
            sat_scenarios="leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
            star_ground_channel_impl="simplified_leo_residual",
            input_repair="raw",
            clean_input_repair_mode="raw",
        )
        adapter, info = build_frozen_backbone_export_adapter(args, torch.device("cpu"))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 0)
        self.assertEqual(info["epochs"], 0)
        self.assertEqual(info["adv3b02_gradient_updates"], 0)
        self.assertEqual(
            info["downstream_feature_adapter"],
            "qknnv42_support_diag_whiten_fisher",
        )

        args.model_adapter_mode = "id_norm_late_feature"
        with self.assertRaisesRegex(ValueError, "model_adapter_mode none"):
            build_frozen_backbone_export_adapter(args, torch.device("cpu"))

    def test_registered_only_optional_unknown_filter_accepts_none(self):
        from train_apply_phase1_iq_preadapter_20260703 import (
            _filter_optional_dataset_to_reference,
        )

        self.assertIsNone(
            _filter_optional_dataset_to_reference(
                None,
                role="target_unknown",
                reference_keys={"target_unknown": [("a", "b", "c", "d", "e")]},
            )
        )


if __name__ == "__main__":
    unittest.main()
