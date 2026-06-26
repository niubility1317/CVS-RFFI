import unittest
from types import SimpleNamespace

import torch


class LeoResidualChannelConfigTest(unittest.TestCase):
    def test_simplified_leo_scenarios_expose_residual_channel_contract(self):
        from training_controls import sat_channel_config_for_scenario

        cfg = sat_channel_config_for_scenario("leo_clear_weak")

        self.assertEqual(cfg["channel_model"], "leo_residual")
        self.assertEqual(cfg["orbit_probs"], {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0})
        self.assertTrue(cfg["use_residual_doppler"])
        self.assertFalse(cfg["apply_path_loss_to_iq"])
        self.assertFalse(cfg["enable_atmospheric_fading"])
        self.assertFalse(cfg["enable_iq_imbalance"])
        self.assertTrue(cfg["enable_multipath"])
        self.assertEqual(cfg["multipath_profile"], "weak")

    def test_leo_residual_transform_reports_weak_residual_physics(self):
        from cvsrffi.eval import apply_sat_channel_for_scenario

        gen = torch.Generator().manual_seed(123)
        x = torch.ones(4, 2, 64, dtype=torch.float32)

        out, meta = apply_sat_channel_for_scenario(
            x,
            "leo_clear_weak",
            SimpleNamespace(sat_fs_hz=25e6, sat_fc_hz=2.462e9),
            gen=gen,
            return_meta=True,
        )

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(meta["channel_model"], "leo_residual")
        self.assertTrue(torch.equal(meta["orbit"], torch.zeros(4, dtype=torch.long)))
        self.assertFalse(meta["orbital_doppler_applied"])
        self.assertFalse(meta["path_loss_iq_applied"])
        self.assertFalse(meta["atmospheric_fading_applied"])
        self.assertFalse(meta["iq_imbalance_applied"])
        self.assertEqual(meta["multipath_profile"], "weak")
        self.assertTrue(torch.all(meta["num_taps"] == 2))
        self.assertLess(float(meta["fD_hz"].abs().max()), 1e-6)
        self.assertLess(float(meta["residual_cfo_hz"].abs().max()), 250.0)


if __name__ == "__main__":
    unittest.main()
