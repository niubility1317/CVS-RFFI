import unittest

import torch


class PaicStarGroundSpecTest(unittest.TestCase):
    def test_curriculum_schedule_matches_design_report(self):
        from cvsrffi.paic_star_ground import PAIC_CURRICULUM_SCHEDULE, PAIC_SCENARIOS

        self.assertEqual(
            PAIC_CURRICULUM_SCHEDULE,
            "1@0.30:mixed_orbit;"
            "41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;"
            "91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp",
        )
        self.assertEqual(
            PAIC_SCENARIOS,
            ("clear_leo", "low_elev_leo", "rain_leo", "storm_mp", "mixed_orbit"),
        )

    def test_satellite_meta_summary_reports_quantiles_and_ratios(self):
        from cvsrffi.paic_star_ground import summarize_satellite_meta

        meta = {
            "orbit": torch.tensor([0, 0, 1, 2]),
            "state": torch.tensor([0, 1, 1, 2]),
            "theta_deg": torch.tensor([10.0, 20.0, 30.0, 40.0]),
            "snr_db": torch.tensor([8.0, 10.0, 12.0, 14.0]),
            "fD_hz": torch.tensor([-100.0, 0.0, 100.0, 200.0]),
            "cfo_hz": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "K_db": torch.tensor([0.0, 4.0, 8.0, 12.0]),
        }

        summary = summarize_satellite_meta(meta, scenario="mixed_orbit")

        self.assertEqual(summary["scenario"], "mixed_orbit")
        self.assertEqual(summary["sample_count"], 4)
        self.assertAlmostEqual(summary["orbit_ratio"]["LEO"], 0.5)
        self.assertAlmostEqual(summary["state_ratio"]["LOO"], 0.5)
        self.assertEqual(summary["theta_deg_p50"], 25.0)
        self.assertEqual(summary["snr_db_p10"], 8.6)
        self.assertEqual(summary["fD_hz_p90"], 170.0)


if __name__ == "__main__":
    unittest.main()
