import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Phase2OrbitSlevCiEvalTest(unittest.TestCase):
    def test_energy_risk_increases_above_support_threshold(self):
        from phase2_orbit_slev_ci_eval import _energy_risk

        low = _energy_risk(-12.0, -10.0, risk_temperature=0.5, margin=0.0)
        high = _energy_risk(-8.0, -10.0, risk_temperature=0.5, margin=0.0)

        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.9)

    def test_slev_augmentation_keeps_unknown_query_out_of_threshold(self):
        from phase2_orbit_slev_ci_eval import SlevEnergyBundle, augment_slev_evidence

        bundle = SlevEnergyBundle(
            query_energy_by_row={
                ("old-ok", "rx-a"): -12.0,
                ("unk-bad", "rx-a"): -7.0,
            },
            threshold_by_receiver={"rx-a": -10.0},
            global_threshold=-10.0,
            support_count=4,
            support_min=-14.0,
            support_median=-12.0,
            support_max=-10.0,
            support_quantile=0.90,
            temperature=1.0,
            risk_temperature=0.5,
            margin=0.0,
            alignment_policy="receiver_domain_ranked",
        )
        rows = [
            {
                "event_id": "old-ok",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "enpc_episode_negative_pressure": 0.20,
            },
            {
                "event_id": "unk-bad",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "enpc_episode_negative_pressure": 0.20,
            },
        ]

        out = augment_slev_evidence(rows, energy_bundle=bundle, energy_weight=0.5)

        self.assertEqual(out[0]["slev_threshold_scope"], "target_old_and_seen_new_support_only")
        self.assertLess(out[0]["slev_energy_risk"], out[1]["slev_energy_risk"])
        self.assertLess(out[0]["slev_combined_pressure"], out[1]["slev_combined_pressure"])

    def test_energy_bundle_uses_only_known_support_rows_for_threshold(self):
        from phase2_orbit_slev_ci_eval import build_slev_energy_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            roles = []
            tx_ids = []
            rx_ids = []
            day_ids = []
            sig_ids = []
            sat = []
            features = []
            logits = []
            for role, tx, base_logit in [
                ("target_old", "old-a", 10.0),
                ("target_new", "new-a", 9.0),
                ("target_unknown", "unk-a", 1.0),
            ]:
                for i in range(4):
                    roles.append(role)
                    tx_ids.append(tx)
                    rx_ids.append("rx-a")
                    day_ids.append("d0")
                    sig_ids.append(f"{role}-{i}")
                    sat.append("leo_clear_weak")
                    features.append([float(i), float(len(features))])
                    logits.append([base_logit, 0.0])
            np.savez(
                path,
                features=np.asarray(features, dtype=np.float32),
                tx_logits=np.asarray(logits, dtype=np.float32),
                raw_labels=np.asarray(tx_ids),
                domain_labels=np.asarray(rx_ids),
                tx_ids=np.asarray(tx_ids),
                rx_ids=np.asarray(rx_ids),
                day_ids=np.asarray(day_ids),
                eq_ids=np.asarray(["eq"] * len(tx_ids)),
                sig_ids=np.asarray(sig_ids),
                dataset_role=np.asarray(roles),
                channel_views=np.asarray(["leo"] * len(tx_ids)),
                sat_scenarios=np.asarray(sat),
            )

            bundle = build_slev_energy_bundle(
                path,
                k_shot=2,
                query_per_class=2,
                seed=1,
                support_selection_policy="stable_first",
                event_alignment_policy="receiver_domain_ranked",
                support_quantile=0.90,
                logit_temperature=1.0,
                risk_temperature=0.5,
                risk_margin=0.0,
            )

        self.assertEqual(bundle.support_count, 4)
        self.assertLess(bundle.global_threshold, -8.0)
        self.assertEqual(bundle.threshold_scope, "target_old_and_seen_new_support_only")


if __name__ == "__main__":
    unittest.main()
