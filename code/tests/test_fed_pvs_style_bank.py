import sys
import unittest
from pathlib import Path
import math

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FedPvsStyleBankTest(unittest.TestCase):
    def test_rf_style_extractor_returns_class_marginalized_packet(self):
        from federated.rf_style_extractor import RFStyleExtractor

        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0, 1.5]],
                [[2.0, 3.0, 4.0, 5.0], [1.0, 1.5, 2.0, 2.5]],
            ]
        )
        y = torch.tensor([0, 1])

        packet = RFStyleExtractor().extract(x, y, client_id="rx0", round_idx=3)
        data = packet.to_dict()

        self.assertEqual(packet.client_id, "rx0")
        self.assertEqual(packet.round_idx, 3)
        self.assertEqual(packet.count, 2)
        self.assertIn("iq_mean", data["stats"])
        self.assertIn("amp_mean", data["stats"])
        self.assertIn("spectrum_centroid", data["stats"])
        self.assertNotIn("class_0_mean", data["stats"])
        self.assertNotIn("labels", data)
        self.assertGreater(packet.size_bytes(), 0)

    def test_rf_style_extractor_emits_physical_receiver_chain_stats(self):
        from federated.rf_style_extractor import RFStyleExtractor

        n = torch.arange(16, dtype=torch.float32)
        phase = 2.0 * torch.pi * 0.0625 * n
        x = torch.stack([torch.cos(phase), torch.sin(phase)], dim=0).unsqueeze(0)

        packet = RFStyleExtractor(sample_rate_hz=16_000.0).extract(x, client_id="rx0", round_idx=1)

        for key in (
            "phys_cfo_cycles_per_sample",
            "phys_cfo_hz",
            "phys_phase_noise_std",
            "phys_iq_gain_imbalance_db",
            "phys_iq_phase_imbalance_deg",
            "phys_agc_gain_db",
            "phys_softclip_level",
            "phys_multipath_strength",
            "phys_lowpass_cutoff_frac",
        ):
            self.assertIn(key, packet.stats)
        self.assertAlmostEqual(float(packet.stats["phys_cfo_cycles_per_sample"]), 0.0625, places=3)
        self.assertAlmostEqual(float(packet.stats["phys_cfo_hz"]), 1000.0, delta=5.0)

    def test_rf_style_extractor_balances_present_classes_before_averaging(self):
        from federated.rf_style_extractor import RFStyleExtractor

        def make_batch(labels):
            xs = []
            n = torch.arange(64, dtype=torch.float32)
            for label in labels:
                freq = 0.06 + 0.02 * int(label)
                amp = 1.0 + 0.40 * int(label)
                phase = 2.0 * math.pi * freq * n
                xs.append(torch.stack([amp * torch.cos(phase), amp * torch.sin(phase)], dim=0))
            return torch.stack(xs, dim=0)

        balanced_y = torch.tensor([0, 1] * 8)
        skewed_y = torch.tensor([0] * 14 + [1] * 2)
        extractor = RFStyleExtractor(sample_rate_hz=1_000_000.0)

        balanced = extractor.extract(make_batch(balanced_y), balanced_y, client_id="rx0_balanced", round_idx=1)
        skewed = extractor.extract(make_batch(skewed_y), skewed_y, client_id="rx0_skewed", round_idx=1)

        self.assertAlmostEqual(float(balanced.stats["amp_mean"]), float(skewed.stats["amp_mean"]), places=5)
        self.assertAlmostEqual(
            float(balanced.stats["phys_cfo_cycles_per_sample"]),
            float(skewed.stats["phys_cfo_cycles_per_sample"]),
            places=5,
        )
        self.assertEqual(balanced.metadata["style_class_balance"], "equal_present_classes")
        self.assertEqual(skewed.metadata["style_num_classes"], 2)

    def test_style_bank_keeps_ema_centroids_and_excludes_same_client_sampling(self):
        from federated.rf_style_extractor import RFStyleExtractor
        from federated.style_bank import FederatedStyleBank

        extractor = RFStyleExtractor()
        packet_a = extractor.extract(torch.ones(2, 2, 8), torch.tensor([0, 1]), client_id="rx0", round_idx=1)
        packet_b = extractor.extract(torch.full((2, 2, 8), 3.0), torch.tensor([0, 1]), client_id="rx1", round_idx=1)
        bank = FederatedStyleBank(momentum=0.5, max_centroids=4)

        summary = bank.update([packet_a, packet_b])
        remote = bank.sample_remote_style(exclude_client_id="rx0")

        self.assertEqual(summary["num_centroids"], 2)
        self.assertEqual(summary["num_packets_seen"], 2)
        self.assertIsNotNone(remote)
        self.assertEqual(remote.client_id, "rx1")
        self.assertGreater(bank.size_bytes(), 0)
        self.assertGreaterEqual(bank.diagnostics()["mean_pairwise_l2"], 0.0)

    def test_style_bank_handles_schema_drift_and_keeps_recent_centroids(self):
        from federated.style_bank import FederatedStyleBank
        from federated.style_packet import StylePacket

        bank = FederatedStyleBank(max_centroids=2)
        bank.update(
            [
                StylePacket(client_id="old", round_idx=1, count=1, stats={"iq_mean": 0.0}),
                StylePacket(client_id="mid", round_idx=2, count=2, stats={"iq_mean": 2.0, "feature_std": 0.5}),
                StylePacket(client_id="new", round_idx=3, count=3, stats={"iq_mean": 3.0, "spectrum_centroid": 0.25}),
            ]
        )

        clients = {c.client_id for c in bank.centroids}

        self.assertEqual(len(bank.centroids), 2)
        self.assertNotIn("old", clients)
        self.assertEqual(len({int(c.vector.numel()) for c in bank.centroids}), 1)
        self.assertGreaterEqual(bank.diagnostics()["mean_pairwise_l2"], 0.0)

    def test_style_bank_normalizes_physical_stats_before_l2_distance(self):
        from federated.style_bank import FederatedStyleBank
        from federated.style_packet import StylePacket

        bank = FederatedStyleBank(max_centroids=4)
        bank.update(
            [
                StylePacket(client_id="rx0", round_idx=1, count=4, stats={"phys_cfo_hz": 35000.0}),
                StylePacket(client_id="rx1", round_idx=1, count=4, stats={"phys_iq_gain_imbalance_db": 3.0}),
            ]
        )

        max_abs = max(float(c.vector.abs().max().item()) for c in bank.centroids)

        self.assertLessEqual(max_abs, 1.0)

    def test_style_bank_preserves_target_domain_metadata_and_does_not_merge_different_targets(self):
        from federated.style_bank import FederatedStyleBank
        from federated.style_packet import StylePacket

        bank = FederatedStyleBank(max_centroids=4, merge_radius=999.0)
        bank.update(
            [
                StylePacket(
                    client_id="rx1_day0",
                    round_idx=1,
                    count=4,
                    stats={"iq_rms": 1.0},
                    metadata={"target_domain_label": 3, "raw_target_domain_label": 13},
                ),
                StylePacket(
                    client_id="rx2_day0",
                    round_idx=1,
                    count=4,
                    stats={"iq_rms": 1.01},
                    metadata={"target_domain_label": 4, "raw_target_domain_label": 14},
                ),
            ]
        )

        packets = bank.sample_remote_styles(exclude_client_id="rx0_day0", k=4)
        targets = sorted(int(p.metadata["target_domain_label"]) for p in packets)

        self.assertEqual(len(bank.centroids), 2)
        self.assertEqual(targets, [3, 4])

    def test_style_bank_target_balanced_sampling_spreads_receiver_targets(self):
        from federated.style_bank import FederatedStyleBank
        from federated.style_packet import StylePacket

        bank = FederatedStyleBank(max_centroids=8)
        bank.update(
            [
                StylePacket(client_id="rx1_a", round_idx=1, count=20, stats={"iq_rms": 1.00}, metadata={"target_domain_label": 1}),
                StylePacket(client_id="rx1_b", round_idx=1, count=18, stats={"iq_rms": 1.01}, metadata={"target_domain_label": 1}),
                StylePacket(client_id="rx2", round_idx=1, count=5, stats={"iq_rms": 1.20}, metadata={"target_domain_label": 2}),
                StylePacket(client_id="rx3", round_idx=1, count=4, stats={"iq_rms": 0.80}, metadata={"target_domain_label": 3}),
            ]
        )

        packets = bank.sample_remote_styles(exclude_client_id="rx0", k=3, policy="target_balanced")
        targets = {int(packet.metadata["target_domain_label"]) for packet in packets}

        self.assertEqual(len(packets), 3)
        self.assertEqual(targets, {1, 2, 3})

    def test_virtual_domain_sampler_builds_d_style_without_mutating_raw_domain(self):
        from federated.virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView

        x = torch.zeros(2, 2, 4)
        y = torch.tensor([0, 1])
        d_raw = torch.tensor([7, 8])
        sampler = VirtualDomainSampler(clean_style_id=0)
        view = VirtualStyleView(x=x + 1.0, source="remote_style", style_id=5)

        batch = sampler.build_batch(x, y, d_raw, [view])

        self.assertEqual(batch.x.shape[0], 4)
        self.assertTrue(torch.equal(batch.y, torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(batch.d_raw, torch.tensor([7, 8, 7, 8])))
        self.assertTrue(torch.equal(batch.d_style, torch.tensor([0, 0, 1, 1])))
        self.assertEqual(batch.sources, ("clean", "remote_style"))
        self.assertEqual(batch.metadata["raw_style_ids"], (0, 5))

    def test_style_conditioned_receiver_dg_uses_style_stats_conservatively(self):
        from federated.conditioned_receiver_dg import StyleConditionedReceiverDG
        from federated.style_packet import StylePacket

        x = torch.ones(2, 2, 4)
        style = StylePacket(
            client_id="rx1",
            round_idx=1,
            count=4,
            stats={"iq_rms": 1.2, "amp_std": 0.0, "phase_diff_mean": 0.0},
        )
        transform = StyleConditionedReceiverDG(max_gain_delta=0.10, max_noise_std=0.0)

        out = transform.transform(x, style)

        self.assertTrue(torch.allclose(out, torch.full_like(x, 1.1)))
        self.assertEqual(out.dtype, x.dtype)

    def test_style_conditioned_receiver_dg_applies_physical_cfo(self):
        from federated.conditioned_receiver_dg import StyleConditionedReceiverDG
        from federated.style_packet import StylePacket

        x = torch.zeros(1, 2, 4)
        x[:, 0, :] = 1.0
        style = StylePacket(
            client_id="rx1",
            round_idx=1,
            count=4,
            stats={
                "phys_cfo_hz": 250.0,
                "phys_sro_ppm": 0.0,
                "phys_agc_gain_db": 0.0,
                "phys_softclip_level": 10.0,
                "phys_iq_gain_imbalance_db": 0.0,
                "phys_iq_phase_imbalance_deg": 0.0,
                "phys_phase_noise_std": 0.0,
                "phys_awgn_snr_db": 120.0,
                "phys_multipath_strength": 0.0,
                "phys_lowpass_cutoff_frac": 1.0,
            },
        )
        transform = StyleConditionedReceiverDG(
            sample_rate_hz=1000.0,
            max_gain_delta=0.0,
            max_noise_std=0.0,
            p_lowpass=0.0,
            p_multipath=0.0,
        )

        out = transform.transform(x, style)

        expected = torch.tensor(
            [[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]],
            dtype=x.dtype,
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
