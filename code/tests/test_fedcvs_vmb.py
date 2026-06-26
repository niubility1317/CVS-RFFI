import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.fedcvs_vmb import (
    FedCVSCoralStatsBank,
    build_class_conditional_coral_stats,
    FedCVSVMBPrototypeBank,
    FedCVSVMBPrototypeStats,
    aggregate_gradients,
    apply_server_gradient_step,
    class_conditional_coral_loss,
    coral_stats_payload_size_bytes,
    domain_balanced_weights,
    gradient_payload_size_bytes,
    merge_coral_stats,
    prototype_contrastive_loss,
    prototype_stats_payload_size_bytes,
    select_domain_balanced_clients,
    select_transmitter_balanced_indices,
)
from evaluation.fedcvs_vmb_analysis import gradient_cosine_report, prototype_drift
from evaluation.fedcvs_vmb_probe import run_four_probes


class FedCVSVMBHelperTest(unittest.TestCase):
    def test_class_conditional_coral_stats_merge_and_loss(self):
        features_a = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
                [0.0, 3.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        stats_a = build_class_conditional_coral_stats(features_a, labels, num_classes=2, mode="diag")
        stats_b = build_class_conditional_coral_stats(features_a + 1.0, labels, num_classes=2, mode="diag")
        merged = merge_coral_stats([stats_a, stats_b])
        bank = FedCVSCoralStatsBank(num_classes=2, momentum=0.0, mode="diag")

        summary = bank.update(merged)
        loss, metrics = class_conditional_coral_loss(features_a, labels, bank.stats, min_count=2)

        self.assertEqual(summary["class_count_nonzero"], 2)
        self.assertGreater(coral_stats_payload_size_bytes(merged), 0)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(metrics["active_classes"], 2)
        self.assertGreater(metrics["mean_dist"], 0.0)
        self.assertGreaterEqual(metrics["cov_dist"], 0.0)

    def test_class_conditional_coral_loss_is_zero_against_matching_stats(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
                [0.0, 3.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        stats = build_class_conditional_coral_stats(features, labels, num_classes=2, mode="diag")

        loss, metrics = class_conditional_coral_loss(features, labels, stats, min_count=2)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(metrics["active_classes"], 2)

    def test_prototype_bank_updates_tx_and_rx_prototypes_with_normalized_ema(self):
        bank = FedCVSVMBPrototypeBank(num_classes=3, ema_alpha=0.5, clip_norm=1.0)
        stats = FedCVSVMBPrototypeStats(
            tx_sum=torch.tensor([[4.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
            tx_count=torch.tensor([4.0, 2.0, 0.0]),
            rx_sum_by_client={"rx0": torch.tensor([3.0, 0.0]), "rx1": torch.tensor([0.0, -2.0])},
            rx_count_by_client={"rx0": 3.0, "rx1": 2.0},
        )

        summary = bank.update(stats)

        self.assertEqual(summary["tx_count_nonzero"], 2)
        self.assertEqual(summary["rx_count_nonzero"], 2)
        self.assertTrue(torch.allclose(bank.tx_proto[0], torch.tensor([1.0, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(bank.rx_proto["rx1"], torch.tensor([0.0, -1.0]), atol=1e-6))

        stats2 = FedCVSVMBPrototypeStats(
            tx_sum=torch.tensor([[0.0, 4.0], [0.0, 2.0], [2.0, 0.0]]),
            tx_count=torch.tensor([4.0, 2.0, 2.0]),
            rx_sum_by_client={"rx0": torch.tensor([0.0, 3.0])},
            rx_count_by_client={"rx0": 3.0},
        )
        bank.update(stats2)

        expected = torch.nn.functional.normalize(torch.tensor([0.5, 0.5]), dim=0)
        self.assertTrue(torch.allclose(bank.tx_proto[0], expected, atol=1e-6))

    def test_prototype_contrastive_loss_uses_only_counted_prototypes(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
        labels = torch.tensor([0, 1, 0])
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        counts = torch.tensor([3.0, 2.0, 0.0])

        loss, metrics = prototype_contrastive_loss(
            features,
            labels,
            prototypes,
            counts,
            temperature=0.1,
            min_count=1,
        )

        self.assertLess(float(loss.item()), 0.25)
        self.assertEqual(metrics["active_prototypes"], 2)
        self.assertGreater(metrics["target_cos"], 0.9)

    def test_domain_balanced_weights_equalize_receiver_domains(self):
        weights = domain_balanced_weights(
            ["rx0_a", "rx0_b", "rx1"],
            {"rx0_a": "rx0", "rx0_b": "rx0", "rx1": "rx1"},
        )

        self.assertAlmostEqual(weights["rx0_a"], 0.25)
        self.assertAlmostEqual(weights["rx0_b"], 0.25)
        self.assertAlmostEqual(weights["rx1"], 0.5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_domain_balanced_client_sampling_spreads_across_domains(self):
        selected = select_domain_balanced_clients(
            ["rx0_a", "rx0_b", "rx1_a", "rx2_a"],
            {"rx0_a": "rx0", "rx0_b": "rx0", "rx1_a": "rx1", "rx2_a": "rx2"},
            clients_per_round=3,
            seed=13,
            round_idx=2,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            len({{"rx0_a": "rx0", "rx0_b": "rx0", "rx1_a": "rx1", "rx2_a": "rx2"}[cid] for cid in selected}),
            3,
        )

    def test_transmitter_balanced_indices_limit_majority_class_shortcut(self):
        metadata = {
            0: {"tx_id": 0},
            1: {"tx_id": 0},
            2: {"tx_id": 0},
            3: {"tx_id": 1},
            4: {"tx_id": 1},
            5: {"tx_id": 2},
        }

        selected = select_transmitter_balanced_indices(
            [0, 1, 2, 3, 4, 5],
            lambda idx: metadata[idx],
            batch_size=4,
            seed=5,
            round_idx=1,
            batch_idx=0,
        )

        counts = {}
        for idx in selected:
            tx = metadata[idx]["tx_id"]
            counts[tx] = counts.get(tx, 0) + 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_gradient_aggregation_and_server_step_use_domain_weights(self):
        gradients = {
            "a": {"w": torch.tensor([1.0, 3.0])},
            "b": {"w": torch.tensor([5.0, 7.0])},
        }
        weights = {"a": 0.25, "b": 0.75}

        aggregated = aggregate_gradients(gradients, weights)
        state = {"w": torch.tensor([10.0, 20.0])}
        new_state, opt_state, metrics = apply_server_gradient_step(
            state,
            aggregated,
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            optimizer_state={},
        )

        self.assertTrue(torch.allclose(aggregated["w"], torch.tensor([4.0, 6.0])))
        self.assertTrue(torch.allclose(new_state["w"], torch.tensor([9.6, 19.4])))
        self.assertIn("w", opt_state)
        self.assertGreater(metrics["grad_norm"], 0.0)
        self.assertEqual(gradient_payload_size_bytes({"w": torch.ones(2, dtype=torch.float32)}), 8)
        self.assertEqual(
            prototype_stats_payload_size_bytes(
                FedCVSVMBPrototypeStats(
                    tx_sum=torch.ones(2, 2),
                    tx_count=torch.ones(2),
                    rx_sum_by_client={"rx0": torch.ones(2)},
                    rx_count_by_client={"rx0": 2.0},
                )
            ),
            40,
        )

    def test_probe_helpers_report_required_four_disentanglement_metrics(self):
        z_t = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        z_r = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        tx = torch.tensor([0, 0, 1, 1, 0, 1])
        rx = torch.tensor([0, 1, 0, 1, 0, 1])

        report = run_four_probes(z_t=z_t, z_r=z_r, tx_labels=tx, rx_labels=rx, epochs=20, val_fraction=0.0)

        self.assertEqual(
            set(report),
            {"acc_y_given_zt", "acc_d_given_zt", "acc_d_given_zr", "acc_y_given_zr"},
        )
        self.assertGreaterEqual(report["acc_y_given_zt"]["acc"], 90.0)
        self.assertGreaterEqual(report["acc_d_given_zr"]["acc"], 90.0)

    def test_analysis_helpers_compute_prototype_drift_and_gradient_cosines(self):
        drift = prototype_drift(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        )
        grad_report = gradient_cosine_report(
            {
                "rx0": {"w": torch.tensor([1.0, 0.0])},
                "rx1": {"w": torch.tensor([0.0, 1.0])},
                "rx2": {"w": torch.tensor([1.0, 1.0])},
            }
        )

        self.assertEqual(drift["matched_prototypes"], 2)
        self.assertGreaterEqual(drift["max"], 0.0)
        self.assertEqual(grad_report["clients"], 3)
        self.assertEqual(grad_report["count"], 3)

    def test_recommended_config_preserves_project_hard_constraints(self):
        config_text = (ROOT / "configs" / "fedcvs_rffi_vmb.yaml").read_text(encoding="utf-8")

        self.assertIn("method: FedCVS-RFFI-VMB", config_text)
        self.assertIn("wisig_train_ratio: 0.1", config_text)
        self.assertIn("fl_rounds: 200", config_text)
        self.assertIn("epochs: 200", config_text)
        self.assertIn("fl_client_key: receiver", config_text)
        self.assertIn("default_on: true", config_text)
        self.assertIn("interval_rounds: 0", config_text)
        self.assertIn("last_n_rounds_every_round: 0", config_text)
        self.assertIn("final_offset_rounds_from_end: [5, 3, 1]", config_text)
        self.assertIn("resolved_default_for_200_rounds: [196, 198, 200]", config_text)
        self.assertIn("stage: auto", config_text)
        self.assertIn("pretrain_rounds_when_auto: 20", config_text)
        self.assertIn("A0: FedAvg baseline", config_text)
        self.assertIn("A14: z_t classifier versus [z_t,z_r] classifier shortcut check", config_text)


if __name__ == "__main__":
    unittest.main()
