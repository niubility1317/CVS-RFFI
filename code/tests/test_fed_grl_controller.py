import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FedCGRLControllerTest(unittest.TestCase):
    def test_disabled_controller_keeps_base_lambda(self):
        from federated.fed_grl_controller import FedCGRLController

        cfg = SimpleNamespace(use_fed_cgrl=False, lambda_rx_adv=1.25)
        controller = FedCGRLController.from_config(cfg)

        decision = controller.lambda_for_client("rx0", round_idx=10)

        self.assertFalse(controller.enabled)
        self.assertAlmostEqual(decision.lambda_rx_adv, 1.25)
        self.assertAlmostEqual(decision.warmup_gate, 1.0)
        self.assertAlmostEqual(decision.leak_gate, 1.0)
        self.assertAlmostEqual(decision.tx_gate, 1.0)
        self.assertAlmostEqual(decision.conflict_gate, 1.0)

    def test_warmup_caps_initial_dynamic_lambda(self):
        from federated.fed_grl_controller import FedCGRLController

        cfg = SimpleNamespace(
            use_fed_cgrl=True,
            lambda_rx_adv=1.0,
            fed_cgrl_warmup_rounds=4,
            fed_cgrl_min_lambda=0.05,
            fed_cgrl_max_lambda=2.0,
        )
        controller = FedCGRLController.from_config(cfg)

        r1 = controller.lambda_for_client("rx0", round_idx=1)
        r4 = controller.lambda_for_client("rx0", round_idx=4)

        self.assertAlmostEqual(r1.lambda_rx_adv, 0.25)
        self.assertAlmostEqual(r1.warmup_gate, 0.25)
        self.assertAlmostEqual(r4.lambda_rx_adv, 1.0)
        self.assertAlmostEqual(r4.warmup_gate, 1.0)

    def test_leakage_increases_and_tx_loss_reduces_next_round_lambda(self):
        from federated.fed_grl_controller import FedCGRLController

        cfg = SimpleNamespace(
            use_fed_cgrl=True,
            lambda_rx_adv=1.0,
            fed_cgrl_warmup_rounds=0,
            fed_cgrl_leak_target_acc=20.0,
            fed_cgrl_leak_gain=1.0,
            fed_cgrl_tx_loss_guard=4.0,
            fed_cgrl_tx_loss_gate_min=0.40,
            fed_cgrl_conflict_gate_min=0.35,
            fed_cgrl_min_lambda=0.05,
            fed_cgrl_max_lambda=2.0,
            fed_cgrl_ema=1.0,
        )
        controller = FedCGRLController.from_config(cfg)
        controller.update_after_round(
            {
                "rx0": {"seen": 16, "grl_target_acc": 60.0, "loss_cls": 2.0},
                "rx1": {"seen": 16, "grl_target_acc": 60.0, "loss_cls": 8.0},
            },
            round_idx=1,
        )

        strong = controller.lambda_for_client("rx0", round_idx=2)
        guarded = controller.lambda_for_client("rx1", round_idx=2)

        self.assertGreater(strong.lambda_rx_adv, 1.0)
        self.assertAlmostEqual(strong.leak_gate, 2.0)
        self.assertAlmostEqual(strong.tx_gate, 1.0)
        self.assertLess(guarded.lambda_rx_adv, strong.lambda_rx_adv)
        self.assertAlmostEqual(guarded.tx_gate, 0.5)

    def test_p90_leakage_reference_changes_next_round_lambda(self):
        from federated.fed_grl_controller import FedCGRLController

        cfg = SimpleNamespace(
            use_fed_cgrl=True,
            lambda_rx_adv=1.0,
            fed_cgrl_warmup_rounds=0,
            fed_cgrl_leak_target_acc=20.0,
            fed_cgrl_leak_gain=1.0,
            fed_cgrl_leak_gate_min=0.50,
            fed_cgrl_leak_gate_max=3.0,
            fed_cgrl_min_lambda=0.05,
            fed_cgrl_max_lambda=3.0,
            fed_cgrl_ema=1.0,
            fed_cgrl_leak_stat="p90",
        )
        controller = FedCGRLController.from_config(cfg)
        controller.update_after_round(
            {
                "rx0": {"seen": 16, "grl_target_acc": 10.0, "loss_cls": 1.0},
                "rx6": {"seen": 16, "grl_target_acc": 90.0, "loss_cls": 1.0},
            },
            round_idx=1,
        )

        decision = controller.lambda_for_client("rx0", round_idx=2)
        metrics = decision.as_metrics()

        self.assertGreater(decision.leak_gate, 1.0)
        self.assertGreater(decision.lambda_rx_adv, 1.0)
        self.assertGreater(metrics["fed_cgrl_leak_reference_acc"], 80.0)
        self.assertEqual(metrics["fed_cgrl_leak_stat"], "p90")

    def test_tx_guard_release_restores_gate_after_ramp(self):
        from federated.fed_grl_controller import FedCGRLController

        cfg = SimpleNamespace(
            use_fed_cgrl=True,
            lambda_rx_adv=1.0,
            fed_cgrl_warmup_rounds=0,
            fed_cgrl_tx_loss_guard=4.0,
            fed_cgrl_tx_loss_gate_min=0.25,
            fed_cgrl_tx_guard_release_rounds=2,
            fed_cgrl_ema=1.0,
        )
        controller = FedCGRLController.from_config(cfg)
        controller.update_after_round(
            {"rx0": {"seen": 8, "grl_target_acc": 20.0, "loss_cls": 8.0}},
            round_idx=1,
        )

        early = controller.lambda_for_client("rx0", round_idx=1)
        released = controller.lambda_for_client("rx0", round_idx=4)

        self.assertAlmostEqual(early.tx_gate, 0.5)
        self.assertAlmostEqual(released.tx_gate, 1.0)
        self.assertAlmostEqual(released.as_metrics()["fed_cgrl_tx_guard_release"], 1.0)

    def test_conflict_summary_reduces_all_clients(self):
        from federated.fed_grl_controller import FedCGRLController

        cfg = SimpleNamespace(
            use_fed_cgrl=True,
            lambda_rx_adv=1.0,
            fed_cgrl_warmup_rounds=0,
            fed_cgrl_conflict_threshold=-0.10,
            fed_cgrl_conflict_gate_min=0.30,
            fed_cgrl_ema=1.0,
        )
        controller = FedCGRLController.from_config(cfg)
        controller.update_after_round(
            {"rx0": {"seen": 4, "grl_target_acc": 20.0, "loss_cls": 1.0}},
            round_idx=1,
            conflict_summary={"grad_cos_min_before": -0.60},
        )

        decision = controller.lambda_for_client("rx0", round_idx=2)
        summary = controller.round_summary()

        self.assertLess(decision.lambda_rx_adv, 1.0)
        self.assertAlmostEqual(decision.conflict_gate, 0.30)
        self.assertEqual(summary["conflict_source"], "unknown")
        self.assertEqual(summary["conflict_signal_available"], 1.0)

    def test_round_summary_reports_weighted_average_and_client_state(self):
        from federated.fed_grl_controller import FedCGRLController

        controller = FedCGRLController.from_config(
            SimpleNamespace(use_fed_cgrl=True, lambda_rx_adv=0.5, fed_cgrl_warmup_rounds=0, fed_cgrl_ema=1.0)
        )
        decisions = {
            "rx0": controller.lambda_for_client("rx0", round_idx=1),
            "rx1": controller.lambda_for_client("rx1", round_idx=1),
        }
        controller.update_after_round(
            {
                "rx0": {"seen": 1, "fed_cgrl_lambda_rx_adv": decisions["rx0"].lambda_rx_adv, "grl_target_acc": 50.0},
                "rx1": {"seen": 3, "fed_cgrl_lambda_rx_adv": decisions["rx1"].lambda_rx_adv, "grl_target_acc": 10.0},
            },
            round_idx=1,
        )

        summary = controller.round_summary()

        self.assertTrue(summary["enabled"])
        self.assertAlmostEqual(summary["lambda_rx_adv_avg"], 0.5)
        self.assertEqual(summary["client_count"], 2)
        self.assertIn("rx0", summary["clients"])
        self.assertTrue(math.isfinite(summary["grl_target_acc_avg"]))
        self.assertAlmostEqual(summary["grl_target_acc_min"], 10.0)
        self.assertAlmostEqual(summary["grl_target_acc_max"], 50.0)
        self.assertGreater(summary["grl_target_acc_p90"], 45.0)
        self.assertEqual(summary["grl_target_acc_worst_client"], "rx0")
        self.assertAlmostEqual(summary["lambda_rx_adv_min"], 0.5)
        self.assertAlmostEqual(summary["lambda_rx_adv_max"], 0.5)
        self.assertAlmostEqual(summary["lambda_rx_adv_p90"], 0.5)


if __name__ == "__main__":
    unittest.main()
