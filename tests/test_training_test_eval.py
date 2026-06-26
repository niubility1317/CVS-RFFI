import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class TrainingTestEvalTest(unittest.TestCase):
    def test_should_run_training_test_matches_baseline_gate(self):
        from training_test_eval import should_run_training_test

        self.assertTrue(should_run_training_test("every_epoch", epoch=3, epochs=10, val_improved=False))
        self.assertTrue(should_run_training_test("val_improved_final", epoch=3, epochs=10, val_improved=True))
        self.assertFalse(should_run_training_test("val_improved_final", epoch=3, epochs=10, val_improved=False))
        self.assertTrue(should_run_training_test("val_improved_final", epoch=10, epochs=10, val_improved=False))

    def test_evaluate_training_tests_returns_val_aggregate_and_formatted_lines(self):
        from training_test_eval import evaluate_training_tests

        calls = []

        def eval_loader(model, loader, device, domain_label_map, max_batches=0):
            calls.append(("val", loader, max_batches))
            return {"tx_acc": 98.0, "dom_acc": 12.0, "tx_correct": 98, "tx_total": 100}

        def eval_named(model, named_loaders, device, domain_label_map, max_batches=0):
            calls.append(("named", tuple(named_loaders.keys()), max_batches))
            return {
                "test_unseen_day_seen_rx": {"tx_acc": 90.0, "tx_correct": 9, "tx_total": 10},
                "test_seen_day_unseen_rx": {"tx_acc": 70.0, "tx_correct": 7, "tx_total": 10},
                "test_unseen_day_unseen_rx": {"tx_acc": 40.0, "tx_correct": 4, "tx_total": 10},
                "test_rx_7": {"tx_acc": 50.0, "tx_correct": 5, "tx_total": 10},
            }

        result = evaluate_training_tests(
            model=object(),
            val_loader="val_loader",
            named_test_loaders={"test_unseen_day_unseen_rx": object()},
            device="cpu",
            domain_label_map={},
            named_test_meta={
                "test_unseen_day_seen_rx": {"days_label": ["d2"], "rxs_idx": [0]},
                "test_seen_day_unseen_rx": {"days_label": ["d0"], "rxs_idx": [7]},
                "test_unseen_day_unseen_rx": {"days_label": ["d2"], "rxs_idx": [7]},
                "test_rx_7": {"days_label": ["d0"], "rxs_idx": [7]},
            },
            dataset="wisig",
            max_batches=2,
            evaluate_loader_fn=eval_loader,
            evaluate_named_loaders_fn=eval_named,
        )

        self.assertEqual(calls, [("val", "val_loader", 2), ("named", ("test_unseen_day_unseen_rx",), 2)])
        self.assertEqual(result.test_stats["tx_correct"], 20)
        self.assertEqual(result.test_stats["tx_total"], 30)
        self.assertAlmostEqual(result.test_stats["tx_acc"], 100.0 * 20 / 30)
        self.assertEqual(result.lines[0], "[TEST]  overall_tx=66.67% (20/30)")
        self.assertEqual(result.lines[1], "[TEST-SPLIT]")
        self.assertIn("unseen_day_unseen_rx(days=['d2'], rxs=[7])", "\n".join(result.lines))


if __name__ == "__main__":
    unittest.main()
