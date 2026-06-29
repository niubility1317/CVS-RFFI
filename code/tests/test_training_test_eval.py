import unittest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class TrainingTestEvalPolicyTest(unittest.TestCase):
    def test_start_epoch_delays_every_epoch_policy(self):
        from training_test_eval import should_run_training_test

        self.assertFalse(
            should_run_training_test(
                "every_epoch",
                epoch=150,
                epochs=170,
                val_improved=True,
                start_epoch=151,
            )
        )
        self.assertTrue(
            should_run_training_test(
                "every_epoch",
                epoch=151,
                epochs=170,
                val_improved=False,
                start_epoch=151,
            )
        )

    def test_start_epoch_delays_val_improved_final_policy(self):
        from training_test_eval import should_run_training_test

        self.assertFalse(
            should_run_training_test(
                "val_improved_final",
                epoch=88,
                epochs=170,
                val_improved=True,
                start_epoch=151,
            )
        )
        self.assertTrue(
            should_run_training_test(
                "val_improved_final",
                epoch=170,
                epochs=170,
                val_improved=False,
                start_epoch=151,
            )
        )

    def test_interval_final_policy_runs_only_interval_and_final(self):
        from training_test_eval import should_run_training_test

        self.assertFalse(
            should_run_training_test(
                "interval_final",
                epoch=9,
                epochs=170,
                val_improved=True,
                start_epoch=1,
                interval=10,
            )
        )
        self.assertTrue(
            should_run_training_test(
                "interval_final",
                epoch=10,
                epochs=170,
                val_improved=False,
                start_epoch=1,
                interval=10,
            )
        )
        self.assertFalse(
            should_run_training_test(
                "interval_final",
                epoch=81,
                epochs=170,
                val_improved=True,
                start_epoch=1,
                interval=10,
            )
        )
        self.assertTrue(
            should_run_training_test(
                "interval_final",
                epoch=170,
                epochs=170,
                val_improved=False,
                start_epoch=1,
                interval=10,
            )
        )

    def test_interval_final_cen31_170_epochs_runs_17_heavy_tests(self):
        from training_test_eval import should_run_training_test

        epochs = [
            epoch
            for epoch in range(1, 171)
            if should_run_training_test(
                "interval_final",
                epoch=epoch,
                epochs=170,
                val_improved=True,
                start_epoch=1,
                interval=10,
            )
        ]

        self.assertEqual(epochs, list(range(10, 171, 10)))
        self.assertEqual(len(epochs), 17)

    def test_interval_final_dense_tail_runs_10_then_last20_every2(self):
        from training_test_eval import should_run_training_test

        epochs = [
            epoch
            for epoch in range(1, 201)
            if should_run_training_test(
                "interval_final",
                epoch=epoch,
                epochs=200,
                val_improved=False,
                start_epoch=1,
                interval=10,
                final_window=20,
                final_interval=2,
            )
        ]

        self.assertEqual(epochs, list(range(10, 181, 10)) + list(range(182, 201, 2)))
        self.assertEqual(len(epochs), 28)

    def test_interval_final_dense_tail_always_runs_final_epoch(self):
        from training_test_eval import should_run_training_test

        self.assertTrue(
            should_run_training_test(
                "interval_final",
                epoch=199,
                epochs=199,
                val_improved=False,
                start_epoch=1,
                interval=10,
                final_window=20,
                final_interval=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
