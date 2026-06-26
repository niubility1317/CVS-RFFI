import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.fed_fishr import (  # noqa: E402
    build_fed_fishr_stats,
    fed_fishr_reweight,
    fed_fishr_target_loss,
    merge_fed_fishr_client_stats,
)


class FedFishrStatsTest(unittest.TestCase):
    def _single_domain_batch(self, shift: float):
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        logits = torch.tensor(
            [
                [2.0 + shift, -0.5],
                [1.6 + shift, -0.1],
                [-0.4, 1.9 - shift],
                [0.2, 1.3 - shift],
            ],
            dtype=torch.float32,
        )
        features = torch.tensor(
            [
                [1.0 + shift, 0.0, 0.2],
                [0.8 + shift, 0.1, 0.1],
                [0.0, 1.0 - shift, 0.3],
                [0.2, 0.8 - shift, 0.4],
            ],
            dtype=torch.float32,
        )
        return logits, labels, features

    def test_merge_fed_fishr_stats_activates_from_multiple_single_domain_clients(self):
        logits0, labels0, features0 = self._single_domain_batch(0.0)
        logits1, labels1, features1 = self._single_domain_batch(0.35)
        stats0 = build_fed_fishr_stats(
            logits0,
            labels0,
            features0,
            num_classes=2,
            scope="classifier_head",
            min_count=2,
        )
        stats1 = build_fed_fishr_stats(
            logits1,
            labels1,
            features1,
            num_classes=2,
            scope="classifier_head",
            min_count=2,
        )

        merged = merge_fed_fishr_client_stats(
            {"rx0": stats0, "rx1": stats1},
            min_clients=2,
            min_count=2,
            momentum=0.0,
        )

        self.assertTrue(merged["enabled"])
        self.assertTrue(merged["active"])
        self.assertEqual(merged["active_classes"], 2)
        self.assertEqual(merged["client_count"], 2)
        self.assertEqual(tuple(merged["target_var"].shape), tuple(stats0["var"].shape))
        self.assertEqual(set(merged["client_mismatch"].keys()), {"rx0", "rx1"})
        self.assertGreaterEqual(merged["payload_bytes"], stats0["payload_bytes"] + stats1["payload_bytes"])
        self.assertTrue(math.isfinite(float(merged["mismatch_mean"])))

    def test_reweight_downweights_high_fed_fishr_mismatch_without_breaking_simplex(self):
        weights, summary = fed_fishr_reweight(
            {"rx0": 0.5, "rx1": 0.5},
            {"rx0": 0.0, "rx1": 3.0},
            alpha=0.6,
            floor=0.10,
            cap=0.90,
        )

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["rx0"], weights["rx1"])
        self.assertGreaterEqual(min(weights.values()), 0.10 - 1e-6)
        self.assertLessEqual(max(weights.values()), 0.90 + 1e-6)
        self.assertTrue(summary["active"])
        self.assertGreater(summary["max_delta"], 0.0)

    def test_target_loss_is_differentiable_against_server_variance_target(self):
        logits0, labels0, features0 = self._single_domain_batch(0.0)
        logits1, labels1, features1 = self._single_domain_batch(0.35)
        merged = merge_fed_fishr_client_stats(
            {
                "rx0": build_fed_fishr_stats(logits0, labels0, features0, num_classes=2, min_count=2),
                "rx1": build_fed_fishr_stats(logits1, labels1, features1, num_classes=2, min_count=2),
            },
            min_clients=2,
            min_count=2,
        )
        logits = logits0.clone().requires_grad_(True)
        features = features0.clone().requires_grad_(True)

        loss, diag = fed_fishr_target_loss(
            logits,
            labels0,
            features,
            target_var=merged["target_var"],
            target_mask=merged["target_mask"],
            min_count=2,
        )

        self.assertGreaterEqual(float(loss.detach().item()), 0.0)
        self.assertEqual(diag["active_classes"], 2)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
