import unittest

import torch


class PaperReproductionFeatureSeparationTest(unittest.TestCase):
    def test_feature_separation_loss_has_terms_and_gradients(self):
        from paper_reproduction.feature_separation_crossrx.losses import feature_separation_loss
        from paper_reproduction.feature_separation_crossrx.model import (
            FeatureSeparationNet,
            build_wisig_fusion_representation,
        )

        torch.manual_seed(7)
        model = FeatureSeparationNet(
            input_channels=3,
            input_length=256,
            embedding_dim=16,
            branch_dim=8,
            num_tx=3,
            num_rx=2,
        )
        batch = build_wisig_fusion_representation(torch.randn(5, 2, 256))
        tx_labels = torch.tensor([0, 1, 2, 1, 0])
        rx_labels = torch.tensor([0, 1, 0, 1, 0])

        outputs = model(batch)
        loss, terms = feature_separation_loss(outputs, tx_labels, rx_labels, lambda_correlation=0.25)
        self.assertEqual(set(terms), {"tx_ce", "rx_ce", "similarity", "tx_entropy", "rx_entropy", "total"})
        self.assertGreater(float(terms["tx_ce"]), 0.0)
        self.assertGreater(float(terms["rx_ce"]), 0.0)
        self.assertGreaterEqual(float(terms["similarity"]), 0.0)

        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0 for g in grads))

    def test_correlation_penalty_distinguishes_separated_and_correlated_features(self):
        from paper_reproduction.feature_separation_crossrx.losses import correlation_penalty

        separated_tx = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        separated_rx = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]])
        correlated_rx = separated_tx.clone()

        separated = correlation_penalty(separated_tx, separated_rx)
        correlated = correlation_penalty(separated_tx, correlated_rx)
        self.assertLess(float(separated), 1e-6)
        self.assertGreater(float(correlated), 0.9)

    def test_reproduction_documents_track_required_checklists(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "paper_reproduction"
        required = [
            root / "protonet_cda" / "paper_checklist.md",
            root / "feature_separation_crossrx" / "paper_checklist.md",
            root / "repro_gap.md",
        ]
        for path in required:
            text = path.read_text(encoding="utf-8")
            self.assertIn("paper-unspecified", text)
            self.assertIn("implementation choice", text)


if __name__ == "__main__":
    unittest.main()
