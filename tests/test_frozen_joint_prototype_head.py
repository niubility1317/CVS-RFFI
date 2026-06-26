import math
import unittest

import torch
import torch.nn.functional as F

from FJMP.frozen_joint_prototype_head import (
    CalibratedFusion,
    ConfidenceGate,
    FrozenJointPrototypeClassifier,
    MultiPrototypeHead,
    apply_zdom_mode,
    compute_ece,
    delta_regularization_loss,
    extract_frozen_features,
    init_prototypes_by_class_domain,
    true_class_usage_loss,
)


class FrozenJointPrototypeHeadTest(unittest.TestCase):
    def test_classifier_detaches_backbone_features_and_returns_class_logits(self):
        torch.manual_seed(7)
        model = FrozenJointPrototypeClassifier(
            id_dim=5,
            dom_dim=3,
            num_classes=4,
            num_prototypes=3,
            proto_dim=6,
            hidden_dim=8,
            dom_drop_prob=0.0,
        )
        z_id_raw = torch.randn(2, 5, requires_grad=True)
        z_dom = torch.randn(2, 3, requires_grad=True)

        out = model(z_id_raw, z_dom)
        loss = F.cross_entropy(out["logits"], torch.tensor([0, 2]))
        loss.backward()

        self.assertEqual(out["logits"].shape, (2, 4))
        self.assertEqual(out["proto_scores"].shape, (2, 4, 3))
        self.assertIsNone(z_id_raw.grad)
        self.assertIsNone(z_dom.grad)
        self.assertTrue(torch.allclose(out["z_joint"].norm(dim=1), torch.ones(2), atol=1e-5))

    def test_multi_prototype_head_uses_logsumexp_over_prototypes(self):
        head = MultiPrototypeHead(num_classes=2, feat_dim=2, num_prototypes=2, init_scale=4.0)
        with torch.no_grad():
            head.prototypes.copy_(
                torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[-1.0, 0.0], [0.0, -1.0]],
                    ]
                )
            )
        out = head(torch.tensor([[1.0, 0.0]]))
        expected_c0 = torch.logsumexp(torch.tensor([4.0, 0.0]) - math.log(2), dim=0)
        expected_c1 = torch.logsumexp(torch.tensor([-4.0, 0.0]) - math.log(2), dim=0)

        self.assertEqual(out["logits"].shape, (1, 2))
        self.assertTrue(torch.allclose(out["logits"][0], torch.stack([expected_c0, expected_c1]), atol=1e-5))

    def test_usage_loss_uses_only_true_class_assignments(self):
        scores = torch.full((3, 2, 3), -10.0)
        y = torch.tensor([0, 0, 1])
        scores[0, 0] = torch.tensor([10.0, 0.0, 0.0])
        scores[1, 0] = torch.tensor([10.0, 0.0, 0.0])
        scores[2, 1] = torch.tensor([0.0, 10.0, 0.0])
        for i, yi in enumerate(y.tolist()):
            scores[i, 1 - yi] = 50.0

        loss = true_class_usage_loss(scores, y, num_classes=2, min_usage=0.20, max_usage=0.70)

        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_usage_loss_accepts_half_scores_from_amp(self):
        scores = torch.full((3, 2, 3), -10.0, dtype=torch.float16)
        y = torch.tensor([0, 0, 1])
        scores[0, 0] = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float16)
        scores[1, 0] = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float16)
        scores[2, 1] = torch.tensor([0.0, 10.0, 0.0], dtype=torch.float16)

        loss = true_class_usage_loss(scores, y, num_classes=2)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))

    def test_zdom_modes_cover_shortcut_ablations(self):
        torch.manual_seed(3)
        z_dom = torch.arange(12, dtype=torch.float32).view(4, 3)
        mean = z_dom.mean(dim=0)

        self.assertTrue(torch.equal(apply_zdom_mode(z_dom, "zero"), torch.zeros_like(z_dom)))
        self.assertTrue(torch.equal(apply_zdom_mode(z_dom, "mean", zdom_mean=mean), mean.expand_as(z_dom)))
        shuffled = apply_zdom_mode(z_dom, "shuffled")
        random_source = apply_zdom_mode(z_dom, "random_source")

        self.assertEqual(shuffled.shape, z_dom.shape)
        self.assertEqual(random_source.shape, z_dom.shape)
        self.assertTrue(torch.equal(torch.sort(shuffled[:, 0]).values, torch.sort(z_dom[:, 0]).values))

    def test_center_initialization_and_fusion_gate_diagnostics(self):
        head = MultiPrototypeHead(num_classes=2, feat_dim=2, num_prototypes=2)
        z_joint = F.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32),
            dim=1,
        )
        y = torch.tensor([0, 0, 1, 1])
        domain = torch.tensor([0, 1, 0, 1])

        init_prototypes_by_class_domain(head, z_joint, y, domain, num_domains=2)

        self.assertEqual(head.prototypes.shape, (2, 2, 2))
        self.assertTrue(torch.allclose(head.prototypes.norm(dim=-1), torch.ones(2, 2), atol=1e-5))

        fusion = CalibratedFusion(alpha=0.5, beta=1.2, mode="calibrated_logit")
        base_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        proto_logits = torch.tensor([[0.5, 0.0], [2.0, -1.0]])
        fused = fusion(base_logits=base_logits, proto_logits=proto_logits)
        self.assertEqual(fused["logits"].shape, (2, 2))

        gate = ConfidenceGate(threshold_score=-1.0, threshold_margin=-1.0, threshold_entropy=10.0)
        accepted = gate(proto_logits=proto_logits, proto_scores=torch.randn(2, 2, 2), nearest_proto_score=torch.ones(2))
        self.assertEqual(accepted["accept_proto"].dtype, torch.bool)

    def test_delta_regularization_and_ece_are_finite(self):
        proj_aux = {
            "delta": torch.ones(3, 4),
            "z_base": torch.ones(3, 4) * 2.0,
            "gate": torch.full((3, 4), 0.5),
            "residual_scale": torch.tensor(0.2),
        }
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]])
        labels = torch.tensor([0, 1, 1])

        self.assertTrue(torch.isfinite(delta_regularization_loss(proj_aux)))
        self.assertGreaterEqual(float(compute_ece(logits, labels, n_bins=3)), 0.0)

    def test_extract_frozen_features_is_strict_about_raw_identity_feature(self):
        outputs = {
            "z_id": torch.randn(2, 5),
            "z_dom": torch.randn(2, 3),
            "tx_logits": torch.randn(2, 4),
        }
        with self.assertRaises(KeyError):
            extract_frozen_features(outputs)

        features = extract_frozen_features(outputs, strict_raw=False)
        self.assertEqual(features["z_id_raw"].shape, (2, 5))
        self.assertEqual(features["base_logits"].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
