import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class TinyBaseTeacher(nn.Module):
    def __init__(self, num_classes=4, feat_dim=6):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(2, feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def extract_feature(self, x):
        return self.proj(self.pool(x).squeeze(-1))

    def classify(self, x, feat=None):
        if feat is None:
            feat = self.extract_feature(x)
        return self.head(feat)


class SGCv3Test(unittest.TestCase):
    def test_physical_canonicalizer_returns_safe_candidate_views_without_trainable_tcn(self):
        from SGC.v3.physical_canonicalizer import PhysicalSafeCanonicalizer

        x = torch.randn(3, 2, 32)
        psc = PhysicalSafeCanonicalizer(cfo_betas=(0.0, 0.5, -0.5), shifts=(-1, 0, 1), envelope_gammas=(0.0, 0.2))

        out = psc(x)

        self.assertEqual(out["views"].shape[:3], (3, 8, 2))
        self.assertEqual(out["views"].shape[-1], 32)
        self.assertIn("identity", out["view_names"])
        self.assertIn("cfo_beta_0.5", out["view_names"])
        self.assertIn("shift_1", out["view_names"])
        self.assertEqual(sum(p.numel() for p in psc.parameters()), 0)
        self.assertTrue(torch.isfinite(out["stats"]["cfo_hat"]).all())

    def test_satellite_evidence_encoder_uses_statistics_to_gate_views(self):
        from SGC.v3.satellite_evidence_encoder import SatelliteEvidenceEncoder

        x = torch.randn(5, 2, 40)
        encoder = SatelliteEvidenceEncoder(num_views=6, scenario_dim=8, num_experts=3)

        out = encoder(x)

        self.assertEqual(out["scenario_code"].shape, (5, 8))
        self.assertEqual(out["view_weights"].shape, (5, 6))
        self.assertTrue(torch.allclose(out["view_weights"].sum(dim=-1), torch.ones(5), atol=1e-5))
        self.assertEqual(out["expert_weights"].shape, (5, 3))
        self.assertEqual(out["sat_logit"].shape, (5, 1))
        self.assertTrue(torch.logical_and(out["sat_score"] >= 0, out["sat_score"] <= 1).all())
        self.assertTrue(torch.allclose(out["sat_score"], torch.sigmoid(out["sat_logit"]), atol=1e-6))
        self.assertIn("spectral_flatness", out["channel_stats"])

    def test_prototype_bank_scores_and_pull_push_loss(self):
        from SGC.v3.prototype_bank import PrototypeBank

        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])

        bank = PrototypeBank.from_features(features, labels, num_classes=2, prototypes_per_class=2)
        scores = bank.class_scores(features)
        loss = bank.pull_push_loss(features, labels)

        self.assertEqual(bank.prototypes.shape, (2, 2, 3))
        self.assertEqual(scores.shape, (4, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))

    def test_feature_adapter_and_logit_calibrator_are_norm_clamped_and_topk_limited(self):
        from SGC.v3.feature_adapter import IdentityPreservingFeatureAdapter
        from SGC.v3.logit_calibrator import BaseAnchoredLogitCalibrator

        torch.manual_seed(7)
        z = torch.randn(4, 6)
        code = torch.randn(4, 8)
        p_stats = torch.randn(4, 4)
        adapter = IdentityPreservingFeatureAdapter(feature_dim=6, scenario_dim=8, rank=3, hidden_dim=12, epsilon_z=0.02)
        z_sgc, aux_z = adapter(z, code, p_stats, gate=torch.ones(4, 1))

        self.assertEqual(z_sgc.shape, z.shape)
        self.assertLessEqual(float(aux_z["delta_z_ratio"].detach().max()), 0.0201)

        logits = torch.randn(4, 5)
        calibrator = BaseAnchoredLogitCalibrator(
            feature_dim=6,
            scenario_dim=8,
            num_classes=5,
            topk_only=3,
            epsilon_logit=0.5,
            hidden_dim=16,
        )
        logits_sgc, aux_l = calibrator(z_sgc, code, logits, gate=torch.ones(4, 1))
        changed = aux_l["delta_logits"].abs() > 1e-8

        self.assertEqual(logits_sgc.shape, logits.shape)
        self.assertLessEqual(float(aux_l["delta_logit_norm"].detach().max()), 0.5001)
        self.assertTrue((changed.sum(dim=-1) <= 3).all())

    def test_sgc_v3_model_returns_documented_outputs_and_keeps_teacher_frozen(self):
        from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model

        torch.manual_seed(11)
        teacher = TinyBaseTeacher(num_classes=4, feat_dim=6)
        model = SGCv3Model(
            teacher,
            SGCv3Config(num_classes=4, feature_dim=6, scenario_dim=8, psc_cfo_betas=(0.0, 0.5), psc_shifts=(0,)),
        )
        x = torch.randn(3, 2, 48)

        out = model(x)

        for key in (
            "logits_base",
            "logits_sgc",
            "logits_final",
            "prob_final",
            "z_base",
            "z_sgc",
            "scenario_code",
            "gate",
            "view_weights",
            "sat_logit",
            "sat_score",
            "pseudo_y",
            "pseudo_weight",
            "x_phys_best",
            "metrics",
        ):
            self.assertIn(key, out)
        self.assertEqual(out["logits_final"].shape, (3, 4))
        self.assertEqual(out["z_sgc"].shape, (3, 6))
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        self.assertTrue(torch.logical_and(out["gate"] >= 0, out["gate"] <= 1).all())

    def test_losses_metrics_and_checkpoint_constraints_cover_v3_training_contract(self):
        from SGC.v3.losses_v3 import compute_sgc_v3_losses
        from SGC.v3.metrics_v3 import check_constrained_improvement, compute_flip_metrics
        from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model

        torch.manual_seed(13)
        teacher = TinyBaseTeacher(num_classes=4, feat_dim=6)
        model = SGCv3Model(teacher, SGCv3Config(num_classes=4, feature_dim=6, scenario_dim=8))
        batch = {
            "x_clean": torch.randn(5, 2, 32),
            "x_sat": torch.randn(5, 2, 32),
            "y": torch.tensor([0, 1, 2, 3, 1]),
            "scenario": torch.tensor([1, 1, 2, 2, 0]),
            "x_target": torch.randn(5, 2, 32),
        }

        loss, logs = compute_sgc_v3_losses(model, batch)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for key in (
            "train/loss_clean_kl",
            "train/loss_pair_logit",
            "train/loss_sat_ce",
            "train/loss_gate_safety",
            "target/pseudo_coverage",
            "sgc/delta_z_ratio_mean",
            "sgc/delta_logit_norm_mean",
        ):
            self.assertIn(key, logs)
        self.assertTrue(any(p.grad is not None for p in model.feature_adapter.parameters()))
        self.assertFalse(any(p.grad is not None for p in teacher.parameters()))

        base_logits = torch.tensor([[2.0, 1.0], [0.1, 2.0], [2.0, 0.1]])
        final_logits = torch.tensor([[1.0, 2.0], [0.1, 2.0], [3.0, 0.1]])
        labels = torch.tensor([1, 1, 0])
        flip = compute_flip_metrics(base_logits, final_logits, labels)
        self.assertEqual(float(flip["wrong_to_right_count"]), 1.0)
        self.assertEqual(float(flip["right_to_wrong_count"]), 0.0)
        self.assertTrue(
            check_constrained_improvement(
                {"clear_leo_tx": 50.4},
                {"clear_leo_tx": 50.0},
                {
                    "clean_drop": 0.2,
                    "normal_drop": 0.3,
                    "gate_clean_mean": 0.03,
                    "net_gain_sat": 0.01,
                },
            )
        )

    def test_training_log_helpers_match_backbone_test_and_satellite_style(self):
        from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model
        from SGC.v3.train_sgc_v3 import SGCv3EvalAdapter, format_sgc_v3_epoch_report

        teacher = TinyBaseTeacher(num_classes=4, feat_dim=6)
        model = SGCv3Model(teacher, SGCv3Config(num_classes=4, feature_dim=6, scenario_dim=8))
        wrapped = SGCv3EvalAdapter(model)
        out = wrapped(torch.randn(2, 2, 32), y_tx=None, grl_lambda=1.0, return_aux=True)

        self.assertIn("tx_logits", out)
        self.assertIn("dom_logits", out)
        self.assertEqual(out["tx_logits"].shape, (2, 4))
        self.assertEqual(out["dom_logits"].shape[0], 2)

        report = format_sgc_v3_epoch_report(
            epoch=7,
            epochs=40,
            lr=1e-4,
            epoch_time_s=3.25,
            logs={
                "train/loss_total": 2.1486,
                "train/loss_clean_kl": 0.0126,
                "train/loss_clean_feat": 0.6579,
                "train/loss_pair_logit": 0.9084,
                "train/loss_pair_feat": 0.019,
                "train/loss_proto": 0.0,
                "train/loss_sat_ce": 1.7616,
                "train/loss_gate_safety": 0.033,
                "target/pseudo_coverage": 0.125,
                "sgc/gate_clean_mean": 0.012,
                "sgc/gate_sat_mean": 0.231,
                "sgc/delta_z_ratio_mean": 0.0178,
                "sgc/delta_z_ratio_p95": 0.021,
                "sgc/delta_logit_norm_mean": 0.1096,
                "sgc/net_gain": 0.0019,
                "sgc/wrong_to_right": 0.004,
                "sgc/right_to_wrong": 0.002,
                "sgc/top1_flip_rate": 0.006,
            },
            val_stats={"tx_acc": 98.38, "dom_acc": float("nan"), "tx_correct": 100, "tx_total": 102},
            test_stats={"tx_acc": 88.22, "tx_correct": 179962, "tx_total": 204000},
            named_test_stats={
                "test_unseen_day_seen_rx": {"tx_acc": 92.35, "tx_correct": 77575, "tx_total": 84000},
                "test_seen_day_unseen_rx": {"tx_acc": 86.28, "tx_correct": 51770, "tx_total": 60000},
                "test_unseen_day_unseen_rx": {"tx_acc": 84.36, "tx_correct": 50617, "tx_total": 60000},
            },
            named_test_meta={
                "test_unseen_day_seen_rx": {"days_label": ["2021_03_15"], "rxs_idx": [0, 1]},
                "test_seen_day_unseen_rx": {"days_label": ["2021_03_01"], "rxs_idx": [7]},
                "test_unseen_day_unseen_rx": {"days_label": ["2021_03_15"], "rxs_idx": [7]},
            },
            sat_test_stats={
                "mixed_orbit": {
                    "aggregate": {"tx_acc": 42.06, "tx_correct": 25234, "tx_total": 60000},
                    "strict_udu": 42.06,
                    "selected_names": ["test_unseen_day_unseen_rx"],
                }
            },
            best_score=0.0019,
            best_epoch=7,
            latest_path="latest_sgc_v3.pth",
            best_path="best_sgc_v3.pth",
            best_updated=True,
        )

        self.assertIn("[SGC-LOSS]", report)
        self.assertIn("[SGC-METRIC]", report)
        self.assertIn("[TEST]  overall_tx=88.22%", report)
        self.assertIn("[TEST-SPLIT]", report)
        self.assertIn("[SAT-TEST] scenario=mixed_orbit", report)


if __name__ == "__main__":
    unittest.main()
