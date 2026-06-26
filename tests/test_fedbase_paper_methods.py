import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class FedRIEIMethodTest(unittest.TestCase):
    def test_fedriei_loss_terms_match_paper_signs(self):
        from federated.fedriei import fedriei_loss_terms

        outputs = {
            "z_e": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "z_r": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "emitter_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            "receiver_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "cross_emitter_logits": torch.zeros(2, 2),
            "cross_receiver_logits": torch.zeros(2, 2),
        }
        terms = fedriei_loss_terms(
            outputs,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            lambda_mi=1.2,
            lambda_ie=1.2,
        )

        expected = terms["loss_ce"] + 1.2 * terms["loss_mi"] - 1.2 * terms["loss_ie"]
        self.assertTrue(torch.allclose(terms["loss"], expected))
        self.assertGreater(float(terms["loss_ie"]), 0.0)
        self.assertIn("loss_ce_e", terms)
        self.assertIn("loss_ce_r", terms)

    def test_fedriei_alternating_step_reports_two_paper_phases(self):
        from baselines.riei_fd.model import RIEIModel
        from federated.fedriei import fedriei_alternating_step

        torch.manual_seed(7)
        model = RIEIModel(
            num_emitters=3,
            num_receivers=2,
            feature_dim=32,
            classifier_hidden_dim=16,
            dropout=0.0,
            encoder_use_projection=True,
        )
        optimizers = {
            "ce": torch.optim.SGD(model.parameters(), lr=1e-4),
            "disentangle": torch.optim.SGD(model.fed.parameters(), lr=1e-4),
        }
        batch = {
            "iq": torch.randn(4, 2, 128),
            "label": torch.tensor([0, 1, 2, 1]),
            "receiver": torch.tensor([0, 1, 0, 1]),
        }

        metrics = fedriei_alternating_step(model, batch, optimizers, torch.device("cpu"))

        self.assertEqual(metrics["phase_order"], ["ce_phase", "disentangle_phase"])
        self.assertIn("ce_phase_loss_ce", metrics)
        self.assertIn("disentangle_phase_loss_mi", metrics)
        self.assertIn("disentangle_phase_loss_ie", metrics)

    def test_fedriei_compression_variants_follow_paper_sign_semantics(self):
        from federated.fedriei import compress_gradient_tensor

        grad = torch.tensor([-2.0, 0.0, 3.0])

        self.assertTrue(torch.equal(compress_gradient_tensor(grad, method="signsgd"), torch.tensor([-1.0, 1.0, 1.0])))
        self.assertTrue(
            torch.equal(
                compress_gradient_tensor(grad, method="1-signsgd", noise_std=0.0),
                torch.tensor([-1.0, 1.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                compress_gradient_tensor(grad, method="infinity-signsgd", noise_std=0.0),
                torch.tensor([-1.0, 1.0, 1.0]),
            )
        )

    def test_fedriei_server_update_applies_weighted_compressed_gradient_step(self):
        from federated.fedriei import apply_fedriei_server_gradient_step

        global_state = {"w": torch.tensor([1.0, -1.0])}
        client_gradients = {
            "rx0": {"w": torch.tensor([1.0, -1.0])},
            "rx1": {"w": torch.tensor([-1.0, -1.0])},
        }

        new_state = apply_fedriei_server_gradient_step(
            global_state,
            client_gradients,
            client_num_samples={"rx0": 3, "rx1": 1},
            selected=["rx0", "rx1"],
            server_lr=0.1,
        )

        expected_grad = 0.75 * client_gradients["rx0"]["w"] + 0.25 * client_gradients["rx1"]["w"]
        self.assertTrue(torch.allclose(new_state["w"], global_state["w"] - 0.1 * expected_grad))


class FedFAMethodTest(unittest.TestCase):
    def test_pairwise_coral_alignment_uses_full_covariance_and_paper_scale(self):
        from federated.feature_alignment import pairwise_coral_alignment_loss

        a = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        b = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

        loss = pairwise_coral_alignment_loss([a, b])

        ca = torch.cov(a.T)
        cb = torch.cov(b.T)
        expected = torch.sum((ca - cb) ** 2) / (4.0 * (2 ** 2))
        self.assertTrue(torch.allclose(loss, expected))

    def test_fedfa_complex_cnn_returns_log_probs_and_512_embedding_by_default(self):
        from federated.feature_alignment import FedFAComplexCNN

        model = FedFAComplexCNN(num_classes=4)
        out = model(torch.randn(3, 2, 256))

        self.assertEqual(tuple(out["embedding"].shape), (3, 512))
        self.assertEqual(tuple(out["log_probs"].shape), (3, 4))
        self.assertTrue(torch.allclose(out["log_probs"].exp().sum(dim=1), torch.ones(3), atol=1e-5))

    def test_fedfa_complex_blocks_use_explicit_real_imag_bn_relu(self):
        from federated.feature_alignment import FedFAComplexCNN

        model = FedFAComplexCNN(num_classes=4)
        block = next(module for module in model.blocks.modules() if module.__class__.__name__ == "ComplexConvBlock1d")

        self.assertIsInstance(block.bn_real, torch.nn.BatchNorm1d)
        self.assertIsInstance(block.bn_imag, torch.nn.BatchNorm1d)
        self.assertIsInstance(block.act_real, torch.nn.ReLU)
        self.assertIsInstance(block.act_imag, torch.nn.ReLU)
        self.assertNotEqual(block.bn_real, block.bn_imag)

    def test_fedfa_fc_stage_has_three_linear_layers_and_dropout(self):
        from federated.feature_alignment import FedFAComplexCNN

        model = FedFAComplexCNN(num_classes=4)
        linear_layers = [module for module in model.fc_stage.modules() if isinstance(module, torch.nn.Linear)]
        dropout_layers = [module for module in model.fc_stage.modules() if isinstance(module, torch.nn.Dropout)]
        out = model(torch.randn(2, 2, 256))

        self.assertEqual([layer.out_features for layer in linear_layers], [512, 512, 512])
        self.assertGreaterEqual(len(dropout_layers), 2)
        self.assertEqual(tuple(out["embedding"].shape), (2, 512))

    def test_fedfa_peer_coral_alignment_detaches_other_client_statistics(self):
        from federated.feature_alignment import peer_coral_alignment_losses

        a = torch.randn(4, 3, requires_grad=True)
        b = torch.randn(4, 3, requires_grad=True)
        losses = peer_coral_alignment_losses([a, b])

        losses[0].backward(retain_graph=True)

        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)
        self.assertIsNone(b.grad)


class FUCLMethodTest(unittest.TestCase):
    def test_nt_xent_uses_same_sample_two_view_positives(self):
        from federated.contrastive_fl import nt_xent_loss

        view_a = F.normalize(torch.eye(4), dim=1)
        view_b = view_a.clone()
        loss_same = nt_xent_loss(view_a, view_b, temperature=0.05)
        loss_shuffled = nt_xent_loss(view_a, view_b.flip(0), temperature=0.05)

        self.assertLess(float(loss_same), float(loss_shuffled))

    def test_encoder_only_state_dict_excludes_classifier_and_projection_head(self):
        from federated.contrastive_fl import encoder_only_state_dict

        state = {
            "encoder.conv.weight": torch.ones(1),
            "feature_extractor.fc.weight": torch.ones(1) * 2,
            "classifier.weight": torch.ones(1) * 3,
            "projection_head.weight": torch.ones(1) * 4,
            "rf_fingerprint_head.weight": torch.ones(1) * 5,
        }
        filtered = encoder_only_state_dict(state)

        self.assertIn("encoder.conv.weight", filtered)
        self.assertIn("feature_extractor.fc.weight", filtered)
        self.assertIn("rf_fingerprint_head.weight", filtered)
        self.assertNotIn("classifier.weight", filtered)
        self.assertNotIn("projection_head.weight", filtered)

    def test_fucl_model_matches_paper_spectrogram_cnn_and_classifier_head(self):
        from federated.fedbase_paper_trainer import FUCL1DModel

        model = FUCL1DModel(num_classes=6)
        conv_layers = [module for module in model.conv_layers.modules() if isinstance(module, torch.nn.Conv2d)]
        pools = [module for module in model.conv_layers.modules() if isinstance(module, torch.nn.MaxPool2d)]
        rf_layers = [module for module in model.rf_fingerprint_head.modules() if isinstance(module, torch.nn.Linear)]
        linear_layers = [module for module in model.classifier.modules() if isinstance(module, torch.nn.Linear)]
        out = model(torch.randn(3, 1, 26, 126))

        self.assertEqual([layer.out_channels for layer in conv_layers], [8, 16, 32])
        self.assertEqual(len(pools), 2)
        self.assertEqual([layer.out_features for layer in rf_layers], [128, 64, 128])
        self.assertEqual(tuple(out["feature"].shape), (3, 128))
        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(linear_layers[0].out_features, 64)
        self.assertEqual(linear_layers[1].out_features, 6)

    def test_fucl_model_rejects_raw_iq_input(self):
        from federated.fedbase_paper_trainer import FUCL1DModel

        model = FUCL1DModel(num_classes=6)

        with self.assertRaises(ValueError):
            model(torch.randn(3, 2, 256))

    def test_fucl_tdl_views_return_channel_independent_spectrograms(self):
        from federated.contrastive_fl import TDLChannelConfig, make_two_channel_views

        gen_a = torch.Generator().manual_seed(101)
        gen_b = torch.Generator().manual_seed(202)
        config = TDLChannelConfig(num_taps=3)
        view_a, view_b = make_two_channel_views(
            torch.randn(4, 2, 256),
            generator_a=gen_a,
            generator_b=gen_b,
            tdl_config=config,
        )

        self.assertEqual(tuple(view_a.shape), (4, 1, 26, 126))
        self.assertEqual(tuple(view_b.shape), (4, 1, 26, 126))
        self.assertFalse(torch.allclose(view_a, view_b))

    def test_fucl_cis_has_paper_shape_for_lora_length(self):
        from federated.contrastive_fl import channel_independent_spectrogram

        spec = channel_independent_spectrogram(torch.randn(2, 2, 4096))

        self.assertEqual(tuple(spec.shape), (2, 1, 26, 126))


class RAFLMethodTest(unittest.TestCase):
    def test_receiver_agnostic_loss_logs_tx_and_receiver_terms(self):
        from federated.receiver_agnostic_fl import receiver_agnostic_loss

        outputs = {
            "tx_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            "rx_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        }
        terms = receiver_agnostic_loss(outputs, torch.tensor([0, 1]), torch.tensor([0, 1]), lambda_rx=0.1)

        expected = terms["loss_tx"] + 0.1 * terms["loss_rx"]
        self.assertTrue(torch.allclose(terms["loss"], expected))
        self.assertEqual(terms["receiver_gradient_semantics"], "grl_reversed_feature_path")

    def test_receiver_agnostic_loss_can_leave_rx_head_loss_unscaled_for_paper_eq_11_12(self):
        from federated.receiver_agnostic_fl import receiver_agnostic_loss

        outputs = {
            "tx_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            "rx_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        }
        terms = receiver_agnostic_loss(
            outputs,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            lambda_rx=0.1,
            scale_receiver_loss=False,
        )

        expected = terms["loss_tx"] + terms["loss_rx"]
        self.assertTrue(torch.allclose(terms["loss"], expected))
        self.assertEqual(float(terms["receiver_loss_weight"]), 1.0)

    def test_grl_lambda_scales_feature_gradient_only(self):
        from federated.fedbase_paper_trainer import _grad_reverse

        torch.manual_seed(23)
        labels = torch.tensor([0, 1, 0])
        head_grl = torch.nn.Linear(2, 2, bias=False)
        head_plain = torch.nn.Linear(2, 2, bias=False)
        head_plain.load_state_dict(head_grl.state_dict())
        feature_grl = torch.randn(3, 2, requires_grad=True)
        feature_plain = feature_grl.detach().clone().requires_grad_(True)

        F.cross_entropy(head_grl(_grad_reverse(feature_grl, 0.1)), labels).backward()
        F.cross_entropy(head_plain(feature_plain), labels).backward()

        self.assertTrue(torch.allclose(feature_grl.grad, -0.1 * feature_plain.grad, atol=1e-6))
        self.assertTrue(torch.allclose(head_grl.weight.grad, head_plain.weight.grad, atol=1e-6))

    def test_rafl_model_accepts_paper_spectrogram_and_l2_normalizes_feature(self):
        from federated.fedbase_paper_trainer import RAFLPaperResNet2D

        model = RAFLPaperResNet2D(num_classes=4, num_receivers=3, feature_dim=32)
        out = model(torch.randn(5, 1, 52, 126))

        self.assertEqual(len(model.residual_blocks), 4)
        self.assertTrue(torch.allclose(out["feature"].norm(dim=1), torch.ones(5), atol=1e-5))
        self.assertTrue(torch.allclose(out["tx_probs"].sum(dim=1), torch.ones(5), atol=1e-5))
        self.assertTrue(torch.allclose(out["rx_probs"].sum(dim=1), torch.ones(5), atol=1e-5))

    def test_rafl_model_rejects_raw_iq_input(self):
        from federated.fedbase_paper_trainer import RAFLPaperResNet2D

        model = RAFLPaperResNet2D(num_classes=4, num_receivers=3, feature_dim=32)

        with self.assertRaises(ValueError):
            model(torch.randn(5, 2, 128))

    def test_rafl_model_accepts_wisig_native_spectrogram_shape(self):
        from federated.fedbase_paper_trainer import RAFLPaperResNet2D

        model = RAFLPaperResNet2D(num_classes=4, num_receivers=3, feature_dim=32)
        out = model(torch.randn(5, 1, 64, 9))

        self.assertEqual(tuple(out["tx_logits"].shape), (5, 4))
        self.assertTrue(torch.allclose(out["feature"].norm(dim=1), torch.ones(5), atol=1e-5))

    def test_rafl_model_accepts_wisig_complex_two_channel_spectrogram(self):
        from federated.fedbase_paper_trainer import RAFLPaperResNet2D

        model = RAFLPaperResNet2D(num_classes=4, num_receivers=3, feature_dim=32, input_channels=2)
        out = model(torch.randn(5, 2, 52, 129))

        self.assertEqual(tuple(out["tx_logits"].shape), (5, 4))
        self.assertTrue(torch.allclose(out["feature"].norm(dim=1), torch.ones(5), atol=1e-5))

    def test_label_loss_driven_selection_uses_label_wise_losses_not_overall_loss(self):
        from federated.receiver_agnostic_fl import label_loss_driven_client_selection

        label_losses = {
            "rx0": {0: 0.10, 1: 0.20, 2: 4.00},
            "rx1": {0: 3.00, 1: 0.10},
            "rx2": {1: 2.50, 2: 0.30},
        }

        result = label_loss_driven_client_selection(label_losses, clients_per_round=2)

        self.assertEqual(result["selected_labels"], [2, 0])
        self.assertEqual(result["selected_clients"], ["rx0", "rx1"])
        self.assertGreater(result["aggregated_label_losses"][2], result["aggregated_label_losses"][1])

    def test_label_loss_driven_selection_uses_candidate_set_and_random_fill(self):
        from federated.receiver_agnostic_fl import label_loss_driven_client_selection

        label_losses = {
            "rx0": {0: 5.0},
            "rx1": {1: 4.0},
            "rx2": {2: 99.0},
            "rx3": {3: 0.1},
        }

        result = label_loss_driven_client_selection(
            label_losses,
            clients_per_round=3,
            candidate_clients=["rx0", "rx1", "rx3"],
            random_seed=17,
        )

        self.assertEqual(set(result["candidate_clients"]), {"rx0", "rx1", "rx3"})
        self.assertNotIn("rx2", result["selected_clients"])
        self.assertEqual(len(result["selected_clients"]), 3)


if __name__ == "__main__":
    unittest.main()
