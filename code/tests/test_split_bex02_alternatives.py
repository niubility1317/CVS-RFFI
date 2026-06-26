import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.activation_tokens import ActivationTokenCodec
from federated.distill_anchors import (
    LogitAnchorBank,
    build_logit_anchor_stats,
    logit_anchor_kd_loss,
    logit_anchor_stats_payload_size_bytes,
)
from federated.gradient_stats import conflict_aware_aggregate_gradients
from federated.style_packet import StylePacket, style_code_from_stats


class SplitBEX02AlternativeUtilityTest(unittest.TestCase):
    def test_conflict_aware_gradient_aggregation_reports_and_resolves_opposing_clients(self):
        client_gradients = {
            "rx0": {"w": torch.tensor([1.0, 0.0])},
            "rx1": {"w": torch.tensor([-0.5, 0.0])},
            "rx2": {"w": torch.tensor([0.5, 0.0])},
        }
        weights = {"rx0": 1.0 / 3.0, "rx1": 1.0 / 3.0, "rx2": 1.0 / 3.0}

        plain, plain_metrics = conflict_aware_aggregate_gradients(client_gradients, weights, mode="none")
        clipped, clipped_metrics = conflict_aware_aggregate_gradients(client_gradients, weights, mode="cosine_clip")

        self.assertTrue(torch.allclose(plain["w"], torch.tensor([1.0 / 3.0, 0.0]), atol=1e-6))
        self.assertEqual(plain_metrics["conflict_mode"], "none")
        self.assertGreaterEqual(clipped_metrics["conflicts_detected"], 1)
        self.assertGreaterEqual(clipped_metrics["conflicts_resolved"], 1)
        self.assertGreaterEqual(float(clipped["w"][0].item()), float(plain["w"][0].item()))
        self.assertIn("grad_cos_mean_before", clipped_metrics)
        self.assertIn("grad_cos_mean_after", clipped_metrics)

    def test_conflict_aware_gradient_aggregation_preserves_non_common_gradient_keys(self):
        client_gradients = {
            "rx0": {"shared": torch.tensor([1.0]), "head_a": torch.tensor([2.0])},
            "rx1": {"shared": torch.tensor([3.0]), "head_b": torch.tensor([4.0])},
        }
        weights = {"rx0": 0.5, "rx1": 0.5}

        aggregated, metrics = conflict_aware_aggregate_gradients(client_gradients, weights, mode="none")

        self.assertTrue(torch.allclose(aggregated["shared"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(aggregated["head_a"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(aggregated["head_b"], torch.tensor([2.0])))
        self.assertEqual(metrics["missing_gradient_entries"], 2)

    def test_logit_anchor_bank_gates_unreliable_teacher_logits_before_distillation(self):
        logits = torch.tensor(
            [
                [5.0, 0.1, -1.0],
                [0.2, 3.0, -0.5],
                [0.2, 2.0, 2.1],
                [0.1, 0.2, 4.0],
            ]
        )
        labels = torch.tensor([0, 1, 1, 2])
        stats = build_logit_anchor_stats(
            logits,
            labels,
            confidence_min=0.70,
            margin_min=0.50,
            require_correct=True,
        )
        bank = LogitAnchorBank(num_classes=3, ema_alpha=0.0)
        summary = bank.update(stats)
        anchor_logits, counts = bank.tensors()
        student_logits = logits + torch.tensor([[0.0, 0.4, 0.0]])
        loss, metrics = logit_anchor_kd_loss(
            student_logits,
            labels,
            anchor_logits,
            counts,
            temperature=2.0,
            min_count=1,
        )

        self.assertEqual(summary["anchor_count_nonzero"], 3)
        self.assertEqual(int(counts.sum().item()), 3)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(metrics["kd_active"], 0.0)
        self.assertGreater(logit_anchor_stats_payload_size_bytes(stats), 0)

    def test_activation_token_codec_quantizes_features_and_accounts_for_payload(self):
        features = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            dtype=torch.float32,
        )
        codec = ActivationTokenCodec(route="quantized", quant_bits=4)
        packet = codec.encode(features)
        decoded = codec.decode(packet)

        self.assertEqual(packet.route, "quantized")
        self.assertEqual(tuple(packet.original_shape), tuple(features.shape))
        self.assertEqual(tuple(decoded.shape), tuple(features.shape))
        self.assertLess(packet.payload_bytes, features.numel() * features.element_size())
        self.assertLess(packet.compression_ratio, 1.0)
        self.assertGreaterEqual(packet.quantization_error, 0.0)

    def test_style_packet_carries_bounded_fixed_dim_style_code(self):
        stats = {"phys_cfo_hz": 1200.0, "phys_sro_ppm": -8.0, "phys_awgn_snr_db": 32.0}
        code = style_code_from_stats(stats, dim=6)
        packet = StylePacket(
            client_id="rx0",
            round_idx=1,
            count=8,
            stats=stats,
            style_code=code,
        )
        restored = StylePacket.from_dict(packet.to_dict())

        self.assertEqual(len(code), 6)
        self.assertTrue(all(-1.0 <= float(v) <= 1.0 for v in code))
        self.assertEqual(len(restored.style_code or []), 6)
        self.assertEqual(packet.vector(keys=("__style_code__",)).numel(), 6)


if __name__ == "__main__":
    unittest.main()
