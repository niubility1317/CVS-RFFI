import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _LogitDataset(Dataset):
    def __init__(self):
        self.rows = [
            (torch.tensor([0.4, 0.6]), 0, 0, 0),
            (torch.tensor([0.9, 0.1]), 0, 1, 0),
            (torch.tensor([0.1, 0.9]), 1, 0, 1),
            (torch.tensor([0.3, 0.7]), 1, 1, 1),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        logits, y, rx_i, sig_i = self.rows[idx]
        meta = {
            "tx_i": int(y),
            "rx_i": int(rx_i),
            "day_i": 3,
            "eq_i": 1,
            "sig_i": int(sig_i),
        }
        return logits, int(y), 0, meta


class _PartialReceiverDataset(Dataset):
    def __init__(self):
        self.rows = [
            (torch.tensor([0.4, 0.6]), 0, 0, 0),
            (torch.tensor([0.9, 0.1]), 0, 1, 0),
            (torch.tensor([0.1, 0.9]), 1, 0, 1),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        logits, y, rx_i, sig_i = self.rows[idx]
        meta = {
            "tx_i": int(y),
            "rx_i": int(rx_i),
            "day_i": 3,
            "eq_i": 1,
            "sig_i": int(sig_i),
        }
        return logits, int(y), 0, meta


class _TransformDataset(Dataset):
    def __init__(self):
        self.rows = [
            (torch.tensor([0.1, 0.9]), 0, 0),
            (torch.tensor([0.1, 0.9]), 0, 1),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        logits, y, rx_i = self.rows[idx]
        meta = {
            "tx_i": int(y),
            "rx_i": int(rx_i),
            "day_i": 3,
            "eq_i": 1,
            "sig_i": 4,
        }
        return logits, int(y), 0, meta


class _LogitModel(torch.nn.Module):
    def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True):
        del y_tx, grl_lambda, return_aux
        return {"tx_logits": x.float()}


class CollaborativeInferenceEvalTest(unittest.TestCase):
    def test_parse_collab_counts_accepts_all_or_explicit_one_to_receiver_count(self):
        from evaluation.collaborative_inference_eval import parse_collab_counts

        self.assertEqual(parse_collab_counts("all", receiver_count=3), [1, 2, 3])
        self.assertEqual(parse_collab_counts("1,3", receiver_count=3), [1, 3])
        with self.assertRaises(ValueError):
            parse_collab_counts("0", receiver_count=3)
        with self.assertRaises(ValueError):
            parse_collab_counts("4", receiver_count=3)

    def test_collaborative_receiver_fusion_reports_k1_baseline_and_k2_rescue(self):
        from evaluation.collaborative_inference_eval import evaluate_collaborative_receiver_fusion

        loader = DataLoader(_LogitDataset(), batch_size=4, shuffle=False)
        result = evaluate_collaborative_receiver_fusion(
            _LogitModel(),
            loader,
            torch.device("cpu"),
            collab_counts=[1, 2],
            fusion="soft",
        )

        self.assertEqual(result["receiver_count"], 2)
        self.assertEqual(result["counts"]["1"]["total"], 2)
        self.assertEqual(result["counts"]["1"]["base_correct"], 1)
        self.assertEqual(result["counts"]["1"]["fused_correct"], 1)
        self.assertEqual(result["counts"]["1"]["rescue"], 0)
        self.assertEqual(result["counts"]["2"]["total"], 2)
        self.assertEqual(result["counts"]["2"]["base_correct"], 1)
        self.assertEqual(result["counts"]["2"]["fused_correct"], 2)
        self.assertEqual(result["counts"]["2"]["rescue"], 1)
        self.assertEqual(result["counts"]["2"]["harm"], 0)
        self.assertEqual(result["counts"]["2"]["net_gain"], 1)

    def test_collaborative_receiver_fusion_uses_same_eligible_groups_for_all_k(self):
        from evaluation.collaborative_inference_eval import evaluate_collaborative_receiver_fusion

        loader = DataLoader(_PartialReceiverDataset(), batch_size=3, shuffle=False)
        result = evaluate_collaborative_receiver_fusion(
            _LogitModel(),
            loader,
            torch.device("cpu"),
            collab_counts=[1, 2],
            fusion="soft",
        )

        self.assertEqual(result["receiver_count"], 2)
        self.assertEqual(result["counts"]["1"]["total"], 1)
        self.assertEqual(result["counts"]["2"]["total"], 1)
        self.assertEqual(result["counts"]["1"]["excluded_incomplete_groups"], 1)
        self.assertEqual(result["counts"]["2"]["excluded_incomplete_groups"], 1)

    def test_collaborative_receiver_fusion_can_apply_input_transform(self):
        from evaluation.collaborative_inference_eval import evaluate_collaborative_receiver_fusion

        loader = DataLoader(_TransformDataset(), batch_size=2, shuffle=False)
        result = evaluate_collaborative_receiver_fusion(
            _LogitModel(),
            loader,
            torch.device("cpu"),
            collab_counts=[1, 2],
            fusion="soft",
            input_transform=lambda x, _batch_idx: torch.stack(
                [torch.tensor([2.0, 0.0], dtype=x.dtype) for _ in range(x.size(0))], dim=0
            ),
        )

        self.assertEqual(result["counts"]["1"]["base_correct"], 1)
        self.assertEqual(result["counts"]["2"]["fused_correct"], 1)

    def test_checkpoint_identity_and_load_validation_fail_loudly(self):
        from evaluation.collaborative_inference_eval import load_model_state, validate_checkpoint_identity

        payload = {
            "args": {"run_name": "SA33_sa27_ch2_leo3_ce0p7_r010"},
            "model": {"unexpected.weight": torch.tensor([1.0])},
        }
        with self.assertRaises(ValueError):
            validate_checkpoint_identity(
                payload,
                checkpoint_path="best_primary_ood_model.pth",
                expected_run_name="SA34_wrong",
                checkpoint_sha256=None,
                expected_sha256=None,
            )
        with self.assertRaises(ValueError):
            load_model_state(_LogitModel(), payload, max_missing=0, max_unexpected=0)


if __name__ == "__main__":
    unittest.main()
