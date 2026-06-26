import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class PaperOriginalParityTest(unittest.TestCase):
    def test_wisig_episode_requires_receiver_metadata_and_disjoint_source_target(self):
        from paper_reproduction.common.episodes import EpisodeBatch, validate_closed_set_episode

        episode = EpisodeBatch(
            support_x=torch.randn(4, 3, 256),
            support_y=torch.tensor([0, 0, 1, 1]),
            query_x=torch.randn(2, 3, 256),
            query_y=torch.tensor([0, 1]),
            support_ids=("s0", "s1", "s2", "s3"),
            query_ids=("q0", "q1"),
            support_receivers=("rx7", "rx7", "rx7", "rx7"),
            query_receivers=("rx7", "rx7"),
            source_receivers=("rx0", "rx1"),
            target_receiver="rx7",
        )
        validate_closed_set_episode(episode, k_shot=2, n_way=2, require_receiver_metadata=True)

        leaked_receiver = EpisodeBatch(
            support_x=episode.support_x,
            support_y=episode.support_y,
            query_x=episode.query_x,
            query_y=episode.query_y,
            support_ids=episode.support_ids,
            query_ids=episode.query_ids,
            support_receivers=episode.support_receivers,
            query_receivers=("rx8", "rx7"),
            source_receivers=("rx0", "rx1"),
            target_receiver="rx7",
        )
        with self.assertRaisesRegex(ValueError, "target receiver"):
            validate_closed_set_episode(leaked_receiver, k_shot=2, n_way=2, require_receiver_metadata=True)

        overlapping_source = EpisodeBatch(
            support_x=episode.support_x,
            support_y=episode.support_y,
            query_x=episode.query_x,
            query_y=episode.query_y,
            support_ids=episode.support_ids,
            query_ids=episode.query_ids,
            support_receivers=episode.support_receivers,
            query_receivers=episode.query_receivers,
            source_receivers=("rx0", "rx7"),
            target_receiver="rx7",
        )
        with self.assertRaisesRegex(ValueError, "source/target receiver"):
            validate_closed_set_episode(overlapping_source, k_shot=2, n_way=2, require_receiver_metadata=True)

    def test_feature_separation_uses_three_channel_wisig_fusion_and_paper_loss_terms(self):
        from paper_reproduction.feature_separation_crossrx.losses import feature_separation_loss
        from paper_reproduction.feature_separation_crossrx.model import (
            ChannelAttention,
            FeatureSeparationNet,
            build_wisig_fusion_representation,
        )

        torch.manual_seed(11)
        iq = torch.randn(4, 2, 256)
        fused = build_wisig_fusion_representation(iq)
        self.assertEqual(tuple(fused.shape), (4, 3, 256))

        model = FeatureSeparationNet(input_channels=3, input_length=256, num_tx=6, num_rx=3)
        self.assertTrue(any(isinstance(module, ChannelAttention) for module in model.modules()))
        conv2d_count = sum(1 for module in model.modules() if isinstance(module, nn.Conv2d))
        self.assertGreaterEqual(conv2d_count, 17)

        outputs = model(fused)
        for key in ("tx_logits", "rx_logits", "tx_from_rx_logits", "rx_from_tx_logits"):
            self.assertIn(key, outputs)

        labels_tx = torch.tensor([0, 1, 2, 3])
        labels_rx = torch.tensor([0, 1, 2, 0])
        loss, terms = feature_separation_loss(
            outputs,
            labels_tx,
            labels_rx,
            lambda_similarity=0.1,
            lambda_tx_entropy=0.2,
            lambda_rx_entropy=0.3,
        )
        self.assertEqual(
            set(terms),
            {"tx_ce", "rx_ce", "similarity", "tx_entropy", "rx_entropy", "total"},
        )
        from paper_reproduction.feature_separation_crossrx.losses import entropy_loss

        self.assertTrue(torch.allclose(terms["tx_entropy"], entropy_loss(outputs["tx_logits"]).detach()))
        self.assertTrue(torch.allclose(terms["rx_entropy"], entropy_loss(outputs["rx_logits"]).detach()))
        expected = (
            terms["tx_ce"]
            + terms["rx_ce"]
            + 0.1 * terms["similarity"]
            + 0.2 * terms["tx_entropy"]
            + 0.3 * terms["rx_entropy"]
        )
        self.assertTrue(torch.allclose(loss.detach(), expected, atol=1e-6))
        loss.backward()
        self.assertIsNotNone(model.tx_classifier.weight.grad)
        self.assertIsNotNone(model.rx_classifier.weight.grad)

    def test_snapshot_is_immutable_and_rejects_unspecified_formal_configs(self):
        config = {
            "baseline": "protonet_cda",
            "dataset": "WiSig",
            "source_domain": "rx0-rx6",
            "target_domain": "rx7-rx11",
            "commands": ["python -m paper_reproduction.protonet_cda.train --config config.json"],
        }
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            config_path = tmp / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            out_dir = tmp / "snapshot"
            cmd = [
                sys.executable,
                "-m",
                "paper_reproduction.scripts.make_repro_snapshot",
                "--config",
                str(config_path),
                "--command",
                config["commands"][0],
                "--out-dir",
                str(out_dir),
                "--formal",
            ]
            first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            manifest = json.loads((out_dir / "snapshot_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("code_sha256", manifest)
            self.assertEqual(manifest["commands"], config["commands"])

            second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

            bad_config = tmp / "bad.json"
            bad_config.write_text(json.dumps({"dataset": "paper-unspecified"}), encoding="utf-8")
            bad = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "paper_reproduction.scripts.make_repro_snapshot",
                    "--config",
                    str(bad_config),
                    "--command",
                    "python train.py",
                    "--out-dir",
                    str(tmp / "bad_snapshot"),
                    "--formal",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("paper-unspecified", bad.stderr)
            self.assertFalse((tmp / "bad_snapshot").exists())

            placeholder = tmp / "placeholder.json"
            placeholder.write_text(json.dumps({"source_receivers": "two source receivers"}), encoding="utf-8")
            placeholder_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "paper_reproduction.scripts.make_repro_snapshot",
                    "--config",
                    str(placeholder),
                    "--command",
                    "python train.py",
                    "--out-dir",
                    str(tmp / "placeholder_snapshot"),
                    "--formal",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(placeholder_run.returncode, 0)
            self.assertIn("unresolved placeholder", placeholder_run.stderr)

    def test_paper_original_matrix_records_wisig_paper_settings(self):
        matrix = ROOT / "paper_reproduction" / "paper_original_matrix.md"
        text = matrix.read_text(encoding="utf-8")
        for required in (
            "Cross-Domain Adaptation for RF Fingerprinting Using Prototypical Networks",
            "Few-shot Cross-Receiver Radio Frequency Fingerprinting Identification Based on Feature Separation",
            "WiSig",
            "Adam",
            "0.005",
            "batch size 256",
            "30 samples per transmitter",
            "25 samples/class",
            "Euclidean distance",
            "SGD",
        ):
            self.assertIn(required, text)
        self.assertIn("paper-unspecified", text)
        self.assertIn("implementation choice", text)

    def test_train_entrypoints_dry_run_and_reject_unresolved_formal_config(self):
        good = ROOT / "paper_reproduction" / "configs" / "feature_separation_crossrx_smoke.json"
        dry = subprocess.run(
            [
                sys.executable,
                "-m",
                "paper_reproduction.feature_separation_crossrx.train",
                "--config",
                str(good),
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("feature_separation_crossrx", dry.stdout)

        unresolved = ROOT / "paper_reproduction" / "configs" / "protonet_cda_wisig.json"
        formal = subprocess.run(
            [
                sys.executable,
                "-m",
                "paper_reproduction.protonet_cda.train",
                "--config",
                str(unresolved),
                "--dry-run",
                "--formal",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(formal.returncode, 0)
        self.assertIn("paper-unspecified", formal.stderr)


if __name__ == "__main__":
    unittest.main()
