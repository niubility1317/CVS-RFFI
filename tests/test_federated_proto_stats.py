import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" if (ROOT / "code" / "train.py").exists() else ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class FederatedPrototypeStatsTest(unittest.TestCase):
    def test_federated_clients_exchange_global_class_prototypes(self):
        from federated.fed_trainer import FederatedTrainer

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.items = [
                    {"iq": torch.tensor([1.0, 0.0]), "label": 0, "receiver": 0, "day": 0},
                    {"iq": torch.tensor([0.8, 0.2]), "label": 1, "receiver": 0, "day": 0},
                    {"iq": torch.tensor([0.0, 1.0]), "label": 0, "receiver": 1, "day": 0},
                    {"iq": torch.tensor([0.2, 0.8]), "label": 1, "receiver": 1, "day": 0},
                ]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[int(idx)]

        class TinyProtoModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 3)
                self.classifier = nn.Linear(3, 2)

            def forward(self, x, **kwargs):
                z = self.encoder(x.float())
                return {
                    "tx_logits": self.classifier(z),
                    "z_id": z,
                    "z_dom": z.flip(dims=[1]),
                }

        def eval_loader(model, loader, device, **kwargs):
            return {"tx_acc": 0.0, "tx_correct": 0, "tx_total": 1}

        def eval_named(model, loaders, device, **kwargs):
            return {"test_unseen_day_unseen_rx": {"tx_acc": 0.0, "tx_correct": 0, "tx_total": 1}}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                train_mode="fedavg",
                fl_client_key="receiver",
                fl_rounds=2,
                fl_local_epochs=1,
                fl_clients_per_round=1.0,
                fl_agg_weight="num_samples",
                fl_local_objective="ce",
                batch_size=2,
                num_workers=0,
                seed=1337,
                lr=1e-3,
                lr_min=1e-3,
                wd=0.0,
                grad_clip=1.0,
                output_dir=tmp,
                eval_max_batches=0,
                use_fed_proto_stats=True,
                lambda_fed_proto=0.5,
                fed_proto_min_count=1,
                fed_proto_momentum=0.0,
            )
            trainer = FederatedTrainer(
                TinyProtoModel(),
                TinyDataset(),
                val_loader=[],
                named_test_loaders={"test_unseen_day_unseen_rx": []},
                cfg=cfg,
                device=torch.device("cpu"),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=eval_named,
            )
            summary = trainer.train()
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["global_proto_summary"]["class_count_nonzero"], 2)
        self.assertEqual(rows[0]["global_proto_summary"]["class_count_nonzero"], 2)
        self.assertEqual(rows[1]["global_proto_summary"]["class_count_nonzero"], 2)
        self.assertIn("client_loss_fed_proto_avg", rows[1])
        self.assertGreaterEqual(rows[1]["client_loss_fed_proto_avg"], 0.0)

    def test_train_cli_exposes_federated_prototype_switches(self):
        text = (CODE / "train.py").read_text(encoding="utf-8")
        self.assertIn('"use_fed_proto_stats"', text)
        self.assertIn('"--lambda_fed_proto"', text)
        self.assertIn('"--fed_proto_momentum"', text)


if __name__ == "__main__":
    unittest.main()
