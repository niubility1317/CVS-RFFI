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


class FederatedSatelliteEvalTest(unittest.TestCase):
    def test_federated_round_records_satellite_extra_tests(self):
        from federated.fed_trainer import FederatedTrainer

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.items = [
                    {"iq": torch.tensor([1.0, 0.0]), "label": 0, "receiver": 0, "day": 0},
                    {"iq": torch.tensor([0.0, 1.0]), "label": 1, "receiver": 1, "day": 0},
                ]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[int(idx)]

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x, **kwargs):
                return {"tx_logits": self.linear(x.float())}

        sat_calls = []

        def evaluate_loader_fn(model, loader, device, **kwargs):
            return {"tx_acc": 50.0, "tx_correct": 1, "tx_total": 2}

        def evaluate_named_loaders_fn(model, loaders, device, **kwargs):
            return {
                "test_unseen_day_unseen_rx": {
                    "tx_acc": 25.0,
                    "tx_correct": 1,
                    "tx_total": 4,
                }
            }

        def extra_eval_fn(model, device, round_idx):
            sat_calls.append(round_idx)
            return {
                "sat_channel": {
                    "clear_leo": {
                        "aggregate": {"tx_acc": 12.5, "tx_correct": 1, "tx_total": 8},
                        "selected_names": ["test_unseen_day_unseen_rx"],
                    }
                }
            }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                train_mode="fedavg",
                fl_client_key="receiver",
                fl_rounds=1,
                fl_local_epochs=1,
                fl_clients_per_round=1.0,
                fl_agg_weight="num_samples",
                fl_local_objective="ce",
                batch_size=1,
                num_workers=0,
                seed=1337,
                lr=1e-3,
                lr_min=1e-3,
                wd=0.0,
                grad_clip=1.0,
                output_dir=tmp,
                eval_max_batches=0,
            )
            trainer = FederatedTrainer(
                TinyModel(),
                TinyDataset(),
                val_loader=[],
                named_test_loaders={"test_unseen_day_unseen_rx": []},
                cfg=cfg,
                device=torch.device("cpu"),
                evaluate_loader_fn=evaluate_loader_fn,
                evaluate_named_loaders_fn=evaluate_named_loaders_fn,
                extra_eval_fn=extra_eval_fn,
            )

            summary = trainer.train()
            log_row = json.loads((Path(tmp) / "logs.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(sat_calls, [1])
        self.assertIn("sat_channel", summary["best_eval"]["extra_tests"])
        self.assertIn("sat_channel", summary["last_eval"]["extra_tests"])
        self.assertIn("sat_channel", log_row["global_extra_tests"])

    def test_main_train_defaults_to_satellite_evaluation_enabled(self):
        train_py = (CODE / "train.py").read_text(encoding="utf-8")
        self.assertIn('add_bool_arg(parser, "eval_sat_channel", True,', train_py)


if __name__ == "__main__":
    unittest.main()
