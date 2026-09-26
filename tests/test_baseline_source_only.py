import json
import tempfile
import unittest
from pathlib import Path

import torch

from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.pseudo_labels import PseudoLabelConfig, compute_pseudo_label_loss
from baselines.common.pseudo_labels import build_pseudo_step_fn


class NoTruthBatch(dict):
    def __contains__(self, key):
        if key in {"label", "true_label"}:
            raise AssertionError("Unlabeled truth was inspected")
        return super().__contains__(key)

    def __getitem__(self, key):
        if key in {"label", "true_label"}:
            raise AssertionError("Unlabeled truth was read")
        return super().__getitem__(key)


class SourceOnlyTests(unittest.TestCase):
    def test_pseudo_step_updates_both_disjoint_optimizers(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
        head = torch.optim.SGD(model[2].parameters(), lr=.1)
        backbone = torch.optim.SGD(model[0].parameters(), lr=.1)
        before = model[0].weight.detach().clone()
        step = build_pseudo_step_fn(cfg=PseudoLabelConfig(enabled=True, start_epoch=1, threshold=0),
            loader=[{'iq': torch.ones(3, 2)}], optimizer=head, additional_optimizers=(backbone,))
        step(model, torch.device('cpu'), 1, 0)
        self.assertFalse(torch.equal(before, model[0].weight))

    def test_pseudo_loss_does_not_inspect_truth(self):
        model = torch.nn.Linear(2, 2)
        result = compute_pseudo_label_loss(
            model, NoTruthBatch(iq=torch.ones(3, 2)), torch.device("cpu"),
            PseudoLabelConfig(enabled=True, start_epoch=1, threshold=0), epoch=1,
        )
        self.assertEqual(result.selected, 3)
        self.assertNotIn("pseudo/precision", result.metrics)
        result.loss.backward()

    def test_source_only_saves_final_without_test_metrics(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = {"iq": torch.ones(2, 2), "label": torch.tensor([0, 1])}

        def train_step(model, batch, device, epoch, step):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch["iq"]), batch["label"])
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}

        with tempfile.TemporaryDirectory() as directory:
            history = run_validation_gated_training(
                model=model, train_loader=[batch], val_loader=[batch],
                named_test_loaders={}, device=torch.device("cpu"), epochs=2,
                optimizer=optimizer, train_step_fn=train_step, output_dir=directory,
                source_only=True,
            )
            payload = torch.load(Path(directory) / "last.pt", weights_only=False)
            self.assertEqual(payload["epoch"], 2)
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, payload["model"][key]))
            metrics = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(metrics["final"]["status"], "SOURCE_TRAINED")
            self.assertNotIn("test_overall", metrics["final"])
            self.assertTrue(all(not row["tested"] for row in history.epochs))

    def test_source_only_rejects_target_inputs_before_training(self):
        for targets, extra in [({"target": object()}, None), ({}, lambda *a: {})]:
            with self.subTest(targets=bool(targets), extra=bool(extra)):
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(ValueError, "source_only"):
                        run_validation_gated_training(
                            model=None, train_loader=None, val_loader=None,
                            named_test_loaders=targets, device="cpu", epochs=1,
                            optimizer=None, train_step_fn=None, output_dir=directory,
                            source_only=True, extra_test_fn=extra,
                        )


if __name__ == "__main__":
    unittest.main()
