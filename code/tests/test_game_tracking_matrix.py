import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from cvsrffi.game_tracking.config import parse_args
from scripts.build_core90_game_matrix import DEFAULT_ROWS, build_matrix, row_menu, write_matrix
import scripts.core90_game_evaluate as evaluation


def test_default_matrix_paired_seed_scratch_source_only():
    matrix = build_matrix()
    assert len(matrix) == 22
    assert {row["row"] for row in matrix} == set(DEFAULT_ROWS)
    assert {row["seed"] for row in matrix} == {392002, 392003}
    for row in matrix:
        config = row["config"]
        assert config["from_scratch"] and config["baseline_ckpt"] == config["game_resume"] == ""
        assert config["epochs"] == 200 and config["game_split_seed"] == 392002
        assert [config[k] for k in ("labeled_ratio", "unlabeled_ratio", "source_val_ratio")] == [.07, .63, .30]
    assert next(r for r in matrix if r["row"] == "B0")["config"]["game_no_audit"]
    assert next(r for r in matrix if r["row"] == "B2")["config"]["lambda_adv"] == 0.


def test_all_configs_parse_and_replay_is_explicit(tmp_path):
    assert "C4" not in row_menu()
    with pytest.raises(ValueError, match="replay path"):
        build_matrix(rows=["C4"])
    schedule = tmp_path / "source_actions.jsonl"
    schedule.write_text("", encoding="utf-8")
    manifest = write_matrix(tmp_path / "matrix", rows=tuple(row_menu(str(schedule))),
                            seeds=[392003], replay_path=str(schedule))
    assert manifest["status"] == "CONFIGURED_NOT_LAUNCHED" and not manifest["target_evaluation"]
    for row in manifest["runs"]:
        args = parse_args(row["argv"][1:])
        assert args.seed == 392003
        assert args.output_dir == row["config"]["output_dir"]
    with pytest.raises(FileExistsError):
        write_matrix(tmp_path / "matrix", rows=tuple(row_menu(str(schedule))),
                     seeds=[392003], replay_path=str(schedule))


class TinyValidation(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        truth = index % 2
        x = torch.eye(2)[truth]
        meta = {"sample_id": f"sample-{index}", "rx_i": index // 2, "day_i": 0,
                "role": "val", "capture_group": f"hidden-tx-{truth}"}
        return x, truth, index // 2, meta


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.linear.weight.data.copy_(torch.eye(2))
        self.register_buffer("unchanged", torch.ones(1))

    def forward(self, x, *, return_aux=True):
        return {"tx_logits": self.linear(x)}


def test_evaluator_closes_all_predictions_before_truth_and_preserves_state(tmp_path, monkeypatch):
    model = TinyModel().train()
    source = SimpleNamespace(val=TinyValidation())
    args = SimpleNamespace(device="cpu", game_split_seed=8, eval_batch_size=3, num_classes=2, game_synthetic=True)
    original_truth = evaluation.source_truth_records
    observations = []

    def assert_closed(source):
        manifest = json.loads((tmp_path / "source_prediction_manifest.json").read_text())
        rows = list(evaluation._json_rows(tmp_path / "source_predictions.jsonl"))
        assert manifest["complete"] and len(rows) == 16
        assert all("truth" not in row and "capture_group" not in row for row in rows)
        observations.append("all_predictions_fixed_before_truth")
        yield from original_truth(source)

    monkeypatch.setattr(evaluation, "source_truth_records", assert_closed)
    rng = torch.get_rng_state().clone()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    results = evaluation.evaluate_source(model, source, args, tmp_path,
                                         apply_scene=lambda x, scene, generator: x)
    assert observations == ["all_predictions_fixed_before_truth"]
    assert results["complete"] and results["prediction_count"] == 16
    assert set(results["scenes"]) == set(evaluation.EVAL_SCENES)
    assert results["scenes"]["clean"]["accuracy"] == 1.
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        evaluation.evaluate_source(model, source, args, tmp_path, apply_scene=lambda x, s, g: x)


def _prediction_fixture(tmp_path):
    truth = [{"sample_id": str(i), "truth": i % 2, "rx_i": i // 2, "day_i": 0} for i in range(4)]
    predictions = [{"sample_id": row["sample_id"], "scene": scene,
                    "rx_i": row["rx_i"], "day_i": row["day_i"],
                    "prediction": 0, "confidence": .7}
                   for scene in evaluation.EVAL_SCENES for row in truth]
    path = tmp_path / "predictions.jsonl"
    return truth, predictions, path


def test_independent_group_metrics_and_completeness(tmp_path):
    truth, predictions, path = _prediction_fixture(tmp_path)
    path.write_text("".join(json.dumps(row)+"\n" for row in predictions), encoding="utf-8")
    result = evaluation.score_source_predictions(path, truth, expected_samples=4, num_classes=2)
    clean = result["scenes"]["clean"]
    assert clean["accuracy"] == clean["macro_recall"] == clean["macro_rx_accuracy"] == .5
    assert clean["macro_f1"] == pytest.approx(1/3)
    assert clean["worst_tx_accuracy"] == 0.
    assert clean["worst_rx_accuracy"] == .5
    path.write_text("".join(json.dumps(row)+"\n" for row in predictions[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        evaluation.score_source_predictions(path, truth, expected_samples=4, num_classes=2)


@pytest.mark.parametrize("mutation", ["duplicate", "hidden_truth", "target_id", "nan", "bad_class"])
def test_prediction_artifact_negative_cases(tmp_path, mutation):
    truth, predictions, path = _prediction_fixture(tmp_path)
    if mutation == "duplicate":
        predictions.append(predictions[0])
    elif mutation == "hidden_truth":
        predictions[0]["truth"] = 0
    elif mutation == "target_id":
        predictions[0]["sample_id"] = "target"
    elif mutation == "nan":
        predictions[0]["confidence"] = float("nan")
    elif mutation == "bad_class":
        predictions[0]["prediction"] = 2
    path.write_text("".join(json.dumps(row)+"\n" for row in predictions), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.score_source_predictions(path, truth, expected_samples=4, num_classes=2)


class TinyIQValidation(TinyValidation):
    def __getitem__(self, index):
        x, truth, domain, meta = super().__getitem__(index)
        iq = x[:, None].expand(2, 256).clone()
        iq += torch.sin(torch.arange(256.) * .2)[None, :] * .1
        return iq, truth, domain, meta


class TinyIQModel(TinyModel):
    def forward(self, x, *, return_aux=True):
        return {"tx_logits": self.linear(x.mean(-1))}


def test_default_physical_leo_evaluation_is_deterministic(tmp_path):
    model = TinyIQModel()
    source = SimpleNamespace(val=TinyIQValidation())
    args = SimpleNamespace(device="cpu", game_split_seed=392002, eval_batch_size=2,
                           num_classes=2, game_synthetic=True)
    evaluation.evaluate_source(model, source, args, tmp_path / "first")
    evaluation.evaluate_source(model, source, args, tmp_path / "second")
    assert (tmp_path / "first/source_predictions.jsonl").read_bytes() == (tmp_path / "second/source_predictions.jsonl").read_bytes()
