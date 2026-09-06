from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from cvsrffi.ecrs_evaluation import LEO_SCENARIOS, choose_source_score, evaluate_revision_paths


class IdentityOnly(nn.Module):
    def __init__(self, active=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.register_buffer("counter", torch.tensor(0))
        self.child = nn.Dropout()
        self._ecrs_fusion_head_initialized = active

    def forward_identity(self, x):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.counter.add_(1)  # Even a badly behaved module must not change training state.
        self.weight.add_(1)
        torch.rand(1)
        np.random.rand()
        random.random()
        return {"tx_logits_raw_infer": x[:, 0, :], "resp_tx_logits": x[:, 1, :],
                "tx_logits_fused": x[:, 2, :], "z_id_fused": x[:, 2, :]}

    def forward(self, *args, **kwargs):
        raise AssertionError("domain/main forward forbidden")

    def _classify_identity_feature(self, z, y):
        assert y is None
        return z


def batch(labels=None):
    # truth=0,1,0,1; raw=0,0,1,1; fused=0,1,0,0: rescue2/harm1.
    raw = torch.tensor([[3., 0.], [3., 0.], [0., 3.], [0., 3.]])
    fused = torch.tensor([[3., 0.], [0., 3.], [3., 0.], [3., 0.]])
    x = torch.stack((raw, fused, fused), dim=1)
    meta = {"physical_sample_id": ["a", "b", "c", "d"],
            "receiver_id": torch.tensor([1, 1, 3, 3]), "day_id": [1, 2, 1, 2]}
    return x, (torch.tensor([0, 1, 0, 1]) if labels is None else labels), torch.zeros(4), meta


def test_three_paths_scaling_predictions_and_state(tmp_path):
    model = IdentityOnly()
    model.train()
    model.child.eval()  # Preserve heterogeneous per-module modes as well.
    original = deepcopy(model.state_dict())
    torch_rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    target = tmp_path / "source.jsonl"
    result = evaluate_revision_paths(model, [batch()], "cpu", output_path=target)
    assert result["paths"]["raw"]["tx_acc"] == 50.
    assert result["paths"]["fused"]["tx_acc"] == 75.
    assert result["paths"]["raw"]["per_rx"]["1"]["tx_total"] == 2
    assert result["paths"]["fused"]["rx_floor"] == 50.
    assert (result["rescue"], result["harm"], result["net_pp"]) == (2, 1, 25.)
    assert model.training and not model.child.training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert py_rng == random.getstate()
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [record["physical_sample_id"] for record in records] == ["a", "b", "c", "d"]
    assert records[0]["source_truth"] == 0
    assert records[1]["predictions"]["fused"] == 1
    assert set(records[0]["logits"]) == {"raw", "response", "fused", "common_head"}


def test_inactive_fused_head_uses_raw():
    result = evaluate_revision_paths(IdentityOnly(active=False), [batch()], "cpu")
    assert result["paths"]["fused"]["tx_acc"] == 50.
    assert result["rescue"] == result["harm"] == 0
    assert not result["fusion_active"]


def test_b0_has_only_raw_path_and_can_select_source():
    class Baseline(nn.Module):
        def forward_identity(self, x):
            return {"tx_logits": x[:, 0, :]}
    result = evaluate_revision_paths(Baseline(), [batch()], "cpu")
    assert set(result["paths"]) == {"raw"}
    scenes = {name: result for name in ("clean",) + LEO_SCENARIOS}
    assert choose_source_score(scenes, "raw") == 50.


def test_truth_last_does_not_read_or_write_labels(tmp_path):
    class ForbiddenTruth:
        def __int__(self):
            raise AssertionError("truth read")

    target = tmp_path / "predictions.jsonl"
    result = evaluate_revision_paths(IdentityOnly(), [batch(ForbiddenTruth())], "cpu",
                                     source_metrics=False, output_path=target)
    assert result["paths"] == {}
    assert result["net"] is None
    assert "truth" not in target.read_text(encoding="utf-8")


def test_failure_restores_state_and_no_overwrite(tmp_path):
    model = IdentityOnly()
    state = deepcopy(model.state_dict())
    def fail(x, idx):
        raise RuntimeError("transform failed")
    with pytest.raises(RuntimeError):
        evaluate_revision_paths(model, [batch()], "cpu", transform=fail)
    assert model.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    target = tmp_path / "existing.jsonl"
    target.write_text("protected", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate_revision_paths(model, [batch()], "cpu", output_path=target)
    assert target.read_text(encoding="utf-8") == "protected"


def scene(raw, fused):
    return {"source_metrics": True, "fusion_active": True, "net": 2,
            "paths": {key: {"tx_acc": value, "per_rx": {"1": {"tx_acc": value}}}
                      for key, value in (("raw", raw), ("fused", fused))}}


def test_declared_source_selection_no_head_cherry_picking():
    scenes = {"clean": scene(80, 80), **{s: scene(50, 52) for s in LEO_SCENARIOS}}
    assert choose_source_score(scenes) == 50.
    assert choose_source_score(scenes, "fused") == 52.
    scenes["clean"] = scene(80, 79)
    assert choose_source_score(scenes, "fused") is None
    assert choose_source_score(scenes, "raw") == 50.
    with pytest.raises(ValueError):
        choose_source_score(scenes, "response")


@pytest.mark.parametrize("failure", ["missing", "nan", "target", "floor", "net", "inactive"])
def test_source_score_rejects_unusable_or_failed_candidates(failure):
    scenes = {"clean": scene(80, 80), **{s: scene(50, 52) for s in LEO_SCENARIOS}}
    if failure == "missing":
        del scenes[LEO_SCENARIOS[0]]
    elif failure == "nan":
        scenes["clean"]["paths"]["fused"]["tx_acc"] = float("nan")
    elif failure == "target":
        scenes["clean"]["source_metrics"] = False
    elif failure == "floor":
        scenes[LEO_SCENARIOS[0]]["paths"]["fused"]["per_rx"]["1"]["tx_acc"] = 48.
    elif failure == "net":
        for s in LEO_SCENARIOS:
            scenes[s]["net"] = 0
    else:
        scenes["clean"]["fusion_active"] = False
    assert choose_source_score(scenes, "fused") is None
