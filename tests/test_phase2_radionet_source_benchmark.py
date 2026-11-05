"""The benchmark must exercise both norms at the real step's lookup point."""
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pamf_phase2_practical import group2_radionet_source_benchmark as bench


def test_benchmark_switches_step_norm_and_restores_it(monkeypatch):
    class SourceRows:
        def __getitem__(self, indices):
            return self

        def to(self, device):
            return torch.zeros(1)

    monkeypatch.setattr(bench, "activate", lambda rt: None)
    monkeypatch.setattr(bench, "source_scope", lambda rt: {})
    monkeypatch.setattr(bench, "load_source", lambda *args: {
        "model": torch.nn.Linear(1, 1), "source_checkpoint": "source-only"})
    monkeypatch.setattr(bench, "source_data", lambda *args: SimpleNamespace(
        metadata={"y": np.repeat(np.arange(6), 27)}, x=SourceRows()))
    monkeypatch.setattr(bench, "CrossAdaptationAdapter", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(bench.adaptation, "RadioNetADAHead", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "synthetic CPU")
    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: None)
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: tensor(
        *args, **{k: v for k, v in kwargs.items() if k != "device"}))
    original = bench.adaptation._grad_norm
    used = []

    def step(model, discriminator, *args):
        used.append(bench.adaptation._grad_norm)
        bench.adaptation._grad_norm(model)
        bench.adaptation._grad_norm(discriminator)

    monkeypatch.setattr(bench.adaptation, "radionet_ada_batch_step", step)
    result = bench.run({"c_source_roots": {"mopc_hr": "source"}, "model_seeds": [1]})
    assert result["same_gradient"] and len({id(norm) for norm in used}) == 3
    assert bench.adaptation._grad_norm is original
