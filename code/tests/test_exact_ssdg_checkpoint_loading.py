from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint


class _Parser:
    def parse_args(self, _argv):
        return argparse.Namespace(output_dir="")


class _FakeSSDG:
    def __init__(self, *, add_extra_parameter: bool = False) -> None:
        self.add_extra_parameter = bool(add_extra_parameter)
        self.seen = None

    def build_arg_parser(self):
        return _Parser()

    def merge_checkpoint_args(self, checkpoint, parsed, *, input_len, num_domains):
        self.seen = {
            "checkpoint": checkpoint,
            "parsed": parsed,
            "input_len": input_len,
            "num_domains": num_domains,
        }
        return parsed

    def _apply_model_cli_args(self, merged, _parsed):
        return merged

    def build_baseline_model(self, _merged, _device):
        model = torch.nn.Module()
        model.dom_head = torch.nn.Module()
        model.dom_head.net = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Linear(2, 3),
        )
        if self.add_extra_parameter:
            model.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
        return model


def _checkpoint() -> dict:
    reference = _FakeSSDG().build_baseline_model(None, None)
    state = {f"module.{key}": value.clone() for key, value in reference.state_dict().items()}
    return {"args": {"model_variant": "dual"}, "model": state}


def test_exact_loader_uses_checkpoint_args_and_loads_every_tensor() -> None:
    fake = _FakeSSDG()
    model, audit = build_exact_ssdg_model_from_checkpoint(
        _checkpoint(),
        input_len=256,
        device=torch.device("cpu"),
        ssdg_module=fake,
    )
    assert isinstance(model, torch.nn.Module)
    assert fake.seen["parsed"].model_variant == "dual"
    assert fake.seen["input_len"] == 256
    assert fake.seen["num_domains"] == 3
    assert audit["checkpoint_load_strict"] is True
    assert audit["crra_enabled"] is False
    assert audit["missing_keys"] == 0
    assert audit["unexpected_keys"] == 0
    assert audit["skipped_mismatch"] == 0


def test_exact_loader_fails_closed_on_missing_parameter() -> None:
    with pytest.raises(ValueError, match="strict checkpoint reconstruction failed"):
        build_exact_ssdg_model_from_checkpoint(
            _checkpoint(),
            input_len=256,
            device=torch.device("cpu"),
            ssdg_module=_FakeSSDG(add_extra_parameter=True),
        )
