from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from SSDG import train_ssdg


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "smoke_phase1_bicad_xr_real_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("bicad_xr_real_checkpoint_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _SmokeIQModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Linear(128, 8)
        self.domain = nn.Linear(128, 8)
        self.classifier = nn.Linear(8, 6)
        self.forward_batch_sizes: list[int] = []
        self.received_tx_history: list[torch.Tensor | None] = []

    def forward(
        self,
        x: torch.Tensor,
        y_tx: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
        **_: object,
    ):
        self.forward_batch_sizes.append(int(x.size(0)))
        self.received_tx_history.append(
            None if y_tx is None else y_tx.detach().clone()
        )
        del y_tx, domain_labels
        flat = x.flatten(1)
        z_id = self.identity(flat)
        z_dom = self.domain(flat)
        logits = self.classifier(z_id)
        if not return_aux:
            return logits
        return {
            "tx_logits": logits,
            "z_id": z_id,
            "z_dom": z_dom,
            "shared_features": flat,
            "identity_features": z_id,
            "domain_features": z_dom,
        }


def _checkpoint_loader(_: Path):
    return {"args": {"input_len": 64}, "model": {"placeholder": torch.ones(1)}}


def _strict_rebuilder(payload, *, input_len, device, ssdg_module):
    del payload, ssdg_module
    assert input_len == 64
    return _SmokeIQModel().to(device), {
        "loader": "test_exact_structure",
        "checkpoint_load_strict": True,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
    }


def test_real_checkpoint_smoke_runs_optimizer_and_four_registered_scenarios(tmp_path) -> None:
    checkpoint = tmp_path / "historical_adv3b02.pth"
    checkpoint.write_bytes(b"fixture")

    result = smoke.run_smoke(
        checkpoint,
        output_dir=tmp_path / "out",
        seed=392002,
        checkpoint_loader=_checkpoint_loader,
        model_rebuilder=_strict_rebuilder,
        ssdg_module=train_ssdg,
    )

    assert result["status"] == "PASS"
    assert result["optimizer_step_complete"] is True
    assert result["backward_controls_complete"] is True
    assert isinstance(result["backward_controls"], dict)
    assert result["four_scenarios_complete"] is True
    assert set(result["evaluations"]) == set(smoke.SCENARIOS)
    assert all(row["finite"] for row in result["evaluations"].values())
    assert result["source_only"] is True
    assert result["target_access"] is False
    assert result["phase2_access"] is False
    assert result["query_access"] is False
    assert result["concat_forward_batch_size"] == 16
    persisted = json.loads((tmp_path / "out" / "smoke_result.json").read_text("utf-8"))
    assert persisted == result


def test_pairbicad_real_checkpoint_smoke_uses_16l32u_single_forward_without_u_tx(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "historical_adv3b02.pth"
    checkpoint.write_bytes(b"fixture")
    rebuilt: dict[str, _SmokeIQModel] = {}

    def rebuilder(payload, *, input_len, device, ssdg_module):
        model, audit = _strict_rebuilder(
            payload,
            input_len=input_len,
            device=device,
            ssdg_module=ssdg_module,
        )
        rebuilt["model"] = model
        return model, audit

    result = smoke.run_smoke(
        checkpoint,
        output_dir=tmp_path / "out",
        candidate_id="P4",
        seed=392002,
        checkpoint_loader=_checkpoint_loader,
        model_rebuilder=rebuilder,
        ssdg_module=train_ssdg,
    )

    assert result["status"] == "PASS"
    assert result["strict_reconstruction"] is True
    assert result["checkpoint_load_strict"] is True
    assert result["strict_pair_concat"] is True
    assert result["physical_batch_size"] == 48
    assert result["labeled_count"] == 16
    assert result["unlabeled_count"] == 32
    assert result["concat_forward_batch_size"] == 96
    assert result["satellite_training_scenario"] in {
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    }
    assert result["unlabeled_tx_access"] is False
    assert result["four_scenarios_complete"] is True
    assert set(result["evaluations"]) == set(smoke.SCENARIOS)
    assert result["query_access"] is False
    assert result["target_access"] is False
    assert result["phase2_access"] is False
    assert result["truth_access"] is False
    model = rebuilt["model"]
    assert model.forward_batch_sizes[0] == 96
    assert model.received_tx_history[0] is None
    assert all(row["finite"] for row in result["evaluations"].values())


def test_real_checkpoint_smoke_fails_closed_on_reconstruction_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "historical_adv3b02.pth"
    checkpoint.write_bytes(b"fixture")

    def mismatched(*args, **kwargs):
        del args, kwargs
        return _SmokeIQModel(), {
            "missing_keys": 0,
            "unexpected_keys": 1,
            "skipped_mismatch": 0,
        }

    with pytest.raises(RuntimeError, match="strict checkpoint reconstruction failed"):
        smoke.run_smoke(
            checkpoint,
            output_dir=tmp_path / "out",
            checkpoint_loader=_checkpoint_loader,
            model_rebuilder=mismatched,
            ssdg_module=train_ssdg,
        )
    assert not (tmp_path / "out" / "smoke_result.json").exists()


def test_real_checkpoint_smoke_requires_recorded_input_length(tmp_path) -> None:
    checkpoint = tmp_path / "historical_adv3b02.pth"
    checkpoint.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="positive input length"):
        smoke.run_smoke(
            checkpoint,
            output_dir=tmp_path / "out",
            checkpoint_loader=lambda _: {"args": {}, "model": {}},
            model_rebuilder=_strict_rebuilder,
            ssdg_module=train_ssdg,
        )
