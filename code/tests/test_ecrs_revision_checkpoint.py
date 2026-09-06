from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch import nn

from cvsrffi.checkpoint import AveragedModelState, save_checkpoint
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.ecrs_config import ecrs_model_config
from model_dual_cvsincnet import build_dual_model


class ExtraStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))
        self.extra = {"schema": "v2", "nested": [1, {"value": 2}]}

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, value):
        self.extra = value


def test_averaging_deepcopies_extra_state():
    model = ExtraStateModel()
    avg = AveragedModelState("ema", .5)
    avg.update(model, 1, ema=True)
    model.extra["nested"][1]["value"] = 99
    with torch.no_grad():
        model.weight.fill_(4.)
    assert avg.cpu_state_dict(model)["_extra_state"]["nested"][1]["value"] == 2
    avg.update(model, 2, ema=True)
    state = avg.cpu_state_dict(model)
    assert state["weight"].item() == 3.
    assert state["_extra_state"]["nested"][1]["value"] == 99
    state["_extra_state"]["nested"].append(100)
    assert len(avg.cpu_state_dict(model)["_extra_state"]["nested"]) == 2


@pytest.mark.parametrize("version", ["v1r", "v2"])
def test_revision_save_schema_runtime_and_exact_route(tmp_path, version):
    torch.set_num_threads(1)
    args = SimpleNamespace(num_classes=3, num_domains=2, model_size="M", dataset="wisig",
                           input_len=64, sample_rate_hz=25e6, model_variant="lite_d",
                           branch_ablation="no_dac", domain_branch_ablation="no_stats",
                           use_ecrs=True, ecrs_version=version, _not_serializable=lambda: None)
    config = ecrs_model_config(args)
    model = build_dual_model(3, 2, model_size="M", dataset="wisig", input_len=64,
                             model_variant="lite_d", branch_ablation="no_dac",
                             domain_branch_ablation="no_stats", use_ecrs=True,
                             ecrs_config=config).eval()
    class Runtime:
        def state_dict(self):
            return {"steps": 3, "rng": torch.get_rng_state()}
    model._ecrs_runtime = Runtime()
    model._ecrs_training_data_state = {"rng": {"torch": torch.get_rng_state()}, "l_sampler": {"epoch": 2}}
    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        before = model.forward_identity(x)["tx_logits"]
    path = tmp_path / (version + ".pth")
    save_checkpoint(str(path), model=model, optimizer=None, scheduler=None, scaler=None,
                    epoch=3, args=args, split_info={}, stats={})
    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert payload["feature_schema"].endswith(":" + version)
    assert "_not_serializable" not in payload["args"]
    assert payload["ecrs_runtime"]["steps"] == 3
    assert payload["training_data_state"]["l_sampler"]["epoch"] == 2
    restored, audit = build_exact_ssdg_model_from_checkpoint(payload, input_len=64, device=torch.device("cpu"))
    assert audit["ecrs_version"] == version
    restored.eval()
    with torch.no_grad():
        after = restored.forward_identity(x)["tx_logits"]
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    payload["feature_schema"] = "ADV3B02:ECRS:z_fused:unit_l2:160:v1"
    with pytest.raises(ValueError, match="schema"):
        build_exact_ssdg_model_from_checkpoint(payload, input_len=64, device=torch.device("cpu"))


def test_b0_preserves_training_rng_without_runtime(tmp_path):
    model = ExtraStateModel()
    model._ecrs_training_data_state = {"rng": {"torch": torch.get_rng_state()}}
    path = tmp_path / "b0.pth"
    save_checkpoint(str(path), model=model, optimizer=None, scheduler=None, scaler=None,
                    epoch=1, args=SimpleNamespace(), split_info={}, stats={})
    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert payload["ecrs_runtime"] is None
    assert torch.equal(payload["training_data_state"]["rng"]["torch"], torch.get_rng_state())


def test_public_reference_restores_without_original_files(tmp_path):
    torch.set_num_threads(1)
    reference_path = tmp_path / "public.npy"
    metadata_path = tmp_path / "public_metadata.json"
    reference = np.exp(1j * np.linspace(0, 12, 64)).astype(np.complex64)
    np.save(reference_path, reference)
    metadata_path.write_text('{"reference_version":"public_test_v1","alignment":"not_claimed"}', encoding="utf-8")
    args = SimpleNamespace(num_classes=3, model_size="M", dataset="wisig", model_variant="lite_d",
                           branch_ablation="no_dac", domain_branch_ablation="no_stats",
                           use_ecrs=True, ecrs_version="v2", ecrs_reference_mode="protocol_reference",
                           ecrs_protocol_reference_path=str(reference_path),
                           ecrs_protocol_reference_metadata=str(metadata_path))
    config = ecrs_model_config(args)
    model = build_dual_model(3, 2, model_size="M", dataset="wisig", input_len=64,
                             model_variant="lite_d", branch_ablation="no_dac",
                             domain_branch_ablation="no_stats", use_ecrs=True,
                             ecrs_config=config).eval()
    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        before = model.forward_identity(x)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(str(path), model=model, optimizer=None, scheduler=None, scaler=None,
                    epoch=1, args=args, split_info={}, stats={})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reference_path.unlink()
    metadata_path.unlink()
    restored, _ = build_exact_ssdg_model_from_checkpoint(payload, input_len=64, device=torch.device("cpu"))
    restored.eval()
    assert restored.ecrs.physical.reference_mode == "public_reference"
    assert restored.ecrs_config["protocol_reference_metadata"] == str(metadata_path)
    with torch.no_grad():
        after = restored.forward_identity(x)
    for key in ("tx_logits", "resp_tx_logits", "resp_coef"):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="existing explicit file"):
        ecrs_model_config(args)  # New training still requires explicit source artifacts.
    del payload["model"]["ecrs.physical.public_reference"]
    with pytest.raises(ValueError, match="missing its public waveform"):
        build_exact_ssdg_model_from_checkpoint(payload, input_len=64, device=torch.device("cpu"))
