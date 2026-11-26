import pytest
import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_packet_unet import weak_private_gain_risk
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, snapshot_objective, trial


def test_private_gain_detects_attenuation_and_quadrature_leakage():
    torch.manual_seed(67)
    phase = torch.arange(384) * (2 * torch.pi * 17 / 384)
    target = torch.polar(torch.ones_like(phase), phase) * torch.randn(2, 8, 1) * .04
    target = target - target.mean(-2, keepdim=True)
    full = weak_private_gain_risk(target, target)
    half = weak_private_gain_risk(.5 * target, target)
    rotated = (1j * target).requires_grad_()
    quadrature = weak_private_gain_risk(rotated, target)
    assert full == 0 and full < half < quadrature
    quadrature.backward()
    assert torch.isfinite(rotated.grad).all() and rotated.grad.norm() > 0
    assert weak_private_gain_risk(rotated, torch.zeros_like(target)) == 0


def test_snapshot_captures_post_step_24000_and_protects_existing_artifact(tmp_path):
    class ArchivedTrainer:
        @staticmethod
        def proof(contract, recipe):
            return dict(contract=contract, steps=recipe.steps)

    def calculate(model, _tracker, raw, _rng, **_):
        return model(raw)

    from dataclasses import dataclass

    @dataclass
    class Recipe:
        steps: int = 48000

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.)
    path = tmp_path / 'w24_example-step24000.pt'
    wrapped = snapshot_objective(calculate, ArchivedTrainer, {}, path, 24000)
    raw = torch.ones(1, 1)
    for step, identity in ((24000, True), (24001, False)):
        wrapped(model, None, raw, None, step=step, with_identity=identity, recipe=Recipe())
        assert not path.exists()
    torch.testing.assert_close(wrapped(model, None, raw, None, step=24001, with_identity=True,
                                       recipe=Recipe()), torch.full_like(raw, 2.))
    model.weight.data.fill_(3.)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['proof'] == dict(contract={}, steps=24000)
    torch.testing.assert_close(checkpoint['weights']['weight'], torch.full_like(raw, 2.))
    with pytest.raises(FileExistsError):
        wrapped(model, None, raw, None, step=24001, with_identity=True, recipe=Recipe())


def test_intermediate_predictions_are_source_only_and_fit_sees_only_raw_u():
    kind = 'weak-transfer'
    spec = trial(kind)
    assert spec.steps == 48000 and spec.source_intermediate == 24000
    assert set(cases(kind)) == {f'{FULL}-{i}' for i in range(3)}
    assert spec.source_modes[-2:] == tuple(f'{arm}_s24000_half' for arm in spec.arms)
    formal = evaluation_modes(FULL, True, third_prior=kind)
    assert formal[-2:] == spec.source_modes[2:4] and not set(formal) & set(spec.source_modes[-2:])
    runtime = dict(source_root='U', run_root='new', classifier_root='forbidden F', native_target_root='forbidden target')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='U', run_root=f'new/{arm}')
