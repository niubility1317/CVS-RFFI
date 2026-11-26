from types import SimpleNamespace

import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_outer_objective_source import objective_for
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


def test_signed_supervision_changes_only_transfer_risk_on_identical_matched_views():
    window = 4
    raw = torch.ones(3, 8 * 1024, dtype=torch.complex64)
    probe = torch.tensor([.18, -.04, -.04, -.10], dtype=raw.dtype)
    seen = []

    def risk(output, target):
        return (output - target).abs().square().mean()

    def propose(y, _quality, _rms, edges):
        seen.append((y.clone(), edges.clone()))
        # An asymmetric response allows the two signed losses to differ.
        return (y, y, y + .5 * (y - 1.) + .5 * (y - 1.).square()), lambda x: x, y.real.new_ones(()), None, None

    archived = SimpleNamespace(
        WIDE=window,
        waveform_risk=risk,
        transfer_risk=risk,
        real_iq=lambda x: x,
        coordinates=lambda _tracker, x: (x, x.new_ones((*x.shape[:-1], 1)), None, 1.),
        weak_difference=lambda framed: probe.expand_as(framed),
        condition_waveforms=lambda x, _rng: x + .3,
        split=lambda x: x,
    )
    tracker = SimpleNamespace(aperture=lambda x: x.unflatten(-1, (8, 1024))[..., :window + 2].clone())
    model = SimpleNamespace(propose=propose)
    recipe = SimpleNamespace(loss_weights=lambda step: (1., 1., 1.))
    results = [objective_for(archived, signed_probe=both)(
        model, tracker, raw, None, with_identity=True, recipe=recipe, step=1)
        for both in (False, True)]

    for control, treated in zip(*seen, strict=True):
        torch.testing.assert_close(control, treated, rtol=0, atol=0)
    y = seen[0][0].unflatten(0, (3, 4))
    torch.testing.assert_close(y[:, 2] - y[:, 0], probe.expand_as(y[:, 0]))
    torch.testing.assert_close(y[:, 0] - y[:, 3], probe.expand_as(y[:, 0]))
    (control_loss, control), (signed_loss, signed) = results
    torch.testing.assert_close(control[4], control[-2])
    torch.testing.assert_close(signed[4], signed[-2:].mean())
    assert not torch.isclose(signed[-2], signed[-1])
    torch.testing.assert_close(control[:4], signed[:4])
    torch.testing.assert_close(control[5:], signed[5:])
    torch.testing.assert_close(signed_loss - control_loss, .30 * (signed[4] - control[4]))


def test_signed_probe_uses_only_registered_source_fit_paths_and_full_shards():
    plan = trial('signed-probe')
    assert plan.steps == 1800 and plan.routes == ('full_native',)
    assert set(cases('signed-probe')) == {f'{FULL}-{i}' for i in range(3)}
    for arm in plan.arms:
        assert fit_inputs(dict(source_root='U', source_parent='parent', run_root='run',
                               classifier_root='F', native_target_root='query'), arm) == {
            'source_root': 'U', 'source_parent': 'parent', 'run_root': f'run/{arm}',
        }
