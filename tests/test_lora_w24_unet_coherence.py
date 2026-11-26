import torch

from experiments.pamf_lora_main.w24_coherence_constraint import (
    coherence_risks, compatible_gradient, constrained_loss,
)
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import OUTER_BASELINE, UNET_BASELINE, fit_inputs, trial


def test_coherence_preserves_shared_phase_and_detects_relative_phase():
    torch.manual_seed(51)
    source = torch.randn(2, 2, 8, 384, dtype=torch.complex64)
    shared = .6 + .8j
    frequency, chirp = coherence_risks(shared * source, source)
    torch.testing.assert_close(frequency, torch.zeros_like(frequency), atol=1e-10, rtol=0.)
    torch.testing.assert_close(chirp, torch.zeros_like(chirp), atol=1e-10, rtol=0.)
    changed = source.clone()
    changed[..., ::2, :] *= shared
    freq_changed, chirp_changed = coherence_risks(changed, source)
    torch.testing.assert_close(freq_changed, torch.zeros_like(freq_changed), atol=1e-10, rtol=0.)
    assert chirp_changed > .01
    silent = torch.zeros_like(source, requires_grad=True)
    sum(coherence_risks(silent, source)).backward()
    assert torch.isfinite(silent.grad).all()


def test_projected_branch_caps_gradient_and_does_not_worsen_primary_at_readout():
    primary = torch.tensor([[2.+0j, 0j], [1j, 0j], [0j, 0j]])
    other = torch.tensor([[-3.+1j, 5j], [4j, 1j], [4j, 5j]])
    projected, conflict = compatible_gradient(primary, other)
    torch.testing.assert_close(conflict, torch.tensor(1/3))
    assert ((primary.conj() * projected).real.sum(-1) >= -1e-6).all()
    assert (projected.abs().square().sum(-1) <= primary.abs().square().sum(-1) + 1e-6).all()
    torch.testing.assert_close(projected[2], torch.zeros_like(projected[2]))

    for observe_only in (False, True):
        output = torch.ones_like(primary, requires_grad=True)
        original = (primary.conj() * output).real.sum()
        auxiliary = (other.conj() * output).real.sum()
        loss, _ = constrained_loss(original, auxiliary, output, observe_only=observe_only)
        loss.backward()
        expected = primary if observe_only else primary + .35 * projected
        torch.testing.assert_close(output.grad, expected)


def test_unet_coherence_adds_9000_source_steps_and_holds_full_scope():
    plan = trial('unet-coherence')
    assert plan.steps == 9000
    assert plan.source_modes[-2:] == (UNET_BASELINE, OUTER_BASELINE)
    assert set(cases('unet-coherence')) == {f'{FULL}-{i}' for i in range(3)}
    for arm in plan.arms:
        paths = dict(source_root='U', run_root='run', unet_parent='G', source_parent='DURA',
                     classifier_root='F', native_target_root='target')
        assert fit_inputs(paths, arm) == dict(source_root='U', run_root=f'run/{arm}', unet_parent='G')
