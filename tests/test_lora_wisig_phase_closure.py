import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.phase_closure import closure_profile, phase_closure_loss
from experiments.pamf_lora_main.shared_gain_dura_fit import CLOSURE_STEPS, CLOSURE_WEIGHT, fitting_plan


@pytest.mark.parametrize('length',(256,8192))
def test_affine_carrier_and_gain_leave_local_closure_unchanged(length):
    torch.manual_seed(12)
    wave = torch.randn(2,length,dtype=torch.complex64)
    time = torch.arange(length) / length
    transformed = 2.7*wave * torch.exp(1j*(.27 + 13.4*time))
    torch.testing.assert_close(closure_profile(transformed),closure_profile(wave),atol=1e-6,rtol=1e-5)
    assert phase_closure_loss(transformed,wave) < 1e-10


def test_local_complex_distortion_is_penalized_and_gradient_is_finite():
    torch.manual_seed(6)
    wave = torch.randn(1,256,dtype=torch.complex64)
    changed = wave.clone()
    changed[:,64:96] *= torch.exp(1j*.22*torch.arange(32).square())
    changed[:,32:64] = 0.
    changed.requires_grad_()
    loss = phase_closure_loss(changed,wave)
    assert loss > 1e-4
    gradient, = torch.autograd.grad(loss,changed)
    assert torch.isfinite(gradient).all()
    assert gradient[:,64:96].abs().mean() > 0.


def test_closure_run_retains_the_same_architecture_and_prior_panel(tmp_path):
    prior = tmp_path/screen.PREAMBLE_NOISE_RUN
    panel = prior/'lora'
    panel.mkdir(parents=True)
    screen.write(panel/'complete.json',screen.panel_receipt('lora',160,screen.PREAMBLE_NOISE_MODES))
    np.savez(panel/'predictions.npz',ids=np.arange(160))
    rt = {'phase_closure':True,'preamble_noise':True,'observed_anchor':True,
          'closure_panel_from':str(prior),'run_root':str(tmp_path/screen.CLOSURE_RUN)}
    ids = screen.panel_rows(rt,'lora',np.arange(200),np.ones(200,dtype=np.int8))
    np.testing.assert_array_equal(ids,np.arange(160))
    seed, modes, options = fitting_plan(rt)
    assert (seed,modes)==(screen.CLOSURE_SEED,screen.CLOSURE_MODES[3:])
    assert options[0]==options[1]
    assert CLOSURE_STEPS==7200 and CLOSURE_WEIGHT==.25


def test_score_counts_recoveries_among_all_control_misses():
    modes = screen.CLOSURE_MODES
    truth = np.zeros(160,dtype=np.int16)
    decisions = np.ones((2,len(modes),160),dtype=np.int16)
    decisions[1,:3,:3] = 0
    decisions[1,3,:7] = 0
    decisions[1,4,:5] = 0
    scores = screen.score_stage_panel(decisions,truth,modes=modes,seed=screen.CLOSURE_SEED)
    scene = 'practical_low_urban'
    assert scores['all_controls_wrong'][scene]==157
    assert scores['corrected_when_all_controls_wrong'][modes[3]][scene]==4
    assert scores['corrected_when_all_controls_wrong'][modes[4]][scene]==2
