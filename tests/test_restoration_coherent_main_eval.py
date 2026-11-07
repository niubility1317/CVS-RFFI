import numpy as np
import pytest
import torch

from experiments.pamf_phase3_restoration.coherent_main_eval import FrozenCoherentMain, collect_task_controls
from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance
from experiments.pamf_phase3_restoration.phase_ramp_posterior import fixed_grid
from experiments.pamf_phase3_restoration.source_coherent_main_phase import readouts


def test_cpu_formal_candidate_matches_source_full_covariance_path():
    G = scratch_covariance(977).eval()
    x = torch.linspace(0,1,256)
    mean = torch.cat((.4+.1*x,.2+.1*x.square()))
    covariance = .1*torch.eye(512) + .001*torch.outer(mean,mean)
    G.bind_prior(mean,covariance)
    phase,drift,prior = fixed_grid('cpu')
    view = torch.stack((.8*x,.4*x+.1)).repeat(2,1,1)
    view[1] = view[1].roll(1,dims=0)
    candidate = FrozenCoherentMain(G).eval().requires_grad_(False)
    with torch.inference_mode():
        expected,_ = readouts(G,view,(phase,drift[4:5],prior[4:5]))
        actual = candidate(view)
    torch.testing.assert_close(actual,expected['main_mean_full']['iq'],rtol=0,atol=0)


def test_task_controls_reject_incomplete_main_before_loading_parent(tmp_path,monkeypatch):
    from experiments.pamf_phase3_restoration import coherent_main_eval as entry

    monkeypatch.setattr(entry,'frozen_controls',
        lambda *args,**kwargs:pytest.fail('Parent predictions must not open for incomplete Main'))
    with pytest.raises(ValueError,match='Full original Main population'):
        collect_task_controls(tmp_path,'cross_rx',np.array(['a']),{})
