import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.residual_diffusion import scratch_residual_diffusion
from experiments.pamf_phase3_restoration.diffusion_source_path import trajectory,summarize,METRICS,parent_ready,parent

torch.set_num_threads(2)


def test_actual_path_final_matches_frozen_deployment_without_state_or_rng_changes():
    g=scratch_residual_diffusion(5).eval().requires_grad_(False)
    with torch.no_grad():g.core.time_out.weight.normal_(0,.001);g.core.freq_out.weight.normal_(0,.001)
    x=torch.randn(3,2,256);before={k:v.clone() for k,v in g.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.inference_mode():
        path=trajectory(g,x)
        assert len(path)==8 and torch.equal(path[-1],g(x))
        assert all(z.shape==x.shape and torch.isfinite(z).all() for z in path)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,g.state_dict()[k]) for k,v in before.items())


def test_complete_source_alignment_and_paired_counts_are_required():
    cases=10240;steps=9
    a=dict(step=np.repeat(np.arange(9),cases),epoch=np.tile(np.repeat(np.arange(1,21),512),steps),
        source_index=np.tile(np.tile(np.arange(512),20),steps),scene=np.full(steps*cases,'test_scene'),lock=np.full(steps*cases,'locked'))
    for key in METRICS:a[key]=np.zeros(steps*cases)
    a['clean_decision_agreement'][(a['step']==8)]=1
    q=summarize(a)
    last=next(x for x in q if x['step']==8 and x['dimension']=='overall')
    assert last['paired_against_first']==dict(repairs=cases,damage=0,net=cases)
    a['source_index'][-1]=-1
    with pytest.raises(ValueError):summarize(a)


def test_parent_only_technical_final_stage_completion_allows_spare_gpu(tmp_path):
    p=tmp_path/'runs'/parent.RUN;p.mkdir(parents=True)
    assert parent_ready(tmp_path) is False
    (p/('terminal-'+parent.ARM+'-cross_both.json')).write_text('{"exit_code":0}',encoding='utf-8')
    assert parent_ready(tmp_path) is True
    (p/('terminal-'+parent.ARM+'-cross_rx.json')).write_text('{"exit_code":1}',encoding='utf-8')
    with pytest.raises(RuntimeError):parent_ready(tmp_path)
