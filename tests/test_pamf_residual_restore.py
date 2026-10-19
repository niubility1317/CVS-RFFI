import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.pamf_residual_restore.models import (NAMES,build,ResidualDeterministic,ResidualDiffusion,ResidualFlow)
from experiments.pamf_residual_restore import design as d
from experiments.pamf_residual_restore import runtime as rt
from experiments.pamf_residual_restore.controller import waveform_choice,source_selection
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_reception_restore.data import shared_pair
from experiments.pamf_restoration.models import ConditionalDiffusion,ConditionalFlow
from experiments.pamf_restoration.runner import predict_batch

torch.set_num_threads(2)


def test_frozen_coarse_parameters_and_normalization_state():
    coarse=nn.BatchNorm1d(2)
    model=ResidualDeterministic(coarse,.2).train()
    assert not model.coarse.training and not any(p.requires_grad for p in coarse.parameters())
    before={key:value.clone() for key,value in coarse.state_dict().items()}
    received=torch.randn(3,2,256)*4
    with torch.no_grad():assert torch.equal(model(received),coarse(received))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    loss=(model(received)-(received+.1)).square().mean();loss.backward();optimizer.step()
    assert all(p.grad is None for p in coarse.parameters())
    assert all(torch.equal(before[key],value) for key,value in coarse.state_dict().items())
    assert model(received).abs().max()>1


def test_diffusion_residual_training_uses_fixed_scale_and_coarse_offset():
    received=torch.randn(2,2,256);clean=received+.2*torch.randn_like(received)
    model=ResidualDiffusion(nn.Identity(),.2)
    seed=129;expected=torch.Generator().manual_seed(seed)
    time=torch.randint(model.steps,(2,),generator=expected)
    noise=torch.randn(clean.shape,generator=expected)
    a=model.alpha_bar[time,None,None].sqrt();b=(1-model.alpha_bar[time,None,None]).sqrt()
    residual=(clean-received)/model.residual_scale
    target=a*noise-b*residual
    model.field=lambda point,condition,t:target
    estimate,loss,diagnostic=model.training_prediction(clean,received,torch.Generator().manual_seed(seed),True)
    assert torch.allclose(estimate,clean,atol=2e-6) and loss==0
    assert diagnostic['time_fraction'].shape==(2,)


def test_flow_training_and_inference_start_in_normalized_residual_space():
    received=torch.randn(2,2,256);clean=received+.15*torch.randn_like(received)
    model=ResidualFlow(nn.Identity(),.15)
    seed=130;expected=torch.Generator().manual_seed(seed)
    noise=torch.randn(clean.shape,generator=expected);time=torch.rand((2,),generator=expected)
    start=.02*noise;residual=(clean-received)/model.residual_scale
    def oracle(point,condition,t):
        assert torch.allclose(point,(1-time[:,None,None])*start+time[:,None,None]*residual)
        assert torch.allclose(t,time*model.network.steps)
        return residual-start
    model.network.field=oracle
    estimate,loss,_=model.training_prediction(clean,received,torch.Generator().manual_seed(seed),True)
    assert torch.allclose(estimate,clean,atol=2e-6) and loss==0
    model.network.field=lambda point,condition,t:torch.zeros_like(point)
    assert torch.allclose(model(received,noise),received+model.residual_scale*.02*noise)


@pytest.mark.parametrize('name',NAMES[1:])
def test_complete_sampling_uses_current_observation_once_without_state_updates(name):
    model=build(name,nn.Identity(),.1).eval()
    assert isinstance(model,(ConditionalDiffusion,ConditionalFlow))
    encoder=model.conditioner if name=='residual_diffusion' else model.network.conditioner
    calls=[];handle=encoder.register_forward_hook(lambda module,args,result:calls.append(args[0].shape))
    received=torch.randn(2,2,256);noise=torch.randn_like(received)
    before={key:value.clone() for key,value in model.state_dict().items()}
    with torch.no_grad():out=model(received,noise)
    handle.remove()
    assert calls==[torch.Size([2,4,256])] and out.shape==received.shape and torch.isfinite(out).all()
    assert all(torch.equal(before[key],value) for key,value in model.state_dict().items())
    with pytest.raises(ValueError):model(received,torch.zeros(2,4,256))
    with torch.no_grad():assert torch.isfinite(predict_batch(model,received,['p0','p1'])).all()


def test_waveform_selector_rejects_labels_nonfinite_and_missing_controls():
    rows=[dict(candidate=name,clean_nmse=.001,mean_residual_nmse=.1) for name in ('coarse',*NAMES)]
    assert waveform_choice(rows)=='coarse'
    polluted=copy.deepcopy(rows);polluted[1]['TX_accuracy']=99.
    with pytest.raises(ValueError):waveform_choice(polluted)
    bad=copy.deepcopy(rows);bad[1]['mean_residual_nmse']=float('nan')
    with pytest.raises(ValueError):waveform_choice(bad)
    with pytest.raises(ValueError):waveform_choice(rows[1:])


def parent_files(parent):
    parent.mkdir();folder=parent/d.COARSE_NAME;folder.mkdir()
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    write(parent/'launch.json',dict(commit=d.PARENT_COMMIT))
    write(parent/'source_decision.json',dict(status='SOURCE_SELECTION_FROZEN',waveform_selected=d.COARSE_NAME,
        waveform_selection_TX_F_access=False,training_TX_labels_read=0,no_target_feedback=True,
        identity_diagnostic=[dict(candidate=d.COARSE_NAME,qualified=True)]))
    write(folder/'training_complete.json',dict(epochs=20,updates=4940,TX_labels_read=0,L_exposures=126000,U_exposures=1134000))
    return folder


def metric(clean_accuracy,low_accuracy,mean,switch=None):
    views={scene:dict(summary=dict(restored=dict(iq_nmse=mean)),classification=dict(AsF=dict(accuracy=low_accuracy)))
           for scene in rt.SCENES}
    if switch is not None:views['practical_low_urban']['classification']['AsF']['stochastic_diagnostic']=dict(count=128,draws=3,any_class_switch_percent=switch)
    views['clean']=dict(summary=dict(restored=dict(iq_nmse=.001)),classification=dict(AsF=dict(accuracy=clean_accuracy)))
    return dict(views=views)


def test_source_selection_relative_to_coarse_and_stochastic_qualification(tmp_path,monkeypatch):
    parent=tmp_path/d.PARENT_RUN;coarse=parent_files(parent)
    (coarse/'metrics.json').write_text(json.dumps(metric(98.,80.,.1)),encoding='utf-8')
    output=tmp_path/'new';output.mkdir()
    for name,acc,switch in zip(NAMES,(80.4,81.,81.),(None,50.,6.25)):
        row=output/name;row.mkdir()
        (row/'metrics.json').write_text(json.dumps(metric(97.9,acc,.09 if name!=NAMES[2] else .08,switch)),encoding='utf-8')
    result=source_selection(output,parent)
    assert result['waveform_selected']=='residual_flow' and result['identity_assisted_selected'] is None
    assert result['target_candidates']==['residual_flow'] and not result['source_qualified']
    with pytest.raises(FileExistsError):source_selection(output,parent)
    alternative=tmp_path/'alternative';alternative.mkdir()
    for name in NAMES:
        (alternative/name).mkdir()
        item=metric(97.9,81.,.08,0.) if name==NAMES[2] else json.loads((output/name/'metrics.json').read_text())
        (alternative/name/'metrics.json').write_text(json.dumps(item),encoding='utf-8')
    result=source_selection(alternative,parent)
    assert result['source_qualified'] and result['target_candidates']==['residual_flow']
    assert result['waveform_selection_TX_F_access'] is False


def test_parent_provenance_wrong_commit_and_target_selection_are_rejected(tmp_path):
    parent=tmp_path/d.PARENT_RUN;parent_files(parent)
    assert rt.parent_metadata(parent)['selector']=='source_waveform_A'
    (parent/'launch.json').write_text(json.dumps(dict(commit='other')),encoding='utf-8')
    with pytest.raises(ValueError):rt.parent_metadata(parent)
    (parent/'launch.json').write_text(json.dumps(dict(commit=d.PARENT_COMMIT)),encoding='utf-8')
    p=parent/'source_decision.json';value=json.loads(p.read_text());value['waveform_selection_TX_F_access']=True
    p.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError):rt.parent_metadata(parent)


def test_checkpoint_portable_includes_frozen_parent_and_rejects_scale_or_route_swap(tmp_path):
    model=ResidualDeterministic(ComplexTFUNet(),.13).eval()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    parent=dict(run=d.PARENT_RUN,commit=d.PARENT_COMMIT,candidate=d.COARSE_NAME,selector='source_waveform_A',
                training_TX_labels_read=0,target_training=False)
    path=tmp_path/'final.pt';rt.save_checkpoint(path,model,optimizer,20,4940,NAMES[0],parent)
    fresh,payload=rt.load_checkpoint(path,NAMES[0],'cpu')
    received=torch.randn(2,2,256)
    with torch.no_grad():assert torch.allclose(model(received),fresh(received))
    with pytest.raises(ValueError):rt.load_checkpoint(path,NAMES[1],'cpu')
    payload['scale']=.2;torch.save(payload,path)
    with pytest.raises(ValueError):rt.load_checkpoint(path,NAMES[0],'cpu')


def test_new_namespace_native_observations_differ_without_changing_physical_roles():
    clean=np.random.default_rng(12).normal(size=(2,256)).astype(np.float32)
    old=shared_pair(clean,'practical_low_urban','source0','rx1:day1:eq1',2026101901,1,2026101902)
    new=shared_pair(clean,'practical_low_urban','source0','rx1:day1:eq1',2026101901,1,2026101902,d.NAMESPACE+'/epoch01')
    replay=shared_pair(clean,'practical_low_urban','source0','rx1:day1:eq1',2026101901,1,2026101902,d.NAMESPACE+'/epoch01')
    assert old[2]['seed']!=new[2]['seed'] and not np.array_equal(old[0],new[0])
    assert np.array_equal(new[0],replay[0]) and np.array_equal(new[1],replay[1])


def test_scale_indices_use_only_fixed_source_training_records():
    selected=rt.calibration_indices(63000)
    assert len(selected)==8192 and len(set(selected.tolist()))==8192
    assert selected.min()>=0 and selected.max()<63000
    assert np.array_equal(selected,rt.calibration_indices(63000))
