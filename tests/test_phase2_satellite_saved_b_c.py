import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical import group2_cil_bridge as cil
from experiments.pamf_phase2_practical.group2_proto_mopc_affine import ProtoMoPCAffine
from experiments.pamf_phase2_practical.satellite_saved_b_c import (
    POINTS, PROFILES, make_plan, native_extractor, orthogonal_anchor, parent_gpu_drained, verify_parent,
)


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(3,2)

    def forward(self,x):
        return {'features':self.layer(x)}


def test_proto_orthogonal_uses_original_protonet_features():
    model=FeatureModel().eval()
    x=torch.randn(12,3)
    y=torch.arange(6).repeat_interleave(2)
    prototypes=torch.randn(6,2)
    affine=ProtoMoPCAffine(model,prototypes)
    saved={'model':model,'old_prototypes':prototypes,
           'anchor':cil.make_anchor(affine,x,y,method='mopc_hr')}
    anchor=orthogonal_anchor(saved,'protonet_cda',x,y)
    assert isinstance(anchor.model,cil.ProtoClassifier)
    assert anchor.prototypes.shape==(6,3)
    assert torch.equal(anchor.model.classifier_weight,anchor.prototypes)
    assert anchor.protocol_metadata['orthogonal_bridge']=='support_prototypes_v1'


def test_parent_must_have_no_queued_or_active_gpu_workers():
    assert not parent_gpu_drained({'pending':1,'active':[]})
    assert not parent_gpu_drained({'pending':0,'active':[{'pid':1}]})
    assert not parent_gpu_drained({'pending':0})
    assert parent_gpu_drained({'pending':0,'active':[],'score_pending':300})


def test_poster_extractor_keeps_source_method_during_registration_loop(monkeypatch):
    from experiments.pamf_phase2_practical import group1, group1_source
    monkeypatch.setattr(group1,'row',lambda profile: {'method':profile})
    monkeypatch.setattr(group1_source,'embedding',lambda model,method,x: method)
    extractor=native_extractor('poster_ft')
    for method in ('lora','orthogonal'):
        assert extractor(None,None)=='poster_ft'


def test_plan_covers_each_parent_once_and_rejects_bad_parent(tmp_path):
    release=tmp_path/'release';release.mkdir()
    jobs=[]
    for profile in PROFILES:
        for i in range(189):
            jobs.append(dict(id=f'{i:04}-hybrid-{profile}',group='hybrid',profile=profile,
                capsule=f'capsules/P01/MIX01/{i}',seed=2026092701,points=POINTS,methods=['mopc_hr'],
                save_B_for_mopc=True,output=str(tmp_path/f'p-{profile}-{i}')))
    parent=dict(entry_module='satellite_mix3',jobs=jobs,run_root=str(tmp_path/'parent'),
                runtimes={'hybrid':str(tmp_path/'runtime.json')})
    config=tmp_path/'parent.json';config.write_text(json.dumps(parent))
    plan=make_plan(config,tmp_path/'new',release)
    assert len(plan['jobs'])==945 and len({j['id'] for j in plan['jobs']})==945
    assert plan['allowed_gpus']==list(range(8)) and plan['max_per_gpu']==6
    jobs[0]['save_B_for_mopc']=False;config.write_text(json.dumps(parent))
    with pytest.raises(ValueError):make_plan(config,tmp_path/'new',release)


def test_recovery_reuses_complete_and_retains_failed_partial(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import satellite_saved_b_c as module
    release=tmp_path/'release';release.mkdir()
    parent_jobs=[]
    for profile in PROFILES:
        for i in range(189):
            parent_jobs.append(dict(id=f'{i:04}-hybrid-{profile}',group='hybrid',profile=profile,
                capsule=f'capsules/P01/MIX01/{i}',seed=2026092701,points=POINTS,methods=['mopc_hr'],
                save_B_for_mopc=True,output=str(tmp_path/f'b-{profile}-{i}')))
    parent={'entry_module':'satellite_mix3','jobs':parent_jobs,'run_root':str(tmp_path/'broot'),
            'runtimes':{'hybrid':str(tmp_path/'rt')}}
    parent_path=tmp_path/'parent.json';parent_path.write_text(json.dumps(parent))
    prior_root=tmp_path/'prior'
    old=make_plan(parent_path,prior_root,release)
    prior_path=tmp_path/'prior.json';prior_path.write_text(json.dumps(old))
    complete=Path(old['jobs'][0]['output']);complete.mkdir(parents=True);(complete/'complete.json').write_text('{}')
    partial=Path(old['jobs'][1]['output']);partial.mkdir(parents=True)
    control=prior_root/'scheduler';control.mkdir()
    state={'status':'FAILED','active':[],'score_active':[],'completed':1,
           'failed':[{'id':old['jobs'][1]['id'],'stage':'prediction'}]}
    (control/'state.json').write_text(json.dumps(state))
    monkeypatch.setattr(module,'verify_prediction',lambda job: None)
    recovered=make_plan(parent_path,tmp_path/'new',release,prior_path)
    assert recovered['jobs'][0]['prediction_reused'] is True
    assert recovered['jobs'][0]['output']==old['jobs'][0]['output']
    assert recovered['jobs'][1]['output']!=old['jobs'][1]['output']
    (partial/'partial.npz').write_bytes(b'partial')
    with pytest.raises(ValueError):make_plan(parent_path,tmp_path/'new',release,prior_path)


def test_parent_cannot_be_reused_before_all_b_states_exist(tmp_path):
    job=dict(profile='poster_ft',seed=2026092701,capsule='capsules/P01/MIX01/1-1',
             parent={'output':str(tmp_path)})
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',
        truth_read=False,query_fit=False,profile='poster_ft',model_seed=2026092701,rows=4)))
    (tmp_path/'predictions.json').write_text(json.dumps(dict(capsule=job['capsule'],model_seed=job['seed'],rows=[])))
    with pytest.raises(ValueError):verify_parent(job)
