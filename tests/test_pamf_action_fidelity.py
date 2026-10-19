import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_action_fidelity import design as d
from experiments.pamf_action_fidelity.models import counterfactuals
from experiments.pamf_action_fidelity import evaluation as e
from experiments.pamf_action_fidelity import runtime
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_action_restore.models import SCALE


def action(gain,batch=2):
    main=torch.full((batch,256),gain,dtype=torch.complex64)
    return Action(torch.ones((batch,1),dtype=torch.complex64),main,torch.zeros_like(main))


def fake_model():
    def estimate(r):return .7*r,action(2.,len(r)),torch.full((len(r),),.02)
    return SimpleNamespace(frozen=SimpleNamespace(estimate=estimate),prior=lambda x:torch.full_like(x[:,:2],.25))


def test_counterfactuals_equal_closed_form_and_keep_prior_fixed():
    r=torch.randn(2,2,256)*.2;known=action(1.5);var=torch.full((2,),.1)
    out=counterfactuals(fake_model(),r,known,var);prior=.7*r+.25*SCALE
    def exact(gain,noise,p):return (gain*r/noise+p/SCALE**2)/(gain**2/noise+1/SCALE**2)
    expected=dict(prior_only=prior,known_action_estvar=exact(1.5,.02,prior),est_action_simvar=exact(2.,.1,prior),
        known_action_simvar=exact(1.5,.1,prior),coarse_known_action_simvar=exact(1.5,.1,.7*r))
    assert tuple(out)==d.ROWS
    for key in out:torch.testing.assert_close(out[key],expected[key],atol=1e-6,rtol=1e-5)
    assert not torch.allclose(out['prior_only'],out['known_action_simvar'])


@pytest.mark.parametrize('variance',[torch.zeros(2),torch.ones(2,256),torch.tensor([float('nan'),.1])])
def test_invalid_variance_rejected(variance):
    with pytest.raises(ValueError):counterfactuals(fake_model(),torch.zeros(2,2,256),action(1.),variance)


def complete(count):
    return dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source',count=count,views=list(e.VIEWS),readouts=list(d.ROWS),TX_truth_reads=0,state_updates=0,
        head_fit_updates=0,target_reads=0,optimizer_updates=0,F_run=e.REFERENCE_RUN,F_seed=2026092701,original_default_head=True,oracle_not_deployable=True)


def test_prediction_contract_rejects_target_incomplete_and_state_updates():
    value=complete(2);e.complete_contract(value,2)
    for key,bad in [('role','target'),('views',list(e.VIEWS)[:-1]),('readouts',list(d.ROWS)[:-1]),('state_updates',1),('head_fit_updates',1),('target_reads',1),('original_default_head',False)]:
        changed=dict(value);changed[key]=bad
        with pytest.raises(ValueError):e.complete_contract(changed,2)


def test_missing_prediction_blocks_truth_open(tmp_path,monkeypatch):
    (tmp_path/'complete.json').write_text(json.dumps(dict(complete(2),views=['clean'])))
    called=[];monkeypatch.setattr(np,'load',lambda *a,**k:called.append(a))
    with pytest.raises(ValueError):e.load_predictions(tmp_path,np.arange(2))
    assert not called


def test_source_cache_cannot_be_replaced_with_target(tmp_path):
    args=SimpleNamespace(parent_run=tmp_path/'target',parent_action=tmp_path/d.ACTION)
    with pytest.raises(ValueError):runtime.source_inputs(args)


def summary():
    value=dict(status='ANALYZED',role='source',count=1800,target_access=False,all_predictions_before_truth=True,F_run=e.REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,optimizer_updates=0,head_fit_updates=0,source_V_state_updates=0,oracle_not_deployable=True,theoretical_upper_bound=False)
    value['views']={view:{name:dict(waveform=dict(iq_nmse=.094 if name=='known_action_simvar' else .1)) for name in (*d.REFERENCES,*d.ROWS)} for view in e.VIEWS}
    records=[]
    for view in e.VIEWS:
        for name in (*d.REFERENCES,*d.ROWS):
            accuracy=98.5 if view=='clean' else (82. if name=='known_action_simvar' else 80.)
            records.append(dict(view=view,readout=name,dimension='overall',value='all',accuracy=accuracy))
            records += [dict(view=view,readout=name,dimension='TX',value=str(i),accuracy=accuracy) for i in range(6)]
    value['records']=records;return value


def test_source_interpretation_never_promotes_oracle_and_enforces_TX_guard():
    good=summary();decision=e.interpret(good)
    assert decision['supports_operator_priority'] and not decision['target_promotion'] and not decision['theoretical_upper_bound']
    bad=copy.deepcopy(good)
    row=next(x for x in bad['records'] if x['view']=='practical_low_urban' and x['readout']=='known_action_simvar' and x['dimension']=='TX')
    row['accuracy']=78.
    assert not e.interpret(bad)['supports_operator_priority']
    for key,value in [('target_access',True),('original_default_head',False),('source_V_state_updates',1)]:
        changed=dict(good);changed[key]=value
        with pytest.raises(ValueError):e.interpret(changed)


def test_score_requires_independent_CPU_before_any_source_IO(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(e,'source_inputs',lambda a:pytest.fail('Forbidden source IO'))
    with pytest.raises(ValueError):e.score(None)


def test_predict_constructs_versioned_F_parameters_and_completes_all_readouts(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();action_cache=tmp_path/'action';output=tmp_path/'output';output.mkdir()
    for view in e.VIEWS:
        np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
        folder=action_cache/'cache/validation_actions'/view;folder.mkdir(parents=True)
        (folder/'complete.json').write_text(json.dumps(dict(count=16,namespace='restoration/validation/draw0',TX_labels_read=0,F_outputs_read=0,target_reads=0,native_engine_modified=False)))
        for key,shape in (('kernel',(16,2,64)),('coefficients',(16,4,256)),('variance',(16,))):np.save(folder/(key+'.npy'),np.ones(shape,np.float32))
    class VersionedReader(torch.nn.Module):
        def __init__(self):super().__init__();self.coordinate=torch.nn.Parameter(torch.ones(1))
        def features(self,x):
            # Mirrors the original F Sinc cache's required parameter-version read.
            assert self.coordinate._version>=0
            return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    monkeypatch.setattr(e,'source_inputs',lambda a:(source,np.arange(16),None,None))
    monkeypatch.setattr(e,'load_model',lambda *a:(None,None))
    monkeypatch.setattr(e,'load_reader',lambda *a:VersionedReader())
    monkeypatch.setattr(e,'teacher_action',lambda *a:None)
    monkeypatch.setattr(e,'counterfactuals',lambda model,r,known,var:{name:r for name in d.ROWS})
    monkeypatch.setattr(e,'array_tensor',lambda value,device:torch.tensor(value.tolist(),dtype=torch.float32))
    monkeypatch.setattr(e,'tensor_array',lambda value:np.asarray(value.tolist(),np.float32))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    args=SimpleNamespace(parent_action=action_cache,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    e.predict(args)
    values=e.load_predictions(output/'prediction',np.arange(16))
    assert len(values)==7 and all(tuple(values[view])==d.ROWS for view in e.VIEWS)
