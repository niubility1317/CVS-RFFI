import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_unrolled_action_fidelity import design as d,models as m,runtime,evaluation as e
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_unrolled_restore.models import UnrolledRestorer

torch.set_num_threads(2)


def action(gain,batch=2):
    return Action(torch.ones(batch,1,dtype=torch.complex64),torch.full((batch,256),gain,dtype=torch.complex64),torch.zeros(batch,256,dtype=torch.complex64))


class Frozen(nn.Module):
    def estimate(self,r):return .7*r,action(2.,len(r)),torch.full((len(r),),.02)


class Prior(nn.Module):
    def __init__(self,offset):super().__init__();self.offset=offset
    def forward(self,x):return .1*x[:,4:6]+.02*x[:,6:8]+self.offset+.01*x[:,8:9]


def model():
    return SimpleNamespace(candidate='unroll_unshared',frozen=Frozen(),prior=nn.ModuleList([Prior(x) for x in (.1,.2,.3)]))


def exact(r,gain,variance):
    base=.7*r;current=base
    for offset in (.1,.2,.3):
        innovation=gain*(r-gain*current)
        innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
        prior=base+SCALE*(.1*current+.02*innovation+offset+.01*np.clip(np.log(variance),-16,4)/8)
        current=(gain*r/variance+prior/SCALE**2)/(gain**2/variance+1/SCALE**2)
    return current,prior


def test_all_three_stage_substitutions_match_independent_closed_form():
    torch.manual_seed(8);r=torch.randn(2,2,256)*.2
    values=m.counterfactuals(model(),r,action(1.5),torch.full((2,),.1))
    expected=dict(native_estimated=exact(r,2.,.02)[0],known_action_estvar=exact(r,1.5,.02)[0],
        est_action_simvar=exact(r,2.,.1)[0],known_action_simvar=exact(r,1.5,.1)[0],prior_only=exact(r,2.,.02)[1])
    assert tuple(values)==d.ROWS
    for key,value in values.items():torch.testing.assert_close(value,expected[key],atol=1e-6,rtol=2e-5)
    assert not torch.allclose(values['native_estimated'],values['known_action_estvar'])
    assert not torch.allclose(values['prior_only'],values['native_estimated'])


def test_native_control_matches_original_unrolled_model_without_state_update():
    torch.manual_seed(6);parent=UnrolledRestorer('unroll_unshared',Frozen()).eval().requires_grad_(False)
    for core in parent.prior:nn.init.normal_(core.out.weight,std=.001)
    before={key:value.clone() for key,value in parent.state_dict().items()}
    r=torch.randn(2,2,256)*.2;_,known,var=parent.frozen.estimate(r)
    values=m.counterfactuals(parent,r,known,var)
    torch.testing.assert_close(values['native_estimated'],parent(r),atol=2e-6,rtol=2e-5)
    for name in d.ROWS[1:-1]:torch.testing.assert_close(values[name],values['native_estimated'])
    assert all(torch.equal(before[key],value) for key,value in parent.state_dict().items())
    assert all(not value.requires_grad for value in values.values())


@pytest.mark.parametrize('variance',[torch.zeros(2),torch.ones(2,256),torch.tensor([float('nan'),.1])])
def test_invalid_noise_proxy_rejected(variance):
    with pytest.raises(ValueError):m.counterfactuals(model(),torch.zeros(2,2,256),action(1.),variance)


def test_wrong_model_or_nonfinite_IQ_rejected():
    wrong=model();wrong.candidate='unroll_shared'
    with pytest.raises(ValueError):m.counterfactuals(wrong,torch.zeros(2,2,256),action(1.),torch.ones(2))
    with pytest.raises(ValueError):m.counterfactuals(model(),torch.full((2,2,256),float('nan')),action(1.),torch.ones(2))


def complete(n):
    return dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source',count=n,views=list(e.VIEWS),readouts=list(d.ROWS),
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,target_reads=0,optimizer_updates=0,
        F_run=e.REFERENCE_RUN,F_seed=2026092701,original_default_head=True,oracle_not_deployable=True)


@pytest.mark.parametrize('key,bad',[('role','target'),('views',['clean']),('readouts',list(d.ROWS)[:-1]),('state_updates',1),('head_fit_updates',1),('target_reads',1),('original_default_head',False)])
def test_complete_contract_rejects_illegal_or_partial_outputs(key,bad):
    value=complete(2);e.complete_contract(value,2);value[key]=bad
    with pytest.raises(ValueError):e.complete_contract(value,2)


def test_score_does_not_open_truth_if_registered_arrays_are_missing(tmp_path,monkeypatch):
    prediction=tmp_path/'prediction';prediction.mkdir();ids=np.arange(2)
    (prediction/'complete.json').write_text(json.dumps(complete(2)));np.save(prediction/'physical_ids.npy',ids)
    opened=[];original=np.load
    def load(path,*args,**kwargs):
        if str(path).endswith('labels.npy'):opened.append(path);pytest.fail('Truth opened before finite prediction coverage')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',load);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'source_inputs',lambda a:(None,ids,None,None))
    with pytest.raises(FileNotFoundError):e.score(SimpleNamespace(output=tmp_path))
    assert not opened


def test_CPU_scoring_required_before_source_IO(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(e,'source_inputs',lambda a:pytest.fail('Forbidden IO'))
    with pytest.raises(ValueError):e.score(None)


def test_target_or_wrong_reference_rejected_before_source_cache_IO(tmp_path):
    for key in ('parent_run','reference_run'):
        args=SimpleNamespace(parent_run=tmp_path/d.PARENT,parent_action=tmp_path/d.ACTION,reference_run=tmp_path/d.REFERENCE)
        setattr(args,key,tmp_path/'target')
        with pytest.raises(ValueError):runtime.source_inputs(args)


def source_summary():
    value=dict(status='ANALYZED',role='source',count=1800,target_access=False,all_predictions_before_truth=True,
        F_run=e.REFERENCE_RUN,F_seed=2026092701,original_default_head=True,optimizer_updates=0,head_fit_updates=0,
        source_V_state_updates=0,oracle_not_deployable=True,theoretical_upper_bound=False)
    value['views']={view:{name:dict(waveform=dict(iq_nmse=.094 if name=='known_action_simvar' else .1)) for name in (*d.REFERENCES,*d.ROWS)} for view in e.VIEWS}
    value['records']=[]
    for view in e.VIEWS:
        for name in (*d.REFERENCES,*d.ROWS):
            accuracy=98.5 if view=='clean' else 82. if name=='known_action_simvar' else 80.
            value['records'].append(dict(view=view,readout=name,dimension='overall',value='all',accuracy=accuracy))
            value['records'] += [dict(view=view,readout=name,dimension='TX',value=str(i),accuracy=accuracy) for i in range(6)]
    return value


def test_source_evidence_cannot_promote_oracle_or_override_TX_guard():
    source=source_summary();result=e.interpret(source)
    assert result['supports_operator_priority'] and not result['target_promotion'] and not result['theoretical_upper_bound']
    bad=copy.deepcopy(source)
    row=next(r for r in bad['records'] if r['view']=='practical_low_urban' and r['readout']=='known_action_simvar' and r['dimension']=='TX')
    row['accuracy']=78.;assert not e.interpret(bad)['supports_operator_priority']
    bad=dict(source,target_access=True)
    with pytest.raises(ValueError):e.interpret(bad)


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
