import importlib
import copy
import pytest
import torch
from torch import nn
from types import SimpleNamespace

torch.set_num_threads(2)
def m():return importlib.import_module('experiments.pamf_noise_covariance.models')


def test_block_precision_matches_independent_real_matrix_inverse():
    torch.manual_seed(30);x=torch.randn(3,8,dtype=torch.complex64)
    variance=torch.tensor([.01,.1,.3]);q=torch.full_like(x,.2+.15j)
    result=m().precision(x,variance,q)
    c=variance[:,None,None]*torch.tensor([[1.2,.15],[.15,.8]])[None]
    expected=torch.linalg.solve(c[:,None],torch.view_as_real(x)[...,None])[...,0]
    torch.testing.assert_close(torch.view_as_real(result),expected,atol=2e-5,rtol=1e-6)
    y=torch.randn_like(x)
    torch.testing.assert_close((x.conj()*m().precision(y,variance,q)).real.sum(-1),(m().precision(x,variance,q).conj()*y).real.sum(-1))
    assert ((x.conj()*result).real.sum(-1)>0).all()


def test_pseudo_covariance_matches_fixed_native_widely_linear_noise():
    torch.manual_seed(31);a=torch.tensor(1.1+.2j);b=torch.tensor(.03-.04j);sigma=.02
    w=torch.complex(torch.randn(200000),torch.randn(200000))*(sigma/2)**.5
    z=a*w+b*w.conj();observed=torch.view_as_real(z).T@torch.view_as_real(z)/len(z)
    n=sigma*(a.abs().square()+b.abs().square())/2;q=m().relative_pseudo(a[None,None],b[None,None])[0,0]
    expected=n*torch.stack((torch.stack((1+q.real,q.imag)),torch.stack((q.imag,1-q.real))))
    torch.testing.assert_close(observed,expected,atol=1e-4,rtol=.02)
    eigen=torch.linalg.eigvalsh(expected)
    torch.testing.assert_close(eigen,torch.stack((n*(1-q.abs()),n*(1+q.abs()))))


def act(gain,batch=2,count=256,image=0.):
    from experiments.pamf_action_restore.operators import Action
    return Action(torch.ones(batch,1,dtype=torch.complex64),torch.full((batch,count),gain,dtype=torch.complex64),torch.full((batch,count),image,dtype=torch.complex64))


def test_two_CG_iterations_match_independent_closed_form_with_warm_start():
    torch.manual_seed(30);y=torch.randn(2,8,dtype=torch.complex64);prior=.3*torch.randn_like(y);initial=.7*prior
    a=act(1.2+.3j,count=8,image=.02j);v=torch.tensor([.03,.08]);q=torch.full_like(y,.2+.1j);tau=.04
    out=m().block_map(a,y,prior,v,q,prior_variance=tau,steps=2,initial=initial)
    h=torch.tensor([[1.2,-.28],[.32,1.2]])
    c=v[:,None,None]*torch.tensor([[1.2,.1],[.1,.8]])[None];w=torch.linalg.inv(c)
    normal=h.T@w@h+torch.eye(2)[None]/tau
    rhs=torch.einsum('bij,btj->bti',h.T@w,torch.view_as_real(y))+torch.view_as_real(prior)/tau
    expected=torch.linalg.solve(normal[:,None],rhs[...,None])[...,0]
    torch.testing.assert_close(torch.view_as_real(out),expected,atol=2e-6,rtol=2e-5)


class Frozen(nn.Module):
    def estimate(self,x):return .7*x,act(1.5+.1j,len(x)),torch.full((len(x),),.02)


class Prior(nn.Module):
    def __init__(self,i):super().__init__();self.i=i
    def forward(self,x):return .1*x[:,4:6]+.02*x[:,6:8]+self.i+.01*x[:,8:9]


def toy():return SimpleNamespace(candidate='unroll_unshared',frozen=Frozen(),prior=nn.ModuleList(Prior(i) for i in (.1,.2,.3)))


def test_zero_image_matches_original_isotropic_trajectories():
    from experiments.pamf_unrolled_action_fidelity.models import trajectory
    torch.manual_seed(32);x=torch.randn(2,2,256)*.2;model=toy();known=act(1.2+.1j);variance=torch.full((2,),.1)
    outputs=m().counterfactuals(model,x,known,variance)
    base,estimated,_=model.frozen.estimate(x)
    for name,action in [('estimated_covariance',estimated),('known_covariance',known)]:
        expected=trajectory(model,x,base,action,variance)[0]
        torch.testing.assert_close(outputs[name],expected,atol=2e-6,rtol=2e-5)
        assert not outputs[name].requires_grad


@pytest.mark.parametrize('variance,q',[ (torch.zeros(2),torch.zeros(2,8,dtype=torch.complex64)),
 (torch.ones(2),torch.ones(2,8,dtype=torch.complex64)),
 (torch.ones(2,8),torch.zeros(2,8,dtype=torch.complex64)),
 (torch.ones(2),torch.full((2,8),complex(float('nan'),0))) ])
def test_invalid_or_singular_covariance_rejected(variance,q):
    with pytest.raises(ValueError):m().precision(torch.ones(2,8,dtype=torch.complex64),variance,q)


def test_model_scope_and_finite_input_required():
    wrong=toy();wrong.candidate='unroll_shared'
    with pytest.raises(ValueError):m().counterfactuals(wrong,torch.zeros(2,2,256),act(1.),torch.ones(2))
    with pytest.raises(ValueError):m().counterfactuals(toy(),torch.full((2,2,256),float('nan')),act(1.),torch.ones(2))


def test_actual_three_stage_state_is_unchanged():
    from experiments.pamf_unrolled_restore.models import UnrolledRestorer
    model=UnrolledRestorer('unroll_unshared',Frozen()).eval().requires_grad_(False);old=copy.deepcopy(model.state_dict())
    out=m().counterfactuals(model,torch.randn(2,2,256)*.2,act(1.,image=.02+.01j),torch.full((2,),.02))
    assert all(value.shape==(2,2,256) and torch.isfinite(value).all() for value in out.values())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.items())


import json
import numpy as np
from experiments.pamf_noise_covariance import design as d,runtime,evaluation as e
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
    for key in ('parent_run','reference_run','covariance_run'):
        args=SimpleNamespace(parent_run=tmp_path/d.PARENT,parent_action=tmp_path/d.ACTION,reference_run=tmp_path/d.REFERENCE,covariance_run=tmp_path/d.COV_REFERENCE)
        setattr(args,key,tmp_path/'target')
        with pytest.raises(ValueError):runtime.source_inputs(args)


def source_summary():
    value=dict(status='ANALYZED',role='source',count=1800,target_access=False,all_predictions_before_truth=True,
        F_run=e.REFERENCE_RUN,F_seed=2026092701,original_default_head=True,optimizer_updates=0,head_fit_updates=0,
        source_V_state_updates=0,oracle_not_deployable=True,theoretical_upper_bound=False)
    value['views']={view:{name:dict(waveform=dict(iq_nmse=.094 if name=='known_covariance' else .1)) for name in (*d.REFERENCES,*d.ROWS)} for view in e.VIEWS}
    value['records']=[]
    for view in e.VIEWS:
        for name in (*d.REFERENCES,*d.ROWS):
            accuracy=98.5 if view=='clean' else 82. if name=='known_covariance' else 80.
            value['records'].append(dict(view=view,readout=name,dimension='overall',value='all',accuracy=accuracy))
            value['records'] += [dict(view=view,readout=name,dimension='TX',value=str(i),accuracy=accuracy) for i in range(6)]
    return value


def test_source_evidence_cannot_promote_oracle_or_override_TX_guard():
    source=source_summary();result=e.interpret(source)
    assert result['supports_covariance_priority'] and not result['target_promotion'] and not result['theoretical_upper_bound']
    bad=copy.deepcopy(source)
    row=next(r for r in bad['records'] if r['view']=='practical_low_urban' and r['readout']=='known_covariance' and r['dimension']=='TX')
    row['accuracy']=78.;assert not e.interpret(bad)['supports_covariance_priority']
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
    monkeypatch.setattr(e,'teacher_action',lambda arrays,index,device:act(1.,batch=len(index)))
    monkeypatch.setattr(e,'counterfactuals',lambda model,r,known,var:{name:r for name in d.ROWS})
    monkeypatch.setattr(e,'array_tensor',lambda value,device:torch.tensor(value.tolist(),dtype=torch.float32))
    monkeypatch.setattr(e,'tensor_array',lambda value:np.asarray(value.tolist(),np.float32))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    args=SimpleNamespace(parent_action=action_cache,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    e.predict(args)
    values=e.load_predictions(output/'prediction',np.arange(16))
    assert len(values)==7 and all(tuple(values[view])==d.ROWS for view in e.VIEWS)
