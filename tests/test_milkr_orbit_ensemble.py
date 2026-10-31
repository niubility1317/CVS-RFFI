import numpy as np
import pytest
import torch
from tests.test_milkr_teacher_kernel import examples
from tests.test_milkr_phase_orbit import fake_capture
from experiments.pamf_phase2_practical.milkr import orbit_ensemble as model,prior_fusion as parent
from experiments.pamf_phase2_practical import phase_orbit,teacher_features
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS


def features(seed,n,identical=False):
    a=examples(seed,n);b=a if identical else examples(seed+1,n)
    return dict(a,**{'C4_'+k:b[k] for k in model.RF})


def rule():return dict(config=['signature_identity','all',.5,'single'],ridge=.1,ridge_reference_classes=6,prior_weight=.5,fusion_policy=parent.FUSION_POLICY,confidence_rule=model.RULE,orbit_ensemble_weight=.5)


def test_dual_kernel_PSD_and_residual_normal_equations_query_independence():
    x=features(121,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=rule())
    sim=model.similarities(state.features,state.features,state.config[0]);K=model.make_kernel(sim,state.tau,state.config[2],state.config[3])
    assert len(state.tau)==2 and np.linalg.eigvalsh(K).min()>-1e-10 and np.allclose(np.diag(K),1)
    target=np.eye(8)[y]-parent.prior(x['teacher_probs'],8,state.prior_gate)
    A=np.block([[K+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solved=np.linalg.solve(A,np.concatenate((target,np.zeros((1,8)))))
    assert np.allclose(solved[:-1],state.alpha) and np.allclose(solved[-1],state.intercept)
    assert np.allclose(state.score(x),parent.prior(x['teacher_probs'],8,state.prior_gate)+K@state.alpha+state.intercept)
    query=features(123,5);score,distance=state.evaluate(query)
    rows=[state.evaluate({k:v[i:i+1] for k,v in query.items()}) for i in range(5)]
    assert np.allclose(score,np.concatenate([r[0] for r in rows])) and np.allclose(distance,np.concatenate([r[1] for r in rows])) and np.allclose(score.sum(1),1)
    assert state.selection['orbit_ensemble']['extra_feature_bytes']==8*1072*8


def test_identical_orbits_collapse_to_original_kernel_and_distance():
    x=features(125,8,True);query=features(126,4,True);y=np.arange(8);a=model.fit(x,y,list(map(str,y)),fixed=rule())
    b=parent.fit({k:v for k,v in x.items() if not k.startswith('C4_')},y,list(map(str,y)),fixed=rule())
    s,d=a.evaluate(query);t,e=b.evaluate({k:v for k,v in query.items() if not k.startswith('C4_')})
    assert np.allclose(s,t) and np.allclose(d,e)
    with pytest.raises(ValueError,match='equal orbit'):model.fit(x,y,list(map(str,y)),fixed=dict(rule(),orbit_ensemble_weight=.7))


def test_dual_context_OOF_fits_both_scales_and_gate_on_train_only(monkeypatch):
    x=features(127,24);y=np.repeat(np.arange(8),3);calls=[];original=model.context
    def capture(features,labels,config):calls.append(len(labels));return original(features,labels,config)
    monkeypatch.setattr(model,'context',capture);state=model.fit(x,y,list(map(str,range(24))),fixed=rule())
    assert set(calls)=={16,24} and calls.count(24)==1 and len(calls)==len(model.sig.CONFIGS)*3+1
    assert state.selection['physical_OOF'] and state.selection['prior_confidence_gate']['old_rows']==18


def test_eight_F_calls_produce_original_C4_component_and_C8_without_extra_G(monkeypatch):
    calls=[];g_calls=[]
    def capture(reader,x):calls.append(x.clone());return fake_capture(reader,x)
    monkeypatch.setattr(phase_orbit,'capture',capture);x=np.random.default_rng(128).normal(size=(3,2,256)).astype(np.float32)
    def engine(value):g_calls.append(value.clone());return (value+.2,)
    both,head=teacher_features.extract_orbit_ensemble(None,x,'cpu',engine,batch=3)
    assert len(calls)==8 and len(g_calls)==1 and torch.equal(g_calls[0],torch.tensor(x))
    c4,l4=teacher_features.extract(None,x+.2,'cpu');c8,l8=teacher_features.extract_phase8(None,x+.2,'cpu')
    assert np.array_equal(head,l4) and np.array_equal(head,l8)
    assert all(np.array_equal(both['C4_'+k],v) for k,v in c4.items() if k!='teacher_phase_logits')
    assert all(np.array_equal(both[k],v) for k,v in c8.items())


def test_dual_source_encoder_payload_and_reload_do_not_use_held_head_columns(tmp_path):
    rng=np.random.default_rng(129);x={}
    for prefix in ('','C4_'):
        for name,d in zip(BLOCKS[:5],DIMS[:5]):x[prefix+name]=rng.normal(size=(80,2*d))
        x[prefix+'sig_views']=rng.normal(size=(80,672))
    x['teacher_phase_logits']=rng.normal(size=(80,4,6));bridge=teacher_features.OrbitEnsembleBridge.fit(x,x)
    assert sum(a.size for encoder in bridge.encoders for p in encoder.parameters for a in p)==557440
    bridge.save(tmp_path/'encoder.npz');loaded=teacher_features.OrbitEnsembleBridge.load(tmp_path/'encoder.npz')
    changed=dict(x,teacher_phase_logits=x['teacher_phase_logits'].copy());changed['teacher_phase_logits'][:,:,[1,3]]=1e6
    for path in (False,True):
        a=loaded.transform(x,restored=path,old_columns=[0,2,4,5]);b=bridge.transform(changed,restored=path,old_columns=[0,2,4,5])
        assert all(np.array_equal(value,b[key]) for key,value in a.items())
        one=loaded.transform({k:v[2:3] for k,v in x.items()},restored=path,old_columns=[0,2,4,5]);assert all(np.allclose(value[2],one[key][0]) for key,value in a.items())
    with np.load(tmp_path/'encoder.npz') as z:assert not any('label' in k or 'covariance' in k for k in z.files)


def test_dual_orbit_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen
    assert screen.fit_function({'kernel_method':'orbit_ensemble'}) is model.fit
    sentinel=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *a:sentinel)
    assert screen.extract_for_method({'kernel_method':'orbit_ensemble'},None,None,'cpu') is sentinel
    assert teacher_features.bridge_type('orbit_ensemble') is teacher_features.OrbitEnsembleBridge
