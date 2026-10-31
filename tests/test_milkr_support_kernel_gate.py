import numpy as np
import pytest
from tests.test_milkr_orbit_ensemble import features,rule
from experiments.pamf_phase2_practical.milkr import support_kernel_gate as gate,orbit_ensemble as model,prior_fusion as prior


def test_balanced_kernel_normal_equations_and_fixed_neutral_factor():
    rng=np.random.default_rng(510);x=rng.normal(size=(12,5));k=x@x.T;labels=np.arange(12);state=gate.fit(k,labels,6)
    weights=np.full(12,1/12);target=(labels<6).astype(float)
    a=k+state.ridge*np.diag(1/weights)
    assert np.allclose(a@state.alpha+state.intercept,target) and abs(state.alpha.sum())<1e-10
    assert np.allclose(state.factors(k),np.minimum(2*np.clip(k@state.alpha+state.intercept,0,1),1))
    constant=gate.fit(np.ones((12,12)),labels,6);assert constant.ridge==0 and np.array_equal(constant.factors(np.ones((3,12))),np.ones(3))
    old=gate.fit(k[:6,:6],labels[:6],6);assert old.new_rows==0 and np.array_equal(old.factors(k[:3,:6]),np.ones(3))
    assert not state.alpha.flags.writeable and state.state_bytes==12*8+16


def test_rf_evidence_prior_residual_normal_equations_query_independence():
    x=features(511,8);y=np.arange(8);fixed=dict(rule(),rf_prior_gate_rule=gate.RULE)
    state=model.fit(x,y,list(map(str,y)),fixed=fixed)
    sim=model.similarities(state.features,state.features,state.config[0]);k=model.make_kernel(sim,state.tau,state.config[2],state.config[3])
    bias=prior.prior(x['teacher_probs'],8,state.prior_gate)*state.rf_gate.factors(k)[:,None]
    system=np.block([[k+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solved=np.linalg.solve(system,np.concatenate((np.eye(8)-bias,np.zeros((1,8)))))
    assert np.allclose(state.alpha,solved[:-1]) and np.allclose(state.intercept,solved[-1])
    query=features(513,4);score=state.score(query)
    assert np.allclose(score,np.concatenate([state.score({key:value[i:i+1] for key,value in query.items()}) for i in range(4)]))
    assert np.allclose(score.sum(1),1) and state.selection['rf_prior_gate']['new_rows']==2


def test_rf_gate_physical_OOF_uses_only_train_support(monkeypatch):
    calls=[];original=gate.fit
    def capture(kernel,labels,old_count):calls.append(len(labels));return original(kernel,labels,old_count)
    monkeypatch.setattr(gate,'fit',capture);y=np.repeat(np.arange(8),3)
    state=model.fit(features(514,24),y,list(map(str,range(24))),fixed=dict(rule(),rf_prior_gate_rule=gate.RULE))
    assert set(calls)=={16,24} and calls.count(24)==1 and len(calls)==73
    assert not state.selection['rf_prior_gate']['query_fit'] and state.selection['rf_prior_gate']['state_bytes']==24*8+16
    with pytest.raises(ValueError,match='RF prior evidence'):model.fit(features(514,24),y,list(map(str,range(24))),fixed=dict(rule(),rf_prior_gate_rule='wrong'))


def test_rf_prior_gate_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen,teacher_features
    assert screen.fit_function({'kernel_method':'kernel_gate_ensemble'}) is model.fit
    marker=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'kernel_gate_ensemble'},None,None,'cpu') is marker
    assert teacher_features.bridge_type('kernel_gate_ensemble') is teacher_features.OrbitEnsembleBridge
