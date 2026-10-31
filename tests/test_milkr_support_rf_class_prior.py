import numpy as np
import pytest
from tests.test_milkr_orbit_ensemble import features,rule
from experiments.pamf_phase2_practical.milkr import support_rf_class_prior as cal,support_kernel_gate as kg,orbit_ensemble as model,prior_fusion as prior


def test_RF_class_product_order_neutrality_and_head_gate_original_input():
    y=np.array([2,0,3,1,0,2,1,3]);c=cal.Calibrator.fit(y,3);p=np.array([[.6,.3,.1],[.2,.2,.6]])
    a=np.array([[1,4,100,2,4,1,2,100],[3,1,100,2,1,3,2,100]],float)
    expected=p*np.array([[4,2,1],[1,2,3]]);expected/=expected.sum(1,keepdims=True)
    assert np.allclose(c.probabilities(p,a),expected) and np.allclose(c.probabilities(p,np.ones((2,8))),p)
    assert np.allclose(cal.Calibrator.fit(y[::-1],3).probabilities(p,a[:,::-1]),expected)
    assert not np.allclose(c.probabilities(p,a[:,::-1]),expected)
    class HeadGate:
        def values(self,value):assert np.array_equal(value,p);return np.array([.2,.7])
    value=prior.prior(p,5,HeadGate(),expected)
    assert np.allclose(value[:,:3],.5*np.array([.2,.7])[:,None]*(expected-1/3)) and np.allclose(value.sum(1),0) and (value[:,3:]==0).all()
    with pytest.raises(FloatingPointError,match='Nonpositive'):c.probabilities(p,np.zeros((2,8)))


def test_RF_class_residual_equations_and_query_independence():
    x=features(611,8);y=np.arange(8);fixed=dict(rule(),rf_prior_gate_rule=kg.RULE,rf_class_prior_rule=cal.RULE)
    state=model.fit(x,y,list(map(str,y)),fixed=fixed);sim=model.similarities(state.features,state.features,state.config[0]);_,_,mix,scales=state.config
    k=model.make_kernel(sim,state.tau,mix,scales);affinity=model.make_kernel(sim,state.tau,0.,scales)
    pcal=state.rf_class.probabilities(x['teacher_probs'],affinity)
    bias=prior.prior(x['teacher_probs'],8,state.prior_gate,pcal)*state.rf_gate.factors(k)[:,None]
    system=np.block([[k+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solved=np.linalg.solve(system,np.concatenate((np.eye(8)-bias,np.zeros((1,8)))))
    assert np.allclose(state.alpha,solved[:-1]) and np.allclose(state.intercept,solved[-1])
    q=features(613,4);s=state.score(q)
    assert np.allclose(s,np.concatenate([state.score({key:value[i:i+1] for key,value in q.items()}) for i in range(4)])) and np.allclose(s.sum(1),1)
    assert state.selection['rf_class_prior']['state_bytes']==8*(8+6) and not state.rf_class.labels.flags.writeable


def test_RF_class_OOF_only_current_physical_train_labels(monkeypatch):
    calls=[];original=cal.Calibrator.fit.__func__
    def capture(cls,labels,old_count):calls.append(len(labels));return original(cls,labels,old_count)
    monkeypatch.setattr(cal.Calibrator,'fit',classmethod(capture));y=np.repeat(np.arange(8),3)
    state=model.fit(features(614,24),y,list(map(str,range(24))),fixed=dict(rule(),rf_prior_gate_rule=kg.RULE,rf_class_prior_rule=cal.RULE))
    assert set(calls)=={16,24} and calls.count(24)==1 and len(calls)==73 and not state.selection['rf_class_prior']['query_fit']
    with pytest.raises(ValueError,match='RF class prior'):model.fit(features(614,24),y,list(map(str,range(24))),fixed=dict(rule(),rf_class_prior_rule='wrong'))


def test_RF_class_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen,teacher_features
    assert screen.fit_function({'kernel_method':'rf_class_ensemble'}) is model.fit
    marker=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'rf_class_ensemble'},None,None,'cpu') is marker
    assert teacher_features.bridge_type('rf_class_ensemble') is teacher_features.OrbitEnsembleBridge
