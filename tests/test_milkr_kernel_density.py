import numpy as np
import pytest
from tests.test_milkr_orbit_ensemble import features,rule
from experiments.pamf_phase2_practical.milkr import orbit_ensemble as model,prior_fusion as prior
from experiments.pamf_phase2_practical.milkr.kernel_density import Density,RULE


def test_positive_degree_congruence_PSD_and_constant_degree_identity():
    rng=np.random.default_rng(310);x=rng.normal(size=(9,5));x/=np.linalg.norm(x,axis=1,keepdims=True)
    k=np.exp(-np.maximum(2-2*x@x.T,0));density=Density.fit(k);normalized=density.cross(k,k)
    assert np.allclose(normalized,normalized.T) and np.linalg.eigvalsh(normalized).min()>-1e-12
    assert np.allclose(normalized,density.average*k/np.sqrt(k.mean(1)[:,None]*k.mean(1)[None,:]))
    constant=.2*np.ones((9,9))+.8*np.eye(9)
    assert np.allclose(Density.fit(constant).cross(constant,constant),constant)
    with pytest.raises(FloatingPointError,match='Nonpositive'):density.cross(np.zeros((2,9)),np.zeros((2,9)))


def test_density_residual_solve_and_independent_query_rows():
    x=features(311,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=dict(rule(),kernel_density_rule=RULE))
    sim=model.similarities(state.features,state.features,state.config[0]);_,_,mix,scales=state.config
    kernel=model.make_kernel(sim,state.tau,mix,scales);affinity=model.make_kernel(sim,state.tau,0.,scales)
    k=state.kernel_density.cross(kernel,affinity);target=np.eye(8)-prior.prior(x['teacher_probs'],8,state.prior_gate)
    a=np.block([[k+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solved=np.linalg.solve(a,np.concatenate((target,np.zeros((1,8)))))
    assert np.allclose(solved[:-1],state.alpha) and np.allclose(solved[-1],state.intercept)
    query=features(313,6);before=state.kernel_density.support_degree.copy();scores=state.score(query)
    assert np.allclose(scores,np.concatenate([state.score({key:value[i:i+1] for key,value in query.items()}) for i in range(6)]))
    assert np.array_equal(before,state.kernel_density.support_degree) and not state.kernel_density.support_degree.flags.writeable
    assert np.allclose(scores.sum(1),1) and state.selection['kernel_density']['state_bytes']==8*8+8


def test_density_OOF_fits_statistics_on_physical_train_only(monkeypatch):
    calls=[];original=Density.fit.__func__
    def capture(cls,affinity):calls.append(affinity.shape);return original(cls,affinity)
    monkeypatch.setattr(Density,'fit',classmethod(capture))
    y=np.repeat(np.arange(8),3);state=model.fit(features(314,24),y,list(map(str,range(24))),fixed=dict(rule(),kernel_density_rule=RULE))
    assert set(calls)=={(16,16),(24,24)} and calls.count((24,24))==1 and len(calls)==73
    assert state.selection['kernel_density']['physical_count']==24 and not state.selection['kernel_density']['query_fit']
    with pytest.raises(ValueError,match='density rule'):model.fit(features(314,24),y,list(map(str,range(24))),fixed=dict(rule(),kernel_density_rule='unknown'))


def test_density_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen,teacher_features
    assert screen.fit_function({'kernel_method':'density_orbit_ensemble'}) is model.fit
    marker=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'density_orbit_ensemble'},None,None,'cpu') is marker
    assert teacher_features.bridge_type('density_orbit_ensemble') is teacher_features.OrbitEnsembleBridge
