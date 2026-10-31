import numpy as np
import pytest
from tests.test_milkr_teacher_kernel import examples
from experiments.pamf_phase2_practical.milkr import prior_fusion as model,support_center
from experiments.pamf_phase2_practical.milkr.support_ridge_prior import RULE


def rule():return dict(config=['signature_identity','all',.5,'single'],ridge=.1,ridge_reference_classes=6,prior_weight=.5,fusion_policy=model.FUSION_POLICY,confidence_rule=RULE,support_center_strength=.5)


def test_support_center_formula_keeps_head_and_query_independence():
    x=model.normalize(examples(101,12));center=support_center.fit(x,.5);mapped=center.apply(x)
    for key in model.base.NAMES:
        expected=x[key]-.5*x[key].mean(0);expected/=np.linalg.norm(expected,axis=1,keepdims=True)
        assert np.allclose(mapped[key],expected)
    assert mapped['teacher_probs'] is x['teacher_probs']
    assert center.state_bytes==8576 and center.audit()['physical_count']==12
    query=model.normalize(examples(102,5));saved={k:v.copy() for k,v in center.means.items()};full=center.apply(query)
    single=[center.apply({k:v[i:i+1] for k,v in query.items()}) for i in range(5)]
    assert all(np.allclose(value,np.concatenate([row[key] for row in single])) for key,value in full.items())
    assert all(np.array_equal(value,saved[key]) for key,value in center.means.items())
    with pytest.raises(ValueError,match='half-shrink'):support_center.fit(x,1.)


def test_centered_KRR_matches_normal_equations_and_scores_original_features():
    x=examples(103,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=rule())
    sim=model.sig.similarity(state.features,state.features,state.config[0]);K=model.base.make_kernel(sim,state.tau,state.config[2],state.config[3])
    prior=model.prior(x['teacher_probs'],8,state.prior_gate);target=np.eye(8)[y]-prior
    A=np.block([[K+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solution=np.linalg.solve(A,np.concatenate((target,np.zeros((1,8)))))
    assert np.allclose(solution[:-1],state.alpha) and np.allclose(solution[-1],state.intercept)
    assert np.allclose(state.score(x),prior+K@state.alpha+state.intercept)
    assert state.selection['extra_state_bytes']==x['teacher_probs'].nbytes+state.prior_gate.state_bytes+8576
    query=examples(104,5);full=state.score(query);single=np.concatenate([state.score({k:v[i:i+1] for k,v in query.items()}) for i in range(5)])
    assert np.allclose(full,single) and np.allclose(full.sum(1),1)


def test_every_physical_OOF_center_is_fitted_only_on_train(monkeypatch):
    x=examples(105,24);y=np.repeat(np.arange(8),3);calls=[];original=support_center.fit
    def capture(features,strength):calls.append(len(features['teacher_probs']));return original(features,strength)
    monkeypatch.setattr(support_center,'fit',capture);state=model.fit(x,y,list(map(str,range(24))),fixed=rule())
    assert set(calls)=={16,24} and calls.count(24)==1 and len(calls)==len(model.sig.CONFIGS)*3+1
    assert state.selection['support_center']['physical_count']==24 and not state.selection['support_center']['query_fit']


def test_center_prior_method_route():
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    assert fit_function({'kernel_method':'center_prior'}) is model.fit
