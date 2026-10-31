import numpy as np
from tests.test_milkr_teacher_kernel import examples
from experiments.pamf_phase2_practical.milkr import support_prior_gate as gate,prior_fusion as model


def rule():return dict(config=['signature_identity','all',0.,'single'],ridge=.1,ridge_reference_classes=6,prior_weight=.5,fusion_policy=model.FUSION_POLICY,confidence_rule=gate.RULE)


def test_entropy_confidence_and_support_group_calibration():
    p=np.array([[.5,.5],[1.,0.],[.9,.1],[.5,.5]])
    assert np.allclose(gate.confidence(p[:2]),[0,1])
    fitted=gate.fit(p,np.array([0,1,2,3]))
    assert fitted.mode=='positive_gap' and fitted.old_count==2 and fitted.old_rows==2 and fitted.new_rows==2
    assert np.array_equal(fitted.values(p[:2]),[0.,1.])
    assert np.allclose(fitted.values(p),np.concatenate([fitted.values(p[i:i+1]) for i in range(4)]))
    reversed_gate=gate.fit(p,np.array([2,3,0,1]))
    assert reversed_gate.mode=='no_positive_gap' and not reversed_gate.values(p).any()
    known_only=gate.fit(p[:2],np.array([0,1]))
    assert known_only.mode=='old_only' and np.array_equal(known_only.values(p),np.ones(4))


def test_calibrated_prior_normal_equations_and_batch_independence():
    x=examples(40,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=rule())
    sim=model.sig.similarity(state.features,state.features,state.config[0]);K=model.base.make_kernel(sim,state.tau,state.config[2],state.config[3])
    target=np.eye(8)[y]-model.prior(x['teacher_probs'],8,state.prior_gate)
    A=np.block([[K+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solved=np.linalg.solve(A,np.concatenate((target,np.zeros((1,8)))))
    assert np.allclose(solved[:-1],state.alpha) and np.allclose(solved[-1],state.intercept)
    query=examples(41,5);before=state.prior_gate.audit();scores=state.score(query)
    single=np.concatenate([state.score({k:v[i:i+1] for k,v in query.items()}) for i in range(5)])
    assert np.allclose(scores,single) and np.allclose(scores.sum(1),1) and state.prior_gate.audit()==before


def test_physical_OOF_fits_confidence_gate_only_on_train_support(monkeypatch):
    x=examples(42,24);y=np.repeat(np.arange(8),3);observed=[];original=gate.fit
    def capture(probability,labels):observed.append((len(labels),int((labels<6).sum()),int((labels>=6).sum())));return original(probability,labels)
    monkeypatch.setattr(gate,'fit',capture);state=model.fit(x,y,list(map(str,range(24))),fixed=rule())
    assert set(observed)=={(16,12,4),(24,18,6)} and observed.count((24,18,6))==1
    assert state.selection['physical_OOF'] and state.selection['prior_confidence_gate']==state.prior_gate.audit()


def test_confidence_prior_method_routes_to_function_prior_fit():
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    assert fit_function({'kernel_method':'confidence_prior'}) is model.fit
