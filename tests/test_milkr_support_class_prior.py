import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import support_class_prior as model,support_ridge_prior as binary,prior_fusion as prior,orbit_ensemble
from tests.test_milkr_orbit_ensemble import features,rule


def test_class_map_matches_weighted_ridge_and_membership_gate():
    rng=np.random.default_rng(141);p=rng.dirichlet(np.ones(6),24);y=np.repeat(np.arange(8),3);gate=model.fit(p,y);base=binary.fit(p,y)
    x=model.features(p);old=y<6;weights=np.where(old,.5/old.sum(),.5/(~old).sum());classes=np.full((len(p),6),1/6);classes[old]=np.eye(6)[y[old]];target=np.column_stack((old.astype(float),classes))
    design=np.column_stack((np.ones(len(p)),x));penalty=np.diag([0.]+[gate.ridge]*x.shape[1]);solution=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights[:,None]*target))
    assert np.allclose(gate.coefficient,solution[1:]) and np.allclose(gate.target_mean-gate.mean@gate.coefficient,solution[0])
    query=rng.dirichlet(np.ones(6),5);assert np.allclose(gate.values(query),base.values(query))
    probability=gate.calibrated_probabilities(query);assert (probability>=0).all() and np.allclose(probability.sum(1),1)
    assert gate.state_bytes==568 and gate.audit()['output_dimension']==7
    before=gate.coefficient.copy();single=np.concatenate([gate.calibrated_probabilities(row[None]) for row in query]);assert np.allclose(single,probability) and np.array_equal(before,gate.coefficient)


def test_class_calibration_corrects_support_head_confusion_without_changing_head():
    p=np.concatenate((np.eye(3)[[1,2,0]]*.8+.2/3,np.full((2,3),1/3)));y=np.arange(5);before=p.copy();gate=model.fit(p,y)
    assert np.array_equal(gate.calibrated_probabilities(p[:3]).argmax(1),np.arange(3))
    assert np.array_equal(p,before) and gate.new_rows==2 and gate.old_rows==3
    old=model.fit(p[:3],y[:3]);assert np.array_equal(old.values(p),np.ones(len(p)))
    flat=model.fit(np.full((5,3),1/3),y);assert flat.mode=='constant_unseparated' and np.allclose(flat.calibrated_probabilities(p),1/3)
    with pytest.raises(ValueError,match='Old physical'):model.fit(p[3:],[3,4])


def test_simplex_projection_and_prior_bounds():
    z=np.array([[3.,-2.,.4],[.2,.2,.2],[-4.,-3.,-2.]]);p=model.simplex(z)
    assert (p>=0).all() and np.allclose(p.sum(1),1)
    for row,out in zip(z,p):
        active=out>0;threshold=(row[active]-out[active]).mean();assert np.allclose(row[active]-out[active],threshold) and (row[~active]<=threshold+1e-12).all()
    rng=np.random.default_rng(142);support=rng.dirichlet(np.ones(6),8);gate=model.fit(support,np.arange(8));value=prior.prior(rng.dirichlet(np.ones(6),20),26,gate)
    assert np.allclose(value.sum(1),0) and not value[:,6:].any() and value[:,:6].max()<=.5*(1-1/6)+1e-12 and value[:,:6].min()>=-.5/6-1e-12


def test_class_prior_dual_kernel_normal_equations_and_train_only_OOF(monkeypatch):
    x=features(143,8);y=np.arange(8);fixed=dict(rule(),confidence_rule=model.RULE);state=orbit_ensemble.fit(x,y,list(map(str,y)),fixed=fixed)
    sim=orbit_ensemble.similarities(state.features,state.features,state.config[0]);K=orbit_ensemble.make_kernel(sim,state.tau,state.config[2],state.config[3]);bias=prior.prior(x['teacher_probs'],8,state.prior_gate)
    A=np.block([[K+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]]);solution=np.linalg.solve(A,np.concatenate((np.eye(8)[y]-bias,np.zeros((1,8)))))
    assert np.allclose(solution[:-1],state.alpha) and np.allclose(solution[-1],state.intercept) and np.allclose(state.score(x).sum(1),1)
    calls=[];original=model.fit
    def capture(p,y):calls.append(len(y));return original(p,y)
    monkeypatch.setattr(orbit_ensemble,'fit_class_gate',capture);x=features(144,24);y=np.repeat(np.arange(8),3);state=orbit_ensemble.fit(x,y,list(map(str,range(24))),fixed=fixed)
    assert set(calls)=={16,24} and calls.count(24)==1 and len(calls)==len(orbit_ensemble.sig.CONFIGS)*3+1
    assert state.selection['prior_confidence_gate']['rule']==model.RULE and state.selection['prior_confidence_gate']['state_bytes']==568
