import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import support_ridge_prior as gate


def test_balanced_gate_matches_weighted_ridge_and_preserves_batch_independence():
    rng=np.random.default_rng(91);p=rng.dirichlet(np.ones(4),12);y=np.repeat(np.arange(6),2);state=gate.fit(p,y)
    x=gate.features(p);old=y<4;weights=np.where(old,.5/old.sum(),.5/(~old).sum());target=old.astype(float)
    design=np.column_stack((np.ones(len(x)),x));penalty=np.diag([0.]+[state.ridge]*x.shape[1])
    beta=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*target))
    assert np.allclose(state.values(p),np.clip(design@beta,0,1))
    mean=state.mean.copy();coefficient=state.coefficient.copy()
    batch=rng.dirichlet(np.ones(4),7);assert np.allclose(state.values(batch),[state.values(row[None])[0] for row in batch])
    assert np.array_equal(state.mean,mean) and np.array_equal(state.coefficient,coefficient)
    # Repeating all physical observations within one group does not change its total weight.
    duplicated=gate.fit(np.concatenate((p,p[~old])),np.concatenate((y,y[~old])))
    assert np.allclose(duplicated.values(batch),state.values(batch))
    assert state.audit()['state_bytes']==2*(4+2)*8 and not state.audit()['query_fit']


def test_old_only_and_unseparated_modes_are_explicit_and_bounded():
    p=np.full((6,4),.25);old=gate.fit(p[:4],np.arange(4));constant=gate.fit(p,np.arange(6))
    assert old.mode=='old_only' and np.array_equal(old.values(p),np.ones(6))
    assert constant.mode=='constant_unseparated' and np.array_equal(constant.values(p),np.full(6,.5))
    with pytest.raises(ValueError,match='Old physical'):gate.fit(p[:2],[4,5])
    with pytest.raises(ValueError,match='Different'):constant.values(np.full((2,6),1/6))


def test_physical_OOF_calibration_uses_training_support_only(monkeypatch):
    from experiments.pamf_phase2_practical.milkr import prior_fusion as model
    from tests.test_milkr_teacher_kernel import examples
    rng=np.random.default_rng(92);x=examples(92,24);y=np.repeat(np.arange(8),3);x['teacher_probs']=rng.dirichlet(np.ones(6),len(y));calls=[];original=gate.fit
    def audit(p,labels):calls.append((p.copy(),labels.copy()));return original(p,labels)
    monkeypatch.setattr(gate,'fit',audit)
    fixed=dict(config=['signature','all',0.,'single'],ridge=.1,ridge_reference_classes=6,prior_weight=.5,fusion_policy=model.FUSION_POLICY,confidence_rule=gate.RULE)
    state=model.fit(x,y,list(map(str,range(len(y)))),old_count=6,fixed=fixed)
    assert state.selection['physical_OOF']
    assert len(calls)==len(model.sig.CONFIGS)*3+1
    assert all(len(labels)<len(y) and len(p)==len(labels) for p,labels in calls[:-1])
    assert len(calls[-1][1])==len(y) and state.selection['prior_confidence_gate']['rule']==gate.RULE
    assert state.selection['extra_state_bytes']==x['teacher_probs'].nbytes+state.prior_gate.state_bytes


def test_ridge_prior_method_route():
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    from experiments.pamf_phase2_practical.milkr.prior_fusion import fit
    assert fit_function({'kernel_method':'ridge_prior'}) is fit
