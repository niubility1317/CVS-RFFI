import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import phase_kernel as model
from experiments.pamf_phase2_practical.milkr.phase_encoder import PhaseEncoder
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS


def examples(seed,n):
    rng=np.random.default_rng(seed)
    result={key:rng.normal(size=(n,4,d)) for key,d in zip(BLOCKS[:5],DIMS[:5])}
    result['sig_views']=rng.normal(size=(n,4,336))
    return result


def test_full_distribution_kernel_is_PSD_phase_invariant_and_not_RBF_of_mean():
    a=model.normalize(examples(21,10));b={key:np.roll(value,1,axis=1) for key,value in a.items()}
    for geometry in ('signature','signature_identity','signature_historical'):
        sim=model.similarity(a,a,geometry)
        K=model.kernel(sim,.7,.5,'multi')
        assert K.shape==(10,10) and np.linalg.eigvalsh(K).min()>-1e-10
        assert np.all(np.diag(K)>0) and np.all(np.diag(K)<=1+1e-10)
        assert np.allclose(K,model.kernel(model.similarity(a,b,geometry),.7,.5,'multi'))
        assert not np.allclose(K,model.base.make_kernel(sim.mean(axis=(1,3)),.7,.5,'multi'))


def test_K1_keeps_one_physical_label_and_query_is_independent():
    support=examples(22,8);query=examples(23,5);y=np.arange(8);ids=[str(i) for i in range(8)]
    rule=dict(config=['signature_identity','nearest',0.,'single'],ridge=.1,ridge_reference_classes=6)
    state=model.fit(support,y,ids,old_count=6,fixed=rule)
    assert state.alpha.shape==(8,8) and state.selection['physical_count']==8
    assert state.ridge==pytest.approx(.1*8/6)
    scores=state.score(query)
    individual=np.concatenate([state.score({key:value[i:i+1] for key,value in query.items()}) for i in range(5)])
    assert np.allclose(scores,individual)
    assert np.allclose(scores,state.score({key:np.roll(value,2,axis=1) for key,value in query.items()}))
    with pytest.raises(ValueError,match='physical'):model.fit(support,y,['same']*8,fixed=rule)
    with pytest.raises(ValueError,match='source'):model.fit(support,y,ids,fixed={})


def test_phase_encoder_fit_save_reload_preserves_pairs_and_schema(tmp_path):
    raw=examples(24,80);encoder=PhaseEncoder.fit(raw,raw)
    encoder.save(tmp_path/'encoder.npz');loaded=PhaseEncoder.load(tmp_path/'encoder.npz')
    left=loaded.transform(raw,restored=False);right=loaded.transform(raw,restored=True)
    assert all(np.allclose(value,right[key],atol=1e-7) and value.shape==raw[key].shape for key,value in left.items())
    shifted=loaded.transform({key:np.roll(value,1,axis=1) for key,value in raw.items()},restored=False)
    assert all(np.allclose(shifted[key],np.roll(value,1,axis=1)) for key,value in left.items())
    with np.load(tmp_path/'encoder.npz') as z:assert str(z['schema'])=='source_paired_cca_v1' and int(z['rank'])==64


def test_physical_K2_LOO_fit_agrees_with_intercept_normal_equations():
    features=examples(25,12);y=np.repeat(np.arange(6),2);ids=[str(i) for i in range(12)]
    rule=dict(config=['signature','all',0.,'single'],ridge=.1,ridge_reference_classes=6)
    state=model.fit(features,y,ids,old_count=4,fixed=rule)
    geometry,_,mix,scales=state.config;K=model.kernel(model.similarity(state.features,state.features,geometry),state.tau,mix,scales)
    A=np.block([[K+state.ridge*np.eye(12),np.ones((12,1))],[np.ones((1,12)),np.zeros((1,1))]])
    solved=np.linalg.solve(A,np.concatenate((np.eye(6)[y],np.zeros((1,6)))))
    assert np.allclose(solved[:-1],state.alpha) and np.allclose(solved[-1],state.intercept)
    assert state.selection['physical_K']==2 and state.selection['candidate_count']==120
    assert not state.selection['K1_current_class_scaling']
