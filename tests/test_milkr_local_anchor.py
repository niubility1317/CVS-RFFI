import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import local_anchor as model
from experiments.pamf_phase2_practical.milkr import signature_kernel as sig


def examples(seed,n):
    rng=np.random.default_rng(seed)
    return {key:rng.normal(size=(n,d)) for key,d in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160),('sig_views',336)]}


def rule():return dict(config=['signature_identity','nearest',.5,'multi'],ridge=.1,ridge_reference_classes=6,local_anchor_policy='column_norm_row_unit_v1')


def test_local_kernel_PSD_and_collapsed_prediction_matches_full_dual():
    x=examples(24,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=rule())
    query=examples(25,5);sim=sig.similarity(sig.normalize(query),state.features,state.config[0])
    qphi=model.embedding(sim,state.tau_vector,state.column_norm,state.config[3])
    ssim=sig.similarity(state.features,state.features,state.config[0]);sphi=model.embedding(ssim,state.tau_vector,state.column_norm,state.config[3])
    K=.5*ssim+.5*sphi@sphi.T
    assert np.linalg.eigvalsh(K).min()>-1e-10 and np.allclose(np.diag(K),1)
    expected=(.5*sim+.5*qphi@sphi.T)@state.alpha+state.intercept
    assert np.allclose(expected,state.score(query))
    single=np.concatenate([state.score({key:value[i:i+1] for key,value in query.items()}) for i in range(5)])
    assert np.allclose(expected,single)
    assert state.selection['centre_count']==8 and state.alpha.shape==(8,8)
    assert state.ridge==pytest.approx(.1*8/6)


def test_embedding_is_stable_for_far_query_and_support_column_norm_is_fixed():
    sim=np.eye(4);labels=np.arange(4)
    tau,column,phi=model.basis(sim,labels,'all','multi')
    raw=np.concatenate([np.exp(-(2-2*sim)/(tau[None,:]*s)) for s in (.25,1.,4.)],axis=1)
    assert np.allclose(column,np.linalg.norm(raw,axis=0)) and (column>=1).all()
    value=model.embedding(np.full((3,4),-1.),np.full(4,1e-6),column,'multi')
    assert np.isfinite(value).all() and np.allclose(np.linalg.norm(value,axis=1),1)
    assert np.allclose(phi,model.embedding(sim,tau,column,'multi'))


def test_physical_OOF_excludes_held_rows_from_centres_bandwidth_and_norms(monkeypatch):
    x=examples(26,24);y=np.repeat(np.arange(8),3);ids=list(map(str,range(24)));observed=[];original=model.context
    def capture(features,labels,config):
        observed.append(len(labels));return original(features,labels,config)
    monkeypatch.setattr(model,'context',capture)
    state=model.fit(x,y,ids,fixed=rule())
    assert set(observed)=={16,24} and observed.count(24)==1
    assert state.selection['physical_OOF'] and state.selection['candidate_count']==120
    assert state.selection['folds']==[dict(train=16,held=8)]*3
    assert state.selection['centre_count']==24
    with pytest.raises(ValueError,match='physical'):model.fit(x,y,['same']*24,fixed=rule())
    with pytest.raises(ValueError,match='policy'):model.fit(x,y,ids,fixed=dict(rule(),local_anchor_policy='query'))


def test_runtime_local_anchor_routes_to_the_custom_fit():
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    assert fit_function({'kernel_method':'local_anchor'}) is model.fit


def test_nearest_scale_excludes_same_class_and_applies_per_centre_column():
    angle=np.array([0.,.1,.3,2.]);z=np.stack((np.cos(angle),np.sin(angle)),axis=1);sim=z@z.T;y=np.array([0,0,1,2])
    tau,column,phi=model.basis(sim,y,'nearest','single');distance=np.maximum(2-2*sim,0)
    expected=np.array([min(row[y!=label]) for row,label in zip(distance,y)])
    assert np.allclose(tau,expected) and np.ptp(tau)>1
    raw=np.exp(-distance/expected[None,:]);expected_phi=raw/np.linalg.norm(raw,axis=0)[None,:]
    expected_phi/=np.linalg.norm(expected_phi,axis=1,keepdims=True)
    assert np.allclose(phi,expected_phi)
