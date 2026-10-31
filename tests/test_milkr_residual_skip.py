import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.residual_skip import ResidualSkip,complement_basis
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS
from experiments.pamf_phase2_practical.milkr.orbit_cca import blocks
from experiments.pamf_phase2_practical import teacher_features


def payload(seed,n):
    rng=np.random.default_rng(seed)
    x={name:rng.normal(size=(n,2*d)) for name,d in zip(BLOCKS[:5],DIMS[:5])}
    x['sig_views']=rng.normal(size=(n,672));x['teacher_phase_logits']=rng.normal(size=(n,4,6));return x


def test_shared_complement_is_orthogonal_to_both_maps_and_keeps_residual_variance():
    rng=np.random.default_rng(410);px=rng.normal(size=(40,8));py=rng.normal(size=(40,8));x=rng.normal(size=(120,40));y=rng.normal(size=(120,40))
    mean,p=complement_basis(px,py,x,y,12)
    assert np.allclose(px.T@p,0,atol=1e-12) and np.allclose(py.T@p,0,atol=1e-12) and np.allclose(p.T@p,np.eye(12))
    span=np.linalg.svd(np.concatenate((px,py),1),full_matrices=False)[0][:,:16];r=np.concatenate((x,y))-mean;r-=r@span@span.T
    eigen=np.linalg.eigvalsh(r.T@r/len(r))[-12:]
    assert np.allclose(np.sort(np.var(r@p,axis=0)),eigen)
    # Coincident CCA maps must remove only their actual rank, not 16 axes.
    _,same=complement_basis(px,px,x,y,28);assert np.allclose(px.T@same,0,atol=1e-12)
    with pytest.raises(FloatingPointError,match='Insufficient'):complement_basis(px,py,x[:1],y[:1],12)
    # A real FFT-like block: both maps cover all active coordinates, so any
    # residual variance is roundoff and must never produce a PCA direction.
    active=rng.normal(size=(120,20));low=np.concatenate((active,np.zeros_like(active)),1)
    covering=np.concatenate((np.eye(20),np.zeros((20,20))),0)
    with pytest.raises(FloatingPointError,match='Insufficient'):complement_basis(covering,covering,low,low,4)


def test_source_encoder_reload_energy_and_per_record_independence(tmp_path):
    a=payload(411,170);b=payload(412,170)
    for v in (a,b):v['fft'][:,96:]=0;v['f_emb'][:,160:]=0
    bridge=teacher_features.ResidualSkipBridge.fit(a,b)
    encoder=bridge.cca;assert isinstance(encoder,ResidualSkip)
    assert encoder.parameters[1] is None and encoder.parameters[3] is None
    assert sum(v.size for p in encoder.cca.parameters for v in p)+sum(v.size for p in encoder.parameters if p is not None for v in p)==404768
    for (_,_,px,py),native in zip(encoder.cca.parameters,encoder.parameters):
        if native is None:continue
        _,p=native
        assert np.linalg.norm(px.T@p)<1e-7 and np.linalg.norm(py.T@p)<1e-7
    bridge.save(tmp_path/'encoder.npz');loaded=teacher_features.ResidualSkipBridge.load(tmp_path/'encoder.npz')
    query=payload(413,4)
    for path in (False,True):
        out=loaded.transform(query,restored=path,old_columns=[0,2,4,5]);one=loaded.transform({k:v[2:3] for k,v in query.items()},restored=path,old_columns=[0,2,4,5])
        assert all(np.allclose(v[2],one[k][0]) for k,v in out.items())
        for name in BLOCKS[:5]:
            if name in ('fft','f_emb'):
                expected=encoder.cca.transform(teacher_features.ordinary(query),restored=path)[name]
                assert np.array_equal(out[name],expected)
            else:assert np.allclose(np.sum(out[name][:,:64]**2,1),.5) and np.allclose(np.sum(out[name][:,64:]**2,1),.5)
        assert np.allclose(out['teacher_probs'],teacher_features.probabilities(query['teacher_phase_logits'],[0,2,4,5]))
    with np.load(tmp_path/'encoder.npz') as z:assert not any('covariance' in k or 'labels' in k or k.startswith(('fft_native_','f_emb_native_')) for k in z.files)
    assert all(not v.flags.writeable for p in encoder.parameters if p is not None for v in p)


def test_residual_skip_routes_to_original_C4_features_and_R31_prior(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen
    from experiments.pamf_phase2_practical.milkr.prior_fusion import fit
    assert teacher_features.bridge_type('residual_skip') is teacher_features.ResidualSkipBridge
    assert screen.fit_function({'kernel_method':'residual_skip'}) is fit
    marker=object();monkeypatch.setattr(teacher_features,'extract',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'residual_skip'},None,None,'cpu') is marker
