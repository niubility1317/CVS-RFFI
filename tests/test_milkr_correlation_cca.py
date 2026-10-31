import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.correlation_cca import pair_projection
from experiments.pamf_phase2_practical.milkr.paired_cca import inverse_root
from experiments.pamf_phase2_practical import teacher_features
from tests.test_milkr_residual_skip import payload


def test_weighted_CCA_equations_and_weaker_directions_shrink():
    rng=np.random.default_rng(710);x=rng.normal(size=(400,40));y=x*np.linspace(.1,.95,40)+rng.normal(size=x.shape)*.5
    (mx,my,px,py),c=pair_projection(x,y,24);xc=x-mx;yc=y-my
    sx=xc.T@xc/len(x);sy=yc.T@yc/len(y);cross=xc.T@yc/len(x)
    rx=.01*np.trace(sx)/40;ry=.01*np.trace(sy)/40
    assert np.allclose(px.T@(sx+rx*np.eye(40))@px,np.diag(c))
    assert np.allclose(py.T@(sy+ry*np.eye(40))@py,np.diag(c))
    assert np.allclose(px.T@cross@py,np.diag(c*c)) and (c>=0).all() and (c<=1).all()
    wx=inverse_root(sx);wy=inverse_root(sy);u,s,vt=np.linalg.svd(wx@cross@wy,full_matrices=False)
    assert np.allclose(px,(wx@u[:,:24])*np.sqrt(s[:24])) and np.allclose(py,(wy@vt.T[:,:24])*np.sqrt(s[:24]))


def test_dual_source_schema_reload_query_independence_and_real_invariant_blocks(tmp_path):
    def dual(seed):
        a=payload(seed,170);b=payload(seed+1,170);a.update({'C4_'+key:v for key,v in b.items() if key!='teacher_phase_logits'})
        for prefix in ('','C4_'):a[prefix+'fft'][:,96:]=0;a[prefix+'f_emb'][:,160:]=0
        return a
    raw=dual(711);restored=dual(713);bridge=teacher_features.CorrelationEnsembleBridge.fit(raw,restored)
    assert sum(v.size for enc in bridge.encoders for p in enc.parameters for v in p)==557440
    bridge.save(tmp_path/'model.npz');loaded=teacher_features.CorrelationEnsembleBridge.load(tmp_path/'model.npz')
    with pytest.raises(ValueError,match='checkpoint'):teacher_features.OrbitEnsembleBridge.load(tmp_path/'model.npz')
    for path in (False,True):
        result=bridge.transform(raw,restored=path,old_columns=[0,2,4,5]);r=loaded.transform(raw,restored=path,old_columns=[0,2,4,5]);single=loaded.transform({k:v[2:3] for k,v in raw.items()},restored=path,old_columns=[0,2,4,5])
        assert all(np.array_equal(v,r[k]) and np.allclose(v[2],single[k][0]) for k,v in result.items())
        assert np.allclose(result['teacher_probs'],teacher_features.probabilities(raw['teacher_phase_logits'],[0,2,4,5]))
    for encoder in bridge.encoders:
        assert all(0<=s['min']<=s['mean']<=s['max']<=1 for s in encoder.correlation_summary)
        assert all(not v.flags.writeable for p in encoder.parameters for v in p)


def test_correlation_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen
    from experiments.pamf_phase2_practical.milkr.orbit_ensemble import fit
    assert screen.fit_function({'kernel_method':'corr_orbit_ensemble'}) is fit
    assert teacher_features.bridge_type('corr_orbit_ensemble') is teacher_features.CorrelationEnsembleBridge
    marker=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'corr_orbit_ensemble'},None,None,'cpu') is marker
