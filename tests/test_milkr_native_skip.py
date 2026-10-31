import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.native_skip import NativeSkip,SCHEMA
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS
from experiments.pamf_phase2_practical.teacher_features import NativeSkipBridge,TeacherBridge,bridge_type


def features(seed,n=90):
    rng=np.random.default_rng(seed)
    result={name:rng.normal(size=(n,2*d)) for name,d in zip(BLOCKS[:5],DIMS[:5])}
    result['sig_views']=rng.normal(size=(n,672))
    return result


def test_equal_energy_preserves_native_directions_and_query_independence(tmp_path):
    x=features(71);encoder=NativeSkip.fit(x,x)
    left=encoder.transform(x,restored=False);right=encoder.transform(x,restored=True)
    assert all(np.allclose(left[k],right[k],atol=1e-7) for k in left)
    for key,d in zip(BLOCKS[:5],DIMS[:5]):
        assert left[key].shape==(90,d)
        assert np.allclose((left[key][:,:64]**2).sum(1),.5)
        assert np.allclose((left[key][:,64:]**2).sum(1),.5)
    encoder.save(tmp_path/'enc.npz');loaded=NativeSkip.load(tmp_path/'enc.npz')
    for path in (False,True):
        value=loaded.transform(x,restored=path)
        one=loaded.transform({k:v[5:6] for k,v in x.items()},restored=path)
        assert all(np.allclose(one[k][0],v[5]) for k,v in value.items())
        assert all(np.array_equal(v,encoder.transform(x,restored=path)[k]) for k,v in value.items())
    with np.load(tmp_path/'enc.npz') as z:
        assert str(z['schema'])==SCHEMA
        assert not any('label' in k or 'covariance' in k or 'class_mean' in k for k in z.files)
    with pytest.raises(FileExistsError):encoder.save(tmp_path/'enc.npz')
    with pytest.raises(ValueError,match='path'):loaded.transform(x,restored=1)


def test_native_teacher_payload_does_not_train_encoder_and_held_columns_removed():
    x=features(72);rng=np.random.default_rng(73);logits=rng.normal(size=(90,4,6))
    payload=dict(x,teacher_phase_logits=logits)
    encoder=NativeSkipBridge.fit(payload,payload)
    transformed=encoder.transform(payload,restored=False,old_columns=[0,2,4,5])
    changed=dict(payload,teacher_phase_logits=logits.copy());changed['teacher_phase_logits'][:,:,[1,3]]=1e6
    altered=encoder.transform(changed,restored=False,old_columns=[0,2,4,5])
    assert all(np.array_equal(v,altered[k]) for k,v in transformed.items())
    assert transformed['teacher_probs'].shape==(90,4)
    assert bridge_type('native_skip') is NativeSkipBridge and bridge_type('confidence_prior') is TeacherBridge


def test_native_method_routes_prior_and_orbit_extraction(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen,teacher_features
    from experiments.pamf_phase2_practical.milkr.prior_fusion import fit
    assert screen.fit_function({'kernel_method':'native_skip'}) is fit
    sentinel=object();monkeypatch.setattr(teacher_features,'extract',lambda *a:sentinel)
    assert screen.extract_for_method({'kernel_method':'native_skip'},None,None,'cpu') is sentinel
