import numpy as np
import pytest
import torch
from experiments.pamf_phase2_practical import phase_orbit as orbit
from experiments.pamf_phase2_practical.milkr.orbit_cca import OrbitCCA
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS


def fake_capture(reader,x):
    # A deliberately phase-sensitive frozen feature function.
    flat=x.flatten(1)
    v={name:flat[:,:d]+2 for name,d in zip(BLOCKS[:5],DIMS[:5]) if name!='fft'}
    v['sig_views']=flat[:,:336]+3
    return v,flat[:,:6]


def test_orbit_is_per_packet_invariant_and_G_precedes_rotations(monkeypatch):
    monkeypatch.setattr(orbit,'capture',fake_capture)
    rng=np.random.default_rng(20);x=rng.normal(size=(5,2,256)).astype(np.float32)
    raw,logits=orbit.extract(None,x,'cpu',batch=3)
    rotated,_=orbit.extract(None,orbit.rotate(x,1),'cpu',batch=2)
    assert all(np.allclose(raw[k],rotated[k],atol=1e-7) for k in raw)
    single=[orbit.extract(None,x[i:i+1],'cpu')[0] for i in range(len(x))]
    assert all(np.allclose(raw[k],np.concatenate([s[k] for s in single])) for k in raw)
    assert np.array_equal(logits,x.reshape(5,-1)[:,:6])
    calls=[]
    def engine(value):
        calls.append(value.clone())
        return (value+torch.tensor([.3,-.4]).reshape(1,2,1),)
    restored,_=orbit.extract(None,x,'cpu',engine,batch=3)
    direct,_=orbit.extract(None,x+np.array([.3,-.4],np.float32).reshape(1,2,1),'cpu',batch=3)
    assert len(calls)==2 and torch.equal(torch.cat(calls),torch.tensor(x))
    assert all(np.allclose(restored[k],direct[k]) for k in raw)
    assert np.all(restored['fft'][:,96:]==0)
    assert all(restored[k].shape==(5,2*d) for k,d in zip(BLOCKS[:5],DIMS[:5]))
    assert restored['sig_views'].shape==(5,672)


def test_orbit_encoder_is_frozen_schema_and_query_independent(tmp_path):
    rng=np.random.default_rng(21)
    parameters=[];features={}
    for name,d in zip(BLOCKS,DIMS):
        parameters.append((np.zeros(2*d),np.zeros(2*d),rng.normal(size=(2*d,64)),rng.normal(size=(2*d,64))))
        features[name]=rng.normal(size=(7,2*d))
    features['sig_views']=np.concatenate([features.pop('sig'+str(i)) for i in range(3)],axis=1)
    encoder=OrbitCCA(parameters,64);encoder.save(tmp_path/'encoder.npz');loaded=OrbitCCA.load(tmp_path/'encoder.npz')
    for path in (False,True):
        all_rows=loaded.transform(features,restored=path)
        one=loaded.transform({key:value[2:3] for key,value in features.items()},restored=path)
        assert all(np.allclose(one[key][0],value[2]) for key,value in all_rows.items())
        assert all(value.shape==(7,d) for value,d in zip([all_rows[key] for key in BLOCKS[:5]],DIMS[:5]))
        assert all_rows['sig_views'].shape==(7,336)
    with pytest.raises(ValueError,match='path'):loaded.transform(features,restored=1)
    with pytest.raises(FileExistsError):encoder.save(tmp_path/'encoder.npz')
    with np.load(tmp_path/'encoder.npz') as z:
        assert str(z['schema'])=='source_phase_orbit_cca_v1'
        assert not any('labels' in key or 'covariance' in key or 'class_mean' in key for key in z.files)


def test_source_fit_preserves_aligned_pair_and_method_routes():
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    from experiments.pamf_phase2_practical.milkr.calibrated_k1 import fit
    rng=np.random.default_rng(22);features={key:rng.normal(size=(80,2*d)) for key,d in zip(BLOCKS[:5],DIMS[:5])}
    features['sig_views']=rng.normal(size=(80,672))
    encoder=OrbitCCA.fit(features,features)
    left=encoder.transform(features,restored=False);right=encoder.transform(features,restored=True)
    assert all(np.allclose(left[key],right[key],atol=1e-7) for key in left)
    assert fit_function({'kernel_method':'phase_orbit'}) is fit
    with pytest.raises(ValueError,match='alignment'):OrbitCCA.fit(features,{key:value[:-1] for key,value in features.items()})
