import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.episodic_metric import Episodes,EpisodicMetric,learn,SCHEMA
from experiments.pamf_phase2_practical.milkr.orbit_cca import OrbitCCA
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS


def metadata():
    labels=np.repeat(np.arange(4),80);rx=np.tile(np.repeat([3,8],40),4)
    ids=[f'{label}/{r}/{i%20}' for label,r,i in zip(labels,rx,np.tile(np.arange(80),4))]
    return labels,rx,ids


def test_episodes_keep_physical_K_and_cross_RX_under_augmented_replay():
    y,rx,ids=metadata();episodes=Episodes(y,rx,ids,42);paths=set();budgets=set()
    for step in range(24):
        sup,held,k,sp,qp=episodes.sample(step);paths.add((sp,qp));budgets.add(k)
        assert len(sup)==4*k and len(held)==32
        assert len({ids[i] for i in sup})==len(sup)
        assert not {ids[i] for i in sup}&{ids[i] for i in held}
        assert len(np.unique(rx[sup]))==len(np.unique(rx[held]))==1 and rx[sup[0]]!=rx[held[0]]
        assert np.array_equal(np.unique(y[sup],return_counts=True)[1],np.full(4,k))
    assert paths=={(0,0),(1,0),(0,1),(1,1)} and budgets=={1,5,20}
    broken=ids.copy();broken[80]=broken[0]
    with pytest.raises(ValueError,match='changes'):Episodes(y,rx,broken,42)


def test_metric_training_learns_bounded_source_weights_without_changing_inputs():
    y,rx,ids=metadata();rng=np.random.default_rng(43)
    z=rng.normal(size=(len(y),8,64));z[:,:,0]+=(y[:,None]-1.5)*3
    z/=np.linalg.norm(z,axis=2,keepdims=True)
    values=[]
    for index,d in enumerate(DIMS):
        a=np.zeros((len(y),d));a[:,:64]=z[:,index];values.append(a)
    features=dict(zip(BLOCKS[:5],values[:5]));features['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
    before={k:v.copy() for k,v in features.items()}
    gains,audit=learn(features,features,y,rx,ids,44,steps=24)
    assert gains.shape==(8,64) and gains.min()>=.25 and gains.max()<=4
    assert not np.allclose(gains,1) and audit['steps']==24 and audit['train_physical_ids']==160
    assert audit['cross_RX'] and all(np.array_equal(v,before[k]) for k,v in features.items())


def test_metric_encoder_reload_and_batch_independence(tmp_path):
    rng=np.random.default_rng(45);parameters=[];raw={}
    for d in DIMS:parameters.append((np.zeros(2*d),np.zeros(2*d),rng.normal(size=(2*d,64)),rng.normal(size=(2*d,64))))
    for key,d in zip(BLOCKS[:5],DIMS[:5]):raw[key]=rng.normal(size=(5,2*d))
    raw['sig_views']=rng.normal(size=(5,672));cca=OrbitCCA(parameters)
    identity=EpisodicMetric(cca,np.ones((8,64)))
    for path in (False,True):
        a=identity.transform(raw,restored=path);b=cca.transform(raw,restored=path)
        assert all(np.allclose(v,b[k]) for k,v in a.items())
    model=EpisodicMetric(cca,rng.uniform(.25,4,(8,64)));model.save(tmp_path/'encoder.npz');loaded=EpisodicMetric.load(tmp_path/'encoder.npz')
    for path in (False,True):
        full=loaded.transform(raw,restored=path);one=loaded.transform({k:v[2:3] for k,v in raw.items()},restored=path)
        assert all(np.allclose(v[2],one[k][0]) for k,v in full.items())
        assert all(np.array_equal(v,model.transform(raw,restored=path)[k]) for k,v in full.items())
    with np.load(tmp_path/'encoder.npz') as payload:
        assert str(payload['schema'])==SCHEMA
        assert not any('label' in key or 'covariance' in key or 'physical' in key for key in payload.files)
    with pytest.raises(ValueError,match='bounded'):EpisodicMetric(cca,np.zeros((8,64)))


def test_source_metric_routes_and_source_label_requirement():
    from experiments.pamf_phase2_practical.teacher_features import bridge_type,EpisodicMetricBridge
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    from experiments.pamf_phase2_practical.milkr.prior_fusion import fit
    assert bridge_type('episodic_metric') is EpisodicMetricBridge
    assert fit_function({'kernel_method':'episodic_metric'}) is fit
    with pytest.raises(TypeError):EpisodicMetric.fit({}, {})
