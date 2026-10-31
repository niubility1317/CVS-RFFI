import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.source_bridge import FeatureBridge
from experiments.pamf_phase2_practical.milkr import signature_kernel as sig


def data(n):
    rng=np.random.default_rng(131);x={k:rng.normal(size=(n,96 if k=='fft' else 160)) for k in ('z_id','fft','t_emb','f_emb','pa_local')}
    x['sig_views']=rng.normal(size=(n,336));y={k:v.copy() for k,v in x.items()}
    for k in y:
        if k=='sig_views':y[k][:,[0,112,224]]+=.3
        else:y[k][:,0]+=.3
    return x,y


def test_source_pair_map_generalizes_small_shift_and_keeps_query_independent(tmp_path):
    x,y=data(384);train=np.arange(320);held=np.arange(320,384)
    subset=lambda d,i:{k:v[i] for k,v in d.items()}
    bridge=FeatureBridge.fit(subset(x,train),subset(y,train));raw=sig.normalize(subset(x,held));truth=sig.normalize(subset(y,held));out=bridge.transform(subset(x,held))
    before=sum(np.mean((raw[k]-truth[k])**2) for k in raw);after=sum(np.mean((out[k]-truth[k])**2) for k in raw)
    assert after<before*.5
    single={k:np.concatenate([bridge.transform(subset(x,[i]))[k] for i in held]) for k in out}
    assert all(np.allclose(out[k],single[k]) for k in out)
    bridge.save(tmp_path/'bridge.npz');loaded=FeatureBridge.load(tmp_path/'bridge.npz')
    assert all(np.array_equal(out[k],loaded.transform(subset(x,held))[k]) for k in out)
    assert all(not a.flags.writeable for p in loaded.parameters for a in p)


def test_bridge_rejects_unregistered_hyperparameters_and_incompatible_checkpoint(tmp_path):
    x,y=data(16)
    with pytest.raises(ValueError,match='hyperparameters'):FeatureBridge.fit(x,y,rank=4)
    np.savez(tmp_path/'bad.npz',schema=np.asarray('unrelated_checkpoint'))
    with pytest.raises(ValueError,match='checkpoint'):FeatureBridge.load(tmp_path/'bad.npz')
    with pytest.raises(ValueError,match='schema'):FeatureBridge([])
