import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.paired_cca import PairedCCA,DIMS,BLOCKS
from experiments.pamf_phase2_practical.milkr.source_bridge import blocks


def test_source_paired_cca_aligns_held_rotation_and_is_per_query_frozen(tmp_path):
    rng=np.random.default_rng(216);x={};y={}
    for name,d in zip(BLOCKS[:5],DIMS[:5]):
        a=rng.normal(size=(800,d));rotation=np.linalg.qr(rng.normal(size=(d,d)))[0];x[name]=a;y[name]=a@rotation
    x['sig_views']=rng.normal(size=(800,336));y['sig_views']=np.concatenate([x['sig_views'][:,i*112:(i+1)*112]@np.linalg.qr(rng.normal(size=(112,112)))[0] for i in range(3)],axis=1)
    sub=lambda data,i:{key:value[i] for key,value in data.items()}
    adapter=PairedCCA.fit(sub(x,slice(600)),sub(y,slice(600)),32)
    raw=adapter.transform(sub(x,slice(600,None)),restored=False);rest=adapter.transform(sub(y,slice(600,None)),restored=True)
    assert np.mean([np.mean(np.sum(a*b,axis=1)) for a,b in zip(blocks(raw),blocks(rest))])>.999
    individual={key:np.concatenate([adapter.transform(sub(x,[i]),restored=False)[key] for i in range(600,800)]) for key in raw}
    assert all(np.allclose(raw[key],individual[key]) for key in raw)
    adapter.save(tmp_path/'cca.npz');loaded=PairedCCA.load(tmp_path/'cca.npz')
    assert all(np.array_equal(value,loaded.transform(sub(x,slice(600,None)),restored=False)[key]) for key,value in raw.items())
    assert all(not a.flags.writeable for p in loaded.parameters for a in p)
    with pytest.raises(ValueError,match='path'):loaded.transform(x,restored=1)
    with pytest.raises(ValueError,match='rank'):PairedCCA.fit(x,y,8)
