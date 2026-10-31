import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.paired_cca import PairedCCA,DIMS
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,blocks
from experiments.pamf_phase2_practical.milkr.source_class_metric import within_class_covariances,fuse_metric


def test_source_metric_suppresses_shared_class_noise_and_exports_only_encoder(tmp_path):
    rng=np.random.default_rng(1701);n=720;y=np.repeat(np.arange(6),120);raw={};rest={}
    parameters=[(np.zeros(d),np.zeros(d),np.eye(d,16),np.eye(d,16)) for d in DIMS];cca=PairedCCA(parameters,16)
    for name,d in zip(BLOCKS[:5],DIMS[:5]):
        latent=np.repeat(rng.normal(size=(6,16)),120,axis=0)+.02*rng.normal(size=(n,16));latent[:,0]+=rng.normal(size=n)*4
        raw[name]=np.pad(latent,((0,0),(0,d-16)));rest[name]=raw[name].copy()
    raw['sig_views']=np.concatenate([np.pad(raw['z_id'][:,:16],((0,0),(0,96))) for _ in range(3)],axis=1);rest['sig_views']=raw['sig_views'].copy()
    covariances=within_class_covariances(cca,raw,rest,y)
    assert fuse_metric(cca,covariances,0.) is cca
    fused=fuse_metric(cca,covariances,.75)
    original=blocks(cca.transform(raw,restored=False))[0][:,:16];after=blocks(fused.transform(raw,restored=False))[0][:,:16]
    def variance(x):return np.mean((x-np.stack([x[y==c].mean(0) for c in range(6)])[y])**2)
    assert variance(after)<variance(original)*.6
    fused.save(tmp_path/'encoder.npz');loaded=PairedCCA.load(tmp_path/'encoder.npz');out=loaded.transform(raw,restored=False)
    one={key:np.concatenate([loaded.transform({name:value[i:i+1] for name,value in raw.items()},restored=False)[key] for i in range(8)]) for key in out}
    assert all(np.allclose(value[:8],one[key]) for key,value in out.items())
    with np.load(tmp_path/'encoder.npz') as z:
        assert not any('covariance' in key or 'class_mean' in key or 'labels' in key for key in z.files)
    assert all(np.array_equal(value,loaded.transform(raw,restored=True)[key]) for key,value in out.items())
    with pytest.raises(ValueError,match='configuration'):fuse_metric(cca,covariances,1.)
