import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.orbit_cca import OrbitCCA
from experiments.pamf_phase2_practical.milkr.source_bridge import BLOCKS,DIMS
from experiments.pamf_phase2_practical.milkr.bounded_pair import BoundedPairEncoder,train


def features(seed,n):
    rng=np.random.default_rng(seed)
    values={key:rng.normal(size=(n,2*d)) for key,d in zip(BLOCKS[:5],DIMS[:5])}
    values['sig_views']=rng.normal(size=(n,672))
    return values


def encoder():
    rng=np.random.default_rng(22)
    return OrbitCCA([(np.zeros(2*d),np.zeros(2*d),rng.normal(size=(2*d,64)),rng.normal(size=(2*d,64))) for d in DIMS])


def test_zero_initialization_is_original_encoder_and_bounded(tmp_path):
    import torch
    base=encoder();model=BoundedPairEncoder(base);x=features(23,7)
    for path in (False,True):
        original=base.transform(x,restored=path);out=model.transform(x,restored=path)
        assert all(np.allclose(value,original[key],atol=1e-14) for key,value in out.items())
    with torch.no_grad():
        for net in model.networks:
            for branch in net:branch[2].bias.fill_(100.)
    delta=model.residuals(x,restored=False)
    assert all(np.max(np.linalg.norm(value,axis=1))<=.250000001 for value in delta)
    model.save(tmp_path/'model.npz');loaded=BoundedPairEncoder.load(tmp_path/'model.npz')
    for path in (False,True):
        batch=loaded.transform(x,restored=path,batch=3)
        single=[loaded.transform({key:value[i:i+1] for key,value in x.items()},restored=path) for i in range(7)]
        assert all(np.allclose(value,np.concatenate([row[key] for row in single]),atol=1e-7) for key,value in batch.items())
    with pytest.raises(FileExistsError):model.save(tmp_path/'model.npz')
    with pytest.raises(ValueError,match='path'):model.transform(x,restored=1)
    with np.load(tmp_path/'model.npz') as z:assert not any('label' in name or 'covariance' in name or 'class_mean' in name for name in z.files)


def test_source_pair_training_reduces_alignment_loss_without_labels():
    base=OrbitCCA([(np.zeros(2*d),np.zeros(2*d),np.eye(2*d,64),np.eye(2*d,64)) for d in DIMS])
    x=features(24,80);y={key:np.tanh(value+.15) for key,value in x.items()}
    model,audit=train(base,x,y,device='cpu',steps=60,seed=25,batch=32)
    assert audit['final_alignment_loss']<audit['initial_alignment_loss']*.9
    assert audit['optimizer_updates']==60 and audit['TX_label_reads']==0
    assert not any(p.requires_grad for net in model.networks for p in net.parameters())
    for path in (False,True):
        out=model.transform(x if not path else y,restored=path)
        assert all(np.isfinite(value).all() for value in out.values())


def test_nonlinear_gate_rejects_one_source_K_query_loss_and_method_routes():
    from experiments.pamf_phase2_practical.milkr_nonlinear_source import assess
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    from experiments.pamf_phase2_practical.milkr.calibrated_k1 import fit
    control=[dict(k=k,S=s,Q=q,H=.6+q*.1,old_accuracy=.6,new_accuracy=.6) for k in (1,20) for s in (0,1) for q in (0,1)]
    candidate=[dict(row,H=row['H']+.01) for row in control]
    assert assess(candidate,control)['eligible']
    candidate[1]['H']=.60
    assert not assess(candidate,control)['eligible']
    assert fit_function({'kernel_method':'nonlinear_pair'}) is fit
