import pytest
import torch

from experiments.pamf_phase3_restoration.coupled_covariance import coupled_covariance,mode_factor
from experiments.pamf_phase3_restoration.low_rank_posterior import conditional_prior_posterior
from experiments.pamf_phase3_restoration.projected_prior_detail import fixed_posterior_frame


def case(batch=2,size=12,rank=4,seed=41):
    g=torch.Generator().manual_seed(seed)
    matrix=torch.randn(size,size,generator=g,dtype=torch.float64)
    covariance=matrix@matrix.T/size+.1*torch.eye(size,dtype=torch.float64)
    modes=.7*torch.linalg.cholesky(covariance)[:,:rank]
    action=torch.randn(batch,size,size,generator=g,dtype=torch.float64)/size**.5+.5*torch.eye(size,dtype=torch.float64)
    observed=torch.randn(batch,size,generator=g,dtype=torch.float64)
    mean=torch.randn(batch,size,generator=g,dtype=torch.float64)*.1
    raw=torch.randn(batch,rank*(rank+1)//2,generator=g,dtype=torch.float64)*.3
    variance=torch.tensor([.02,.2],dtype=torch.float64)[:batch]
    return action,observed,mean,covariance,modes,raw,variance


@pytest.mark.parametrize('batched_prior',[False,True])
def test_mean_and_factor_gradients_match_dense_reference(batched_prior):
    action,observed,mean,covariance,modes,raw,variance=case()
    if batched_prior:
        covariance=torch.stack((covariance,1.3*covariance));modes=torch.stack((modes,1.3**.5*modes))
    target=observed*.7
    outputs=[];gradients=[]
    for low_rank in (False,True):
        m=mean.clone().requires_grad_();r=raw.clone().requires_grad_();factor=mode_factor(r,rank=4)
        if low_rank:
            restored,error=conditional_prior_posterior(action,observed,m,covariance,modes,factor,variance)
        else:
            changed,_=coupled_covariance(covariance,modes,factor)
            restored,_,_,error=fixed_posterior_frame(action,observed,m,changed,variance)
        assert error.max()<1e-10
        loss=(restored-target).square().mean()
        gradients.append(torch.autograd.grad(loss,(r,m)));outputs.append(restored.detach())
    torch.testing.assert_close(outputs[1],outputs[0],rtol=1e-10,atol=1e-11)
    for actual,expected in zip(gradients[1],gradients[0]):
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-11)


def test_identity_zero_and_strong_factors_match_at_low_noise():
    action,observed,mean,covariance,modes,raw,variance=case(batch=1,size=16,rank=5)
    covariance=torch.diag(torch.logspace(-4,1,16,dtype=torch.float64))
    modes=(1-1e-5)**.5*torch.linalg.cholesky(covariance)[:,-5:]
    variance.fill_(1e-7)
    for magnitude in (0.,.025,1.,3.6):
        factor=magnitude*torch.eye(5,dtype=torch.float64)[None]
        changed,_=coupled_covariance(covariance,modes,factor)
        expected,_,_,_=fixed_posterior_frame(action,observed,mean,changed,variance)
        actual,error=conditional_prior_posterior(action,observed,mean,covariance,modes,factor,variance)
        torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-8)
        assert error.max()<1e-8


def test_physical_parameter_gradients_are_explicitly_rejected():
    action,observed,mean,covariance,modes,raw,variance=case()
    factor=mode_factor(raw,rank=4)
    for index in (0,3,4,6):
        values=[action,observed,mean,covariance,modes,factor,variance]
        values[index]=values[index].clone().requires_grad_()
        with pytest.raises(ValueError,match='fixed physical'):
            conditional_prior_posterior(*values)


def test_autograd_does_not_save_a_large_cholesky_factor():
    action,observed,mean,covariance,modes,raw,variance=case(size=64,rank=4)
    saved=[]
    def pack(tensor):
        saved.append((tuple(tensor.shape),tensor.requires_grad));return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        factor=mode_factor(raw.clone().requires_grad_(),rank=4)
        restored,_=conditional_prior_posterior(action,observed,mean.clone().requires_grad_(),covariance,modes,factor,variance)
        restored.square().mean().backward()
    # Constant large matrices may be saved for matrix-vector derivatives. The
    # differentiable factorization and its trainable square state stay rank4.
    assert not any(shape[-2:]==(64,64) and grad for shape,grad in saved if len(shape)>=2)


def test_probe_keeps_all_scenes_receivers_and_both_legal_source_roles():
    import numpy as np
    from experiments.pamf_phase3_restoration.low_rank_source_probe import SCENES,RECEIVERS,select_rows
    cells=[(scene,rx,role) for scene in SCENES for rx in RECEIVERS for role in ['L_s']+['U_s']*6]
    scene=np.asarray([x[0] for x in cells]);rx=np.asarray([x[1] for x in cells]);roles=np.asarray([x[2] for x in cells])
    selected=select_rows(scene,roles,rx)
    assert len(selected)==len(set(selected.tolist()))==224
    assert int((roles[selected]=='L_s').sum())==35 and int((roles[selected]=='U_s').sum())==189
    for name in SCENES:assert set(rx[selected[scene[selected]==name]].tolist())==set(RECEIVERS)
    wrong=roles.copy();wrong[0]='V'
    with pytest.raises(ValueError,match='fixed quota'):select_rows(scene,wrong,rx)


def test_probe_rejects_wrong_checkpoint_or_non_source_provenance():
    from experiments.pamf_phase3_restoration.low_rank_source_probe import REFERENCE_COMMIT,reference,validate_reference
    checkpoint=dict(code_commit=REFERENCE_COMMIT,epoch=20,updates=4940,config=reference.configuration(reference.ARM))
    smoke=dict(status='PASS',TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0)
    validate_reference(checkpoint,smoke)
    with pytest.raises(ValueError,match='E20'):validate_reference(dict(checkpoint,epoch=30),smoke)
    with pytest.raises(ValueError,match='provenance'):validate_reference(checkpoint,dict(smoke,target_inputs=1))


def test_second_probe_cell_is_saved_without_overwriting_the_first(tmp_path):
    import json
    from experiments.pamf_phase3_restoration.low_rank_source_probe import save_cell
    records=[]
    for scene in ('clean','practical_high'):
        records.append(dict(scene=scene,count=32))
        save_cell(tmp_path,records)
    saved=[json.loads(p.read_text()) for p in sorted(tmp_path.glob('cell-*.json'))]
    assert saved==records
    with pytest.raises(FileExistsError):save_cell(tmp_path,records)
