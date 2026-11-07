import math
from types import SimpleNamespace

import torch

from experiments.pamf_phase3_restoration.coupled_covariance import (
    coupled_covariance,mode_factor,normalize_modes,positive_diagonal,positive_diagonal_slope,scratch_coupled_covariance,
)
from experiments.pamf_phase3_restoration.signed_covariance import source_modes,scratch_signed_covariance


def test_positive_diagonal_has_identity_and_polynomial_tail_gradients():
    raw=torch.tensor([-50.,0.,50.],dtype=torch.float64,requires_grad=True)
    diagonal=positive_diagonal(raw)
    diagonal.sum().backward()
    assert (diagonal>0).all() and diagonal[1]==1
    assert raw.grad[1]==1 and raw.grad[0]>1e-4
    torch.testing.assert_close(raw.grad,positive_diagonal_slope(raw.detach()),rtol=1e-12,atol=1e-12)


def test_coupling_can_represent_correlations_and_variances_outside_point_one_to_one_point_nine():
    base=torch.eye(2,dtype=torch.float64)
    modes=normalize_modes(base,source_modes(base,rank=2)).double()
    weight=modes[0,0].square()
    angle=torch.tensor(math.pi/4,dtype=torch.float64)
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin())),torch.stack((angle.sin(),angle.cos()))))
    target=rotation@torch.diag(torch.tensor([.0006,12.46],dtype=torch.float64))@rotation.T
    wanted=torch.linalg.cholesky((target-(1-weight)*base)/weight)
    rows,columns=torch.tril_indices(2,2)
    raw=wanted[rows,columns]*math.sqrt(2)
    on_diagonal=rows==columns
    positive=wanted.diagonal()
    raw[on_diagonal]=(positive-positive.reciprocal())/2
    factor=mode_factor(raw[None],rank=2)
    actual,_=coupled_covariance(base,modes,factor)
    torch.testing.assert_close(actual[0],target,rtol=1e-10,atol=1e-10)
    assert torch.linalg.eigvalsh(actual).min()>0


def test_stored_mode_margin_keeps_covariance_spd_even_for_degenerate_factor():
    generator=torch.Generator().manual_seed(97)
    matrix=torch.randn(6,6,generator=generator,dtype=torch.float64)
    covariance=(matrix@matrix.T+.01*torch.eye(6,dtype=torch.float64)).float().double()
    modes=normalize_modes(covariance,source_modes(covariance,rank=3)).double()
    whitened=torch.linalg.solve_triangular(torch.linalg.cholesky(covariance),modes,upper=False)
    assert torch.linalg.eigvalsh(whitened.T@whitened).max()<1
    raw=torch.zeros(1,6,dtype=torch.float64,requires_grad=True)
    identity=mode_factor(raw,rank=3)
    same,change=coupled_covariance(covariance,modes,identity)
    assert change.count_nonzero()==0 and torch.equal(same,covariance[None])
    for factor in [torch.zeros(1,3,3,dtype=torch.float64),30*torch.eye(3,dtype=torch.float64)[None]]:
        changed,_=coupled_covariance(covariance,modes,factor)
        assert torch.linalg.eigvalsh(changed).min()>0


def test_scratch_physical_and_context_start_match_the_signed_control():
    old=scratch_signed_covariance(2026092701)
    new=scratch_coupled_covariance(2026092701)
    assert all(torch.equal(value,new.physical.state_dict()[key]) for key,value in old.physical.state_dict().items())
    assert all(torch.equal(value,new.detail.state_dict()[key]) for key,value in old.detail.state_dict().items() if key.startswith(('stem.','context.')))
    assert new.detail.output.weight.shape==(2080,64) and not new.detail.output.weight.any()


def test_actual_noisy_readout_learns_coupling_only_and_keeps_deployed_buffers():
    from experiments.pamf_phase3_restoration.coupled_covariance_trial import coupled_readout,configuration,ARM
    from experiments.pamf_phase3_restoration.covariance_detail import detail_objective
    from experiments.pamf_phase3_restoration.objective import PairedObjective
    cfg=configuration(ARM)
    assert (cfg['epochs'],cfg['updates'],cfg['training_stages'])==(30,7410,1)
    model=scratch_coupled_covariance(2026092701)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    received=torch.randn(1,2,256,generator=torch.Generator().manual_seed(101));clean=.7*received
    class Reader:
        def features(self,x):return x.flatten(1)[:,:160]
    reader=Reader();target=reader.features(clean);scale=torch.ones(160);objective=PairedObjective()
    other=.08*torch.eye(512);modes=normalize_modes(other,source_modes(other))
    priors=(.1*torch.ones(5,512),other[None].expand(5,-1,-1),modes[None].expand(5,-1,-1))
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    with torch.no_grad():deployed=model(received)
    restored,aux,loss,metrics=coupled_readout(model,reader,received,clean,target,scale,objective,
        source_indices=torch.tensor([0]),epoch=2,priors=priors,assignment=torch.tensor([0]))
    expected,_=detail_objective(reader,restored,clean,target,scale,objective)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    assert not torch.equal(deployed,restored) and metrics['receiver_mismatch_prior_fraction'].item()==1
    loss.mean().backward()
    assert all(parameter.grad is None for parameter in model.physical.parameters())
    assert model.detail.output.weight.grad.abs().sum()>0 and torch.isfinite(model.detail.output.weight.grad).all()
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())


def test_gpu1_policy_uses_only_original_three_process_admission_limit():
    from experiments.pamf_phase3_restoration.variance_trial import recovery_gpu_choice
    def runner(count):
        def run(command,**kwargs):
            assert command[command.index('-i')+1]=='1'
            if '--query-gpu=index,uuid,memory.free' in command:
                return SimpleNamespace(returncode=0,stdout='1, GPU-card1, 18000\n',stderr='')
            return SimpleNamespace(returncode=0,stdout=''.join(f'GPU-card1, {i}\n' for i in range(count)),stderr='')
        return run
    assert recovery_gpu_choice(runner(2),allowed_gpus=(1,),maximum_processes=3)[0]['GPU']==1
    assert recovery_gpu_choice(runner(3),allowed_gpus=(1,),maximum_processes=3)==[]
