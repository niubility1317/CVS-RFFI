import torch

from experiments.pamf_phase3_restoration import modulation_covariance_trial as candidate
from experiments.pamf_phase3_restoration.covariance_inverse import dense_action
from experiments.pamf_phase3_restoration.modulation_covariance import (
    ReceiverModulationHead, modulation_action_loss, modulation_supervision,
)


def test_receiver_head_identity_and_constant_magnitude_with_independent_phase():
    head = ReceiverModulationHead()
    tokens = torch.randn(2,64,128,generator=torch.Generator().manual_seed(34))
    pooled = tokens.mean(1)
    main,image = head(tokens,pooled)
    torch.testing.assert_close(main,torch.ones_like(main),atol=0,rtol=0)
    torch.testing.assert_close(image,torch.zeros_like(image),atol=0,rtol=0)
    with torch.no_grad():
        head.gains.bias.copy_(torch.tensor([.2,.3,.06,.08]))
        head.slopes.bias.copy_(torch.tensor([.1,-.7]))
        head.local_phase.weight.fill_(.01)
    main,image = head(tokens,pooled)
    for coefficients in (main,image):
        torch.testing.assert_close(coefficients.abs(),coefficients.abs()[:, :1].expand_as(coefficients))
        assert (coefficients[:,0]-coefficients[:,-1]).abs().min()>0
    assert not torch.allclose(main[:,0]/main[:,-1],image[:,0]/image[:,-1])
    (main.imag.square().mean()+image.imag.square().mean()).backward()
    for layer in (head.gains,head.slopes,head.local_phase):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0


def fixture():
    rng = torch.Generator().manual_seed(7)
    true_h = torch.randn(2,2,3,generator=rng)
    coeffs = torch.randn(2,4,9,generator=rng)
    est_h = torch.complex(*torch.randn(2,2,3,generator=rng).unbind(1))
    est_h = (est_h/est_h.norm(dim=1,keepdim=True)).detach().requires_grad_()
    main = torch.complex(*torch.randn(2,2,9,generator=rng).unbind(1)).requires_grad_()
    image = torch.complex(*torch.randn(2,2,9,generator=rng).unbind(1)).requires_grad_()
    noise = torch.tensor([.03,.5],requires_grad=True)
    return dict(kernel=est_h,main=main,image=image,variance=noise),true_h,coeffs,torch.tensor([.04,.2])


def test_direct_modulation_supervision_does_not_move_fir_gauge():
    aux,h,c,_ = fixture()
    target = torch.complex(*h.unbind(1)),torch.complex(c[:,0],c[:,1]),torch.complex(c[:,2],c[:,3])
    mod,fir = modulation_supervision(aux,*target)
    mod.sum().backward()
    assert aux['kernel'].grad is None
    assert aux['main'].grad.abs().sum()>0 and aux['image'].grad.abs().sum()>0
    fir.sum().backward()
    assert aux['kernel'].grad.abs().sum()>0


def test_operator_loss_matches_dense_and_is_invariant_to_factor_gauge():
    aux,h,c,noise = fixture()
    total,metrics = modulation_action_loss(aux,h,c,noise)
    target = dense_action(torch.complex(*h.unbind(1)).to(torch.complex128),
                          torch.complex(c[:,0],c[:,1]).to(torch.complex128),
                          torch.complex(c[:,2],c[:,3]).to(torch.complex128))
    a = dense_action(*(aux[k].to(torch.complex128) for k in ('kernel','main','image')))
    error = ((a-target).square().sum((1,2))/target.square().sum((1,2))).float()
    torch.testing.assert_close(metrics['action_matrix_relative_mse'],error)
    q = torch.tensor(-.5+1.3j,dtype=torch.complex64)
    shifted = dict(kernel=aux['kernel']*q,main=aux['main']/q,image=aux['image']/q.conj(),variance=aux['variance'])
    shifted_total,_ = modulation_action_loss(shifted,h,c,noise)
    torch.testing.assert_close(total,shifted_total)


def test_fixed_training_contract_and_actual_model_forward():
    cfg = candidate.configuration(candidate.ARM)
    assert cfg['epochs']==30 and cfg['updates']==7410 and cfg['training_stages']==1
    assert cfg['source_roles']=={'L_s':6300,'U_s':56700}
    assert cfg['training_posterior_calls']==cfg['training_F0_forward_rows']==0
    assert cfg['checkpoint_selection']=='final30_only' and not cfg['F0_guidance_for_G_updates']
    model = candidate.factory(candidate.ARM).eval()
    assert model.initialization_report['ancestor_checkpoints']==[]
    with torch.inference_mode():
        x = torch.randn(2,2,256,generator=torch.Generator().manual_seed(41))
        output,aux = model(x,return_aux=True)
    assert output.shape==x.shape and torch.isfinite(output).all()
    assert aux['relative_solve_residual'].max()<1e-8
