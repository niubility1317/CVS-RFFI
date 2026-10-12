"""Synthetic audit of label-blind moments; not imported by active training."""
import argparse,itertools,json
from pathlib import Path
import torch
from experiments.cvs_e20_moment_audit.fishr_stats import exact_variances,unbiased_variances,penalty


def interaction(means):
    """[TX,RX,coordinate] residual after TX and RX additive main effects."""
    if means.ndim!=3 or min(means.shape[:2])<2 or not bool(torch.isfinite(means).all()):raise ValueError('Finite TX/RX means required')
    return means-means.mean(1,keepdim=True)-means.mean(0,keepdim=True)+means.mean((0,1),keepdim=True)


def cross_interaction(a,b):
    if a.shape!=b.shape:raise ValueError('Matched TX/RX/coordinate grids required')
    return (interaction(a)*interaction(b)).mean()


def decompose(cells):
    """Exact balanced finite population [TX,RX,packet,coordinate]."""
    if cells.ndim!=4 or min(cells.shape)<1 or not bool(torch.isfinite(cells).all()):raise ValueError('Balanced finite cells required')
    means=cells.mean(2);grand=means.mean((0,1),keepdim=True)
    identity=means.mean(1,keepdim=True)-grand;receiver=means.mean(0,keepdim=True)-grand
    residual=interaction(means);within=cells-means[:,:,None,:]
    parts=dict(identity=identity.square().mean(),receiver=receiver.square().mean(),interaction=residual.square().mean(),within=within.square().mean())
    total=(cells-grand[:,:,None,:]).square().mean()
    return total,parts


def permutation_counterexample(device):
    templates=torch.eye(6,dtype=torch.float64,device=device)
    means=torch.stack([templates.roll(i,0) for i in range(5)],1)
    cells=means[:,:,None,:].expand(-1,-1,2,-1).clone()
    scores=cells.reshape(-1,6);y=torch.arange(6,device=device).repeat_interleave(10);rx=torch.arange(5,device=device).repeat_interleave(2).repeat(6)
    exact=exact_variances(scores,y,rx);estimated=unbiased_variances(scores,y,rx,{(r,k):70 for r in range(5) for k in range(6)})
    torch.testing.assert_close(exact,estimated,rtol=0,atol=1e-15)
    ordinary=penalty(estimated,estimated);semantic=cross_interaction(means,means)
    covariance=[]
    for r in range(5):
        x=means[:,r];z=x-x.mean(0);covariance.append(z.T@z/6)
    covariance=torch.stack(covariance)
    assert float(ordinary)<1e-28 and float(semantic)>.1
    assert float((covariance-covariance[:1]).abs().max())<1e-15
    return dict(mixed_variance_penalty=float(ordinary),class_receiver_interaction=float(semantic),full_covariance_max_domain_difference=float((covariance-covariance[:1]).abs().max()),same_complete_unconditional_distribution=True,interpretation='Counterexample only: conditional class correspondence differs while every label-blind distribution moment agrees.')


def exhaustive_cross_expectation(device):
    # Four finite cells; enumerate all 16 independent one-packet draws, then
    # every 256 A/B combination exactly. Repeated PIDs across A/B are allowed.
    base=torch.tensor([[[.2,1.],[1.2,2.]],[[2.2,3.],[3.2,4.]]],dtype=torch.float64,device=device)
    noise=torch.tensor([[-.7,.4],[.7,-.4]],dtype=torch.float64,device=device)
    outputs=[]
    for strength in (0.,.6):
        means=base.clone();means[0,0,0]+=strength
        pool=means[:,:,None,:]+noise
        draws=[torch.stack([pool[y,r,j] for (y,r),j in zip(itertools.product(range(2),repeat=2),choice)]).reshape(2,2,2) for choice in itertools.product(range(2),repeat=4)]
        values=torch.stack([cross_interaction(a,b) for a in draws for b in draws])
        exact=cross_interaction(means,means)
        torch.testing.assert_close(values.mean(),exact,rtol=1e-12,atol=1e-15)
        squared=torch.stack([cross_interaction(a,a) for a in draws]).mean()
        assert squared>exact and bool((values<0).any())
        outputs.append(dict(interaction_shift=strength,exact=float(exact),cross_expectation=float(values.mean()),same_batch_square_expectation=float(squared),negative_cross_draws=int((values<0).sum()),AB_combinations=len(values)))
    return outputs


def convolution_check(device):
    from experiments.cvs_e20_direction_audit.checks import Model
    from .multilevel import BoundarySensitivity,LEVELS
    from experiments.cvs_e20_boundary_style.checks_fishr import _state,_unchanged
    torch.manual_seed(987);seed=2026092701;model=Model(seed).to(device).double().eval()
    model.multilevel=BoundarySensitivity(model.base,seed,'ce')
    x=torch.randn(60,2,64,dtype=torch.float64,device=device);y=torch.arange(6,device=device).repeat_interleave(10)
    before=_state(model);_,raw,_=model.multilevel.raw_sensitivities(model,x,y)
    terms=[cross_interaction(g.reshape(6,5,2,-1)[:,:,0],g.reshape(6,5,2,-1)[:,:,1]) for g in raw.values()]
    # The two sample slots are synthetic mathematical inputs, not a claim
    # that this test matches the production private source sampler.
    loss=sum(terms);weight=model.base.id_backbone.cls_head.weight
    grad=torch.autograd.grad(loss,weight)[0];assert bool(torch.isfinite(grad).all()) and float(grad.norm())>0
    _unchanged(model,before)
    for g in raw.values():
        total,parts=decompose(g.reshape(6,5,2,-1));torch.testing.assert_close(total,sum(parts.values()),rtol=1e-10,atol=1e-15)
    return dict(loss=float(loss.detach()),gradient_norm=float(grad.norm()),levels=list(LEVELS),model_parameters_buffers_modes_RNG_gradients_unchanged=True)


def run(device):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.manual_seed(898);a=torch.randn(6,5,9,dtype=torch.float64,device=device)
    tx=torch.randn(6,1,9,dtype=torch.float64,device=device);rx=torch.randn(1,5,9,dtype=torch.float64,device=device)
    torch.testing.assert_close(interaction(a+tx+rx),interaction(a),rtol=1e-12,atol=1e-14)
    assert float(cross_interaction(tx+rx,tx+rx))<1e-28
    cells=torch.randn(6,5,7,9,dtype=torch.float64,device=device);total,parts=decompose(cells)
    torch.testing.assert_close(total,sum(parts.values()),rtol=1e-12,atol=1e-14)
    return dict(status='PASS',device=device,synthetic_only=True,source_IQ=False,target_access=False,model_updates=0,
        counterexample=permutation_counterexample(device),finite_population_expectation=exhaustive_cross_expectation(device),
        exact_decomposition=dict(total=float(total),**{k:float(v) for k,v in parts.items()}),additive_TX_RX_invariance=True,synthetic_convolution_CE_mixed_derivative=convolution_check(device),
        caveat='No actual-source severity, unknown-RX effect, optimization advantage or target performance has been measured by this audit.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
