"""Synthetic mathematical check; not a new training objective or candidate.

For main-identity m=z_y-logsumexp(z_other), CE=softplus(-m), so the
unreduced per-example virtual-coordinate gradient is -(1-p_y)*grad(m).
Per-example RMS normalization largely cancels that confidence factor; the
registered epsilon makes this approximate rather than exact near zero.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_e20_boundary_style.multilevel import identity_margin,normalize_scores
from . import design as d


def check():
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(620331)
    h=torch.randn(30,7,dtype=torch.float64,generator=generator)
    theta=torch.randn(7,6,dtype=torch.float64,generator=generator,requires_grad=True)
    a=torch.ones_like(h,requires_grad=True);b=torch.zeros_like(h,requires_grad=True)
    y=torch.arange(30)%6;logits=(h*a+b)@theta
    margin=identity_margin(logits,y);ce=F.cross_entropy(logits,y,reduction='none')
    probability=logits.softmax(1).gather(1,y[:,None]).squeeze(1);confidence_weight=1-probability
    torch.testing.assert_close(ce,F.softplus(-margin),atol=1e-12,rtol=1e-12)
    def deriv(value):
        da,db=torch.autograd.grad(value.sum(),(a,b),create_graph=True,retain_graph=True)
        return torch.cat((da,db),1)
    gm=deriv(margin);gc=deriv(ce)
    expected=-confidence_weight[:,None]*gm
    torch.testing.assert_close(gc,expected,atol=2e-12,rtol=2e-11)
    # Preserve the derivative of confidence_weight: dropping it would change
    # the mixed second derivative used to train the identity network.
    s1=torch.autograd.grad(gc.square().mean(),theta,retain_graph=True)[0]
    s2=torch.autograd.grad(expected.square().mean(),theta,retain_graph=True)[0]
    torch.testing.assert_close(s1,s2,atol=2e-12,rtol=2e-11)
    sm,_=normalize_scores(gm);sc,_=normalize_scores(gc)
    algebra=-gm/(gm.square().mean(1,keepdim=True)+(1e-6/confidence_weight[:,None]).square()).sqrt()
    torch.testing.assert_close(sc,algebra,atol=2e-11,rtol=2e-10)
    equal_epsilon_free=torch.max(torch.abs(gc/gc.square().mean(1,keepdim=True).sqrt()+gm/gm.square().mean(1,keepdim=True).sqrt())).detach()
    assert float(equal_epsilon_free)<1e-10
    # Counterexample to an unconditional equivalence claim at the actual eps:
    # highly confident CE can be below epsilon while margin remains well scaled.
    raw=torch.ones(1,14,dtype=torch.float64)
    unit,_=normalize_scores(raw);tiny,_=normalize_scores(-raw*1e-10)
    assert float((unit+tiny).abs().max())>.9
    detached=torch.autograd.grad((-confidence_weight.detach()[:,None]*gm).square().mean(),theta)[0]
    assert float((s1-detached).norm())>1e-8
    return dict(status='PASS',synthetic_only=True,source_data_access=False,target_access=False,persistent_identity_updates=0,
        CE_softplus_margin_identity=True,raw_gradient_identity=True,mixed_second_derivative_identity=True,
        actual_epsilon_normalization_formula=True,epsilon_free_sign_equivalence_error=float(equal_epsilon_free),
        ordinary_batch_normalized_sign_difference=float((sc+sm).detach().abs().max()),
        confidence_gradient_magnitude_factor_range=[float(confidence_weight.detach().min()),float(confidence_weight.detach().max())],
        near_zero_epsilon_counterexample_max_difference=float((unit+tiny).abs().max()),
        confidence_detach_second_derivative_error=float((s1-detached).norm()),
        interpretation='CE-normalized and margin-normalized scores have opposite signs away from epsilon, hence same variance; near-zero epsilon breaks equivalence. Raw CE retains live confidence scaling and its second derivative. No performance claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=check();d.write(a.output,v);print(json.dumps(v))
