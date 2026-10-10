"""Focused CPU/GPU checks for physical actions and polynomial proposal paths."""
import json
import math
import torch
from torch import nn
from .physics import (factorial_batch, apply_action, apply_temporal, phase_process,
                      sample_parameters, quality_proxies, cfo_additive_diagnostic)
from .actions import (StateAction, SharedActionCore, parameter_basis, polynomial_terms,
                      signal_state, estimate_pair, conditional_edges, chain_metrics,
                      proposal_select, verify_proposal, action_loss)


class _Head(nn.Module):
    def components(self,base,physical):
        return base, physical, torch.tanh(base+physical)


class _Identity(nn.Module):
    """Small nonlinear E/G/C fixture; end-to-end native smoke lives in checks.py."""
    def __init__(self):
        super().__init__()
        self.core=nn.Module(); self.core.id_backbone=nn.Module()
        self.core.id_backbone.cls_head=_Head()
        self.response_projection=nn.Linear(29,160)
        self.response_gain=nn.Parameter(torch.full((160,),.1))
        self.classifier=nn.Linear(160,6)
    def response(self,x):
        return torch.cat((x.mean(2),x.square().mean(2),x.flatten(1)[:,:25]),1)
    def classify_features(self,z):
        return self.classifier(z)
    def intermediate(self,x):
        f=x.flatten(1)
        return torch.cat((f[:,:160].tanh(),f[:,160:320].square(),self.response(x)),1)


def run_checks(device='cpu'):
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(614)
    x=torch.randn(8,2,256,generator=g).to(device)
    x=x/x.square().sum(1).mean(1).sqrt()[:,None,None]
    identity=_Identity().to(device).eval()
    for p in identity.parameters(): p.requires_grad_(False)
    models={k:StateAction(k).to(device) for k in ('linear','temporal')}
    state=torch.random.get_rng_state().clone()
    cuda_state=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
    a=factorial_batch(x,g,step=0); b=factorial_batch(x,g,step=1)
    assert a['order']=='LT' and b['order']=='TL'
    assert torch.equal(state,torch.random.get_rng_state())
    if cuda_state is not None: assert torch.equal(cuda_state,torch.cuda.get_rng_state())
    assert (a['temporal_parameters'][:,3]==0).all()
    assert torch.equal(apply_temporal(x,torch.zeros(8,4,device=device)),x)
    failed=False
    try: apply_temporal(x,a['temporal_parameters'])
    except ValueError: failed=True
    assert failed
    for data in (a,b):
        h={k:identity.intermediate(data[k]).detach() for k in ('x00','x10','x01','x11')}
        edges=conditional_edges(data,h)
        assert len(edges)==2 and sum(e['conditional'] for e in edges)==1
        for edge in edges:
            p=models[edge['kind']](edge['h'],edge['x'],edge['p'],None,
                phase_noise=edge['phase_noise'],cached_reference=edge['cached_reference'],
                cached_endpoint_reference=edge['cached_endpoint_reference'])
            assert torch.equal(p[:,320:],edge['target_delta'][:,320:])
            loss=action_loss(identity,edge['h'],edge['target_delta'],p,torch.arange(8,device=device)%6)
            loss['loss'].backward()
            assert any(v.grad is not None and torch.isfinite(v.grad).all() for v in models[edge['kind']].parameters())
            unlabelled=action_loss(identity,edge['h'],edge['target_delta'],p,None,label_free=True)
            assert torch.isfinite(unlabelled['loss'])
        metrics=chain_metrics(models,data,h,identity.response)
        assert math.isfinite(metrics['predicted_chain_mse'])
    # True pair physical parameter recovery, including the known random process.
    for kind in models:
        p=a[kind+'_parameters']; endpoint=apply_action(x,p,kind,a['phase_noise'])
        estimate,info=estimate_pair(x,endpoint,kind,a['phase_noise'])
        scales=x.new_tensor([1,1,1,1] if kind=='linear' else [1,3120,.025,7.34e8])
        assert float(((estimate-p)/scales).abs().max())<2e-4
    pt=a['temporal_parameters']; pperiod=pt.clone();pperiod[:,0]+=2*math.pi
    assert torch.allclose(parameter_basis(pt,'temporal',a['phase_noise']),
                          parameter_basis(pperiod,'temporal',a['phase_noise']),atol=1e-6)
    # Even polynomial terms survive; no odd-only decoder regression.
    q=torch.randn(8,7,generator=g).to(device)
    assert torch.allclose(polynomial_terms(q)[:,7:],polynomial_terms(-q)[:,7:])
    assert signal_state(x).shape==(8,40)
    assert bool((quality_proxies(x).std(0)>1e-5).all())
    assert torch.allclose(quality_proxies(x),quality_proxies(x*3),atol=1e-6)
    # Shared/independent act identically when weights are matched; same heads.
    shared=SharedActionCore().to(device)
    tied=StateAction('temporal',shared_core=shared).to(device)
    tied.load_state_dict(models['temporal'].state_dict())
    h=identity.intermediate(x)
    assert torch.equal(tied(h,x,pt,identity.response,phase_noise=a['phase_noise']),
                       models['temporal'](h,x,pt,identity.response,phase_noise=a['phase_noise']))
    y=torch.arange(8,device=device)%6
    from experiments.cvs_multi_disentangle.model import classify_intermediate
    for kind in models:
        proposal=proposal_select(models[kind],identity,h,x,y,g,candidates=3,reliability=0.)
        assert not bool(proposal['learned_mask'].any())
        logits=classify_intermediate(identity,identity.intermediate(proposal['x']))
        verification=verify_proposal(proposal,logits,y)
        assert verification['random_used']==8 and 0<verification['reliability_next']<=1
        assert not proposal['parameters'].requires_grad
    report=dict(status='PASS',device=device,rng_preserved=True,physical_units=True,
                conditional_edge_budget=2,periodic_phase=True,exact29_cache=True,
                stochastic_replay=True,chain_both_orders=True,shared_identical_form=True,
                random_fallback_retains_real_IQ=True,variable_quality=True,
                parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()})
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    print(json.dumps(run_checks(args.device),indent=2))
