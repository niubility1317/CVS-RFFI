"""Bounded analytic-gradient and actual identity-path checks; no target access."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .multilevel import MultiLevel, LEVELS, linear_sample_gradients, fishr_variance, weak_mixstyle


def norm(loss, parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return sum(float(v.detach().double().square().sum()) for v in values if v is not None)**.5


def run(device='cpu'):
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(531)
    f=torch.randn(12,9,generator=generator,dtype=torch.float64).to(device).requires_grad_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(667);head=nn.Linear(9,6).to(device=device,dtype=torch.float64)
    labels=torch.arange(12,device=device)%6
    analytic=linear_sample_gradients(f,head,labels)
    losses=F.cross_entropy(head(f),labels,reduction='none')
    actual=[]
    for loss in losses:
        w,b=torch.autograd.grad(loss,(head.weight,head.bias),retain_graph=True,create_graph=True)
        actual.append(torch.cat((w,b[:,None]),1))
    actual=torch.stack(actual)
    torch.testing.assert_close(analytic,actual,rtol=1e-10,atol=1e-12)
    fishr,record=fishr_variance(f,head,labels,torch.tensor([1]*6+[3]*6,device=device))
    v=[g.var(0,unbiased=False) for g in (analytic[:6],analytic[6:])]
    torch.testing.assert_close(fishr,(v[0]-v[1]).square().mean()/4)
    assert norm(fishr,(f,))>0
    # Explicit detach test: output Jacobian has no donor-statistics path.
    fmap=torch.randn(12,9,32,generator=generator).to(device).requires_grad_()
    rx=torch.tensor([1]*6+[3]*6,device=device);day=torch.zeros(12,dtype=torch.long,device=device)
    mixed,style=weak_mixstyle(fmap,labels,rx,day,list(range(12)),123,p=1)
    assert style['changed_packets']==12 and style['movement_max']<=.2
    derivative=torch.autograd.grad(mixed[0].sum(),fmap)[0]
    assert derivative[0].norm()>0 and derivative[1:].abs().sum()==0
    for i,j in enumerate(style['donors']):
        assert labels[i]==labels[j] and rx[i]!=rx[j] and day[i]==day[j] and i!=j

    from . import design as d
    from .model import build
    c=d.config(next(row for row in d.rows() if row['arm']=='PHYS'))
    model=build(c,device)
    rng=torch.random.get_rng_state().clone()
    auxiliary=MultiLevel(model.base,seed=38771).to(device)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert set(auxiliary._modules)=={'heads'} and all(n.startswith('heads.') for n,_ in auxiliary.named_parameters())
    source=dict(x=torch.randn(180,2,256,generator=generator),
                y=torch.arange(6).repeat_interleave(10).repeat(3),
                rx=torch.tensor([1,3,4,6,8]).repeat_interleave(2).repeat(18),
                day=torch.arange(3).repeat_interleave(60),ids=list(range(180)))
    config=dict(enabled=True,mixstyle=False,fishr=True,interval=4,batch_size=60,levels=list(LEVELS),
                mixstyle_p=1.,mixstyle_alpha=.1,mixstyle_strength=.3)
    snapshot={k:v.detach().clone() for k,v in model.state_dict().items()}
    allhooks=sum(len(m._forward_hooks) for m in model.modules())
    model.train(); auxiliary.train()
    native_rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None
    ce,penalty,detail=auxiliary.loss(model,source,4,device,config)
    assert torch.equal(native_rng,torch.random.get_rng_state())
    if cuda_rng is not None:
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert penalty>0 and detail['aux_forwards']==1 and detail['fishr_steps']==1
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    groups={level:tuple(p for n,p in params.items() if match in n) for level,match in
            [('shallow','time_fuse.'),('middle','t2.') ]}
    scales={level:dict(ce=norm(ce,parameters),fishr=norm(penalty,parameters)) for level,parameters in groups.items()}
    assert all(v['ce']>0 and v['fishr']>0 for v in scales.values())
    for value in scales.values():
        value['fishr_to_auxCE_gradient_ratio_at_equal_weights']=value['fishr']/value['ce']
    # Clean statistics are identical in the joint arm, despite the mixed CE path.
    c2=dict(config,mixstyle=True)
    both_ce,both_fishr,both_detail=auxiliary.loss(model,source,4,device,c2)
    assert torch.equal(native_rng,torch.random.get_rng_state())
    if cuda_rng is not None:
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    torch.testing.assert_close(penalty,both_fishr)
    assert both_detail['aux_forwards']==2
    assert all(v['style']['eligible_packets']==60 and v['style']['changed_packets']>0
               for v in both_detail['levels'].values())
    assert norm(both_ce,groups['shallow'])>0 and norm(both_ce,groups['middle'])>0
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items())
    assert sum(len(m._forward_hooks) for m in model.modules())==allhooks
    # Every fourth step only, same day, exact balanced counts, private RNG.
    sample_rng=torch.random.get_rng_state().clone()
    for step in (4,8,12,16):
        index=auxiliary.balanced_indices(source,step)
        assert len(set(source['ids'][i] for i in index.tolist()))==60
        assert source['day'][index].unique().numel()==1
        assert torch.bincount(source['y'][index],minlength=6).tolist()==[10]*6
        assert [int((source['rx'][index]==r).sum()) for r in (1,3,4,6,8)]==[12]*5
    assert torch.equal(sample_rng,torch.random.get_rng_state())
    empty=auxiliary.loss(model,source,5,device,config)
    assert empty[2]['aux_forwards']==0
    # Ordinary inference is bitwise unchanged and eval cannot apply style.
    model.eval();x=source['x'][:4].to(device)
    with torch.no_grad():
        baseline=model(x)
        with auxiliary.capture(mix=True,labels=source['y'][:4].to(device),receivers=source['rx'][:4].to(device),
                               days=source['day'][:4].to(device),ids=list(range(4)),config=c2) as (_,s):
            evaluated=model(x)
        assert all(not value['enabled'] for value in s.values())
        torch.testing.assert_close(baseline,evaluated,rtol=0,atol=0)
    assert auxiliary.loss(model,source,4,device,c2)[2]['aux_forwards']==0
    try:
        with auxiliary.capture():
            raise RuntimeError('intentional context exception')
    except RuntimeError:
        pass
    assert not auxiliary._active and sum(len(m._forward_hooks) for m in model.modules())==allhooks
    return dict(status='PASS',device=device,source_target_access=False,
                analytic_weight_and_bias_autograd_max_error=float((analytic-actual).abs().max().detach()),
                real_layer_paths=auxiliary.paths,actual_feature_shapes={k:v['shape'] for k,v in detail['levels'].items()},
                raw_aux_ce=float(ce.detach()),raw_fishr=float(penalty.detach()),gradient_scales=scales,
                real_clean_Fishr_same_in_joint=True,detached_donor_statistics=True,
                exact_6TX_5RX_2PID_same_day=True,private_sampler_rng=True,
                native_CPU_CUDA_RNG_preserved=True,
                no_EMA=True,no_inference_change=True,no_eval_style=True,context_exception_cleanup=True,
                auxiliary_parameters=sum(p.numel() for p in auxiliary.parameters()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    args=p.parse_args();result=run(args.device)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
