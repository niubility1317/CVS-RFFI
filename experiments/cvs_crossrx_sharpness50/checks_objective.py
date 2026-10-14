"""Focused synthetic feature-transfer graph checks; no data/model training."""
import json
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .objective import objective, select_names


class Scalar(nn.Module):
    def __init__(self,value):super().__init__();self.weight=nn.Parameter(torch.tensor(value,dtype=torch.float64))


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module()
        self.id_backbone.encoder.core=nn.Module();self.id_backbone.encoder.core.id_backbone=nn.Module()
        core=self.id_backbone.encoder.core.id_backbone
        for name,value in (('sinc',.2),('time_fuse',-.1),('t1',.3),('t2',.15)):setattr(core,name,Scalar(value))
        self.id_backbone.encoder.sequence_pool=nn.Module();pool=self.id_backbone.encoder.sequence_pool
        pool.projection=Scalar(.25);pool.unused=nn.Parameter(torch.tensor(.4,dtype=torch.float64))
        pool.gain_raw=nn.Parameter(torch.tensor(.3,dtype=torch.float64))
        self.cls_head=nn.Module();self.cls_head.weight=nn.Parameter(torch.eye(6,160,dtype=torch.float64))
        self.cls_head.gain=nn.Parameter(torch.tensor(.7,dtype=torch.float64))
        self.other=Scalar(.12)
        self.register_buffer('forward_counter',torch.zeros((),dtype=torch.float64))
        self._filter_cache_key=('original_cache',);self._filter_cache=torch.tensor(7.)
    def forward(self,x):
        # Exercise RNG and mutable-buffer restoration without stochastic logits.
        dummy=(torch.rand(())+random.random()+float(np.random.random()))*0
        self.forward_counter.add_(1)
        core=self.id_backbone.encoder.core.id_backbone;pool=self.id_backbone.encoder.sequence_pool
        a=core.sinc.weight if self.training or self._filter_cache is None else self._filter_cache
        if not self.training:self._filter_cache=a*1.
        z=x.new_zeros((len(x),160))
        z[:,0]=a*x[:,0,0]+core.time_fuse.weight*x[:,1,0]+core.t1.weight*x[:,0,1]+pool.gain_raw*pool.projection.weight*x[:,0,2]
        z[:,1]=core.t2.weight*x[:,1,1]+self.other.weight*x[:,1,2]
        z[:,2:6]=x[:,0,3:7]
        return F.linear(z,self.cls_head.weight)*self.cls_head.gain+dummy


def _data():
    y=torch.arange(6).repeat(5);rx=torch.tensor([1,3,4,6,8]).repeat_interleave(6)
    r=torch.arange(5).repeat_interleave(6).double()
    x=torch.sin(torch.arange(30*2*256,dtype=torch.float64).reshape(30,2,256)/37)*.3+.2
    x[:,0,0]=.3*(r+1)+.02*y;x[:,1,0]=((r.remainder(2)*2)-1)*.4
    x[:,0,1]=.15*(y-2);x[:,1,1]=-.25+.1*r
    x[:,0,2]=r*.1+.2;x[:,1,2]=.6-.1*y
    current_ids=[(0,int(receiver),1,i,i) for i,receiver in enumerate(rx)]
    held_ids=[(0,int(receiver),1,100+i,100+i) for i,receiver in enumerate(rx)]
    return x,y,rx,current_ids,held_ids


def _raises(callback):
    try:callback()
    except (ValueError,RuntimeError):return
    raise AssertionError('Expected source/scope/physical-ID rejection')


def check():
    torch.set_num_threads(2)
    m=Fixture().train();m.other.eval()
    x,y,rx,current_ids,held_ids=_data();held=x*1.1
    current=m(x)
    names=select_names(m);named=dict(m.named_parameters());selected=[named[name] for name in names]
    assert len(names)==6 and not any('gain_raw' in name or 'cls_head' in name for name in names)
    before_params={name:(p.detach().clone(),p._version,p.grad) for name,p in named.items()}
    before_buffers={name:(b.clone(),b._version) for name,b in m.named_buffers()}
    before_flags={module:module.training for module in m.modules()}
    cache=m._filter_cache;cache_key=m._filter_cache_key
    py_rng=random.getstate();np_rng=np.random.get_state();rng=torch.get_rng_state().clone()
    args=(m,current,y,rx,held,y,rx,current_ids,held_ids,names,'cls_head.weight')
    results={mode:objective(*args,mode) for mode in ('CROSS_GAP','GLOBAL_GAP','CLEAN_CE','ZERO')}
    for mode,result in results.items():
        summary=result['summary']
        assert result['loss'] is result['weighted_loss']
        assert summary['functional_forwards']==6 and summary['extra_IQ_forwards']==150
        assert summary['selected_count']==6 and summary['rho']==.05 and summary['coefficient']==.1
        assert summary['loss']==result['weighted_loss'].detach().item()
        assert all(not delta.requires_grad and delta.grad_fn is None for values in result['tensors']['deltas_by_RX'].values() for delta in values)
        assert all(row['inner_none_gradient_count']==1 for row in summary['by_RX'].values())
        assert all(abs(row['actual_perturb_norm']-.05)<1e-14 for row in summary['by_RX'].values())
        json.dumps(summary,allow_nan=False)
        all_grads=torch.autograd.grad(result['weighted_loss'],list(named.values()),retain_graph=True,allow_unused=True)
        assert all(g is None for (name,p),g in zip(named.items(),all_grads) if name not in names)
    cross=results['CROSS_GAP'];global_result=results['GLOBAL_GAP']
    assert any(row['gap']>1e-8 for row in cross['summary']['by_RX'].values())
    clean=cross['tensors']['clean_CE'];pert=cross['tensors']['perturbed_CE_by_RX']
    independent=.1/5*sum((pert[str(receiver)].mean()-clean[rx!=receiver].mean()).relu() for receiver in (1,3,4,6,8))
    assert torch.allclose(cross['weighted_loss'],independent,atol=1e-15,rtol=0)
    actual=torch.autograd.grad(cross['weighted_loss'],selected,retain_graph=True,allow_unused=True)
    expected=[torch.zeros_like(p) for p in selected];pert_only=[torch.zeros_like(p) for p in selected]
    # An independent first-derivative assembly proves the clean gradient was
    # subtracted, rather than merely checking the scalar formula.
    for receiver in (1,3,4,6,8):
        if cross['summary']['by_RX'][str(receiver)]['gap']<=0:continue
        gp=torch.autograd.grad(pert[str(receiver)].mean(),selected,retain_graph=True,allow_unused=True)
        gc=torch.autograd.grad(clean[rx!=receiver].mean(),selected,retain_graph=True,allow_unused=True)
        for i,(pgrad,cgrad) in enumerate(zip(gp,gc)):
            if pgrad is not None:expected[i]+=pgrad*.02;pert_only[i]+=pgrad*.02
            if cgrad is not None:expected[i]-=cgrad*.02
    for g,e in zip(actual,expected):
        assert (g is None and torch.count_nonzero(e)==0) or torch.allclose(g,e,atol=1e-15,rtol=1e-12)
    gradient_norm=sum(float(g.square().sum()) for g in actual if g is not None)**.5
    clean_gradient_effect=sum(float((e-p).square().sum()) for e,p in zip(expected,pert_only))**.5
    assert gradient_norm>1e-8 and clean_gradient_effect>1e-8
    global_delta=global_result['tensors']['deltas_by_RX']
    assert all(all(torch.equal(a,b) for a,b in zip(global_delta['1'],values)) for values in global_delta.values())
    assert torch.allclose(results['CLEAN_CE']['weighted_loss'],.02*sum(results['CLEAN_CE']['tensors']['clean_CE'][rx!=r].mean() for r in (1,3,4,6,8)),atol=1e-15,rtol=0)
    zero_grads=torch.autograd.grad(results['ZERO']['weighted_loss'],list(named.values()),retain_graph=True,allow_unused=True)
    assert results['ZERO']['weighted_loss'].item()==0 and all(g is None or torch.count_nonzero(g)==0 for g in zero_grads)
    # Retained current main graph still updates the ordinary head jointly.
    main_grad=torch.autograd.grad(F.cross_entropy(current,y),[m.cls_head.weight,m.cls_head.gain],retain_graph=True)
    assert all(g.norm()>0 for g in main_grad)
    for name,p in named.items():
        value,version,grad=before_params[name]
        assert torch.equal(value,p.detach()) and version==p._version and p.grad is grad
    for name,b in m.named_buffers():assert torch.equal(before_buffers[name][0],b) and before_buffers[name][1]==b._version
    assert all(module.training==flag for module,flag in before_flags.items())
    assert m._filter_cache is cache and m._filter_cache_key is cache_key
    assert random.getstate()==py_rng and torch.equal(rng,torch.get_rng_state())
    now=np.random.get_state();assert now[0]==np_rng[0] and np.array_equal(now[1],np_rng[1]) and now[2:]==np_rng[2:]
    # Truly zero current feature gradients yield zero perturbations, while all
    # budgets remain identical; no hidden active-only renormalization.
    zmodel=Fixture().train();zero_current=zmodel(torch.zeros_like(x))
    zero_inner=objective(zmodel,zero_current,y,rx,held,y,rx,current_ids,held_ids,select_names(zmodel),'cls_head.weight','CROSS_GAP')
    assert all(row['actual_perturb_norm']==0 for row in zero_inner['summary']['by_RX'].values())
    assert zero_inner['weighted_loss'].item()==0 and zero_inner['summary']['extra_IQ_forwards']==150
    for role in ('U_s','V','target','query'):_raises(lambda role=role:objective(*args,'CROSS_GAP',role=role))
    _raises(lambda:objective(*args,'INVALID'))
    _raises(lambda:objective(m,current,y,rx,held,y,rx,current_ids,current_ids,names,'cls_head.weight','CROSS_GAP'))
    duplicate=list(held_ids);duplicate[1]=duplicate[0]
    _raises(lambda:objective(m,current,y,rx,held,y,rx,current_ids,duplicate,names,'cls_head.weight','CROSS_GAP'))
    _raises(lambda:objective(m,current,y,rx,held,y,rx,current_ids,held_ids,names[:-1],'cls_head.weight','CROSS_GAP'))
    _raises(lambda:objective(m,current,y,rx,held,y,rx,current_ids,held_ids,names+['cls_head.weight'],'cls_head.weight','CROSS_GAP'))
    absent_rx=rx.clone();absent_rx[rx==8]=6
    absent_ids=[(0,int(r),1,i,i) for i,r in enumerate(absent_rx)]
    _raises(lambda:objective(m,current,y,absent_rx,held,y,rx,absent_ids,held_ids,names,'cls_head.weight','CROSS_GAP'))
    hidden=y.clone();hidden[0]=-1
    _raises(lambda:objective(m,current,hidden,rx,held,y,rx,current_ids,held_ids,names,'cls_head.weight','CROSS_GAP'))
    _raises(lambda:objective(m,current,y.float(),rx,held,y,rx,current_ids,held_ids,names,'cls_head.weight','CROSS_GAP'))
    return dict(status='VERIFIED',torch=torch.__version__,device='cpu',dtype='float64',
        fixed_scope_and_radius=True,all_modes_equal_forward_budget=True,
        clean_gradient_retained=True,delta_detached_no_second_order=True,
        outer_only_selected_W_gain_other_None=True,ZERO_selected_connected_exact_zero=True,
        zero_inner_gradient_zero_delta=True,current_main_graph_retained=True,
        live_parameters_buffers_gradients_flags_caches_RNG_restored=True,
        source_scope_labels_IDs_RX_negatives=True,cross_loss=float(cross['weighted_loss'].detach()),
        cross_outer_gradient_norm=gradient_norm,clean_gradient_effect_norm=clean_gradient_effect,
        source_IQ_reads=0,head_fit_updates=0,optimizer_updates=0,
        boundary='Synthetic feature-transfer loss only; no source efficacy or real-training claim')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
