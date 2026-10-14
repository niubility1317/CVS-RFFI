"""Synthetic four-evidence PA exchange and gradient-path checks, no data fit."""
import copy
import json
import random

import torch
from torch import nn
from torch.nn import functional as F

from .objective import objective
from experiments.cvs_residual_identity.model import ResidualPhysicalHead


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallow=nn.Linear(8,160);self.PA=nn.Linear(8,160)
        self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module();encoder=self.id_backbone.encoder
        encoder.core=nn.Module();encoder.core.id_backbone=nn.Module()
        encoder.core.id_backbone.cls_head=ResidualPhysicalHead(160)
        encoder.response_gain=nn.Parameter(torch.full((160,),.08))
        encoder.response_projection=nn.Sequential(nn.Linear(8,160),nn.LayerNorm(160))
        encoder.sequence_pool=nn.Module();encoder.sequence_pool.output_projection=nn.Linear(8,160)
        encoder.sequence_pool.output_norm=nn.LayerNorm(160)
        encoder.sequence_pool.gain_raw=nn.Parameter(torch.full((160,),.12))
    def components(self,x):
        encoder=self.id_backbone.encoder;head=encoder.core.id_backbone.cls_head;pool=encoder.sequence_pool
        b=head.base_norm(self.shallow(x));p=head.pa_norm(self.PA(x))
        r=encoder.response_projection(x);s=pool.output_norm(pool.output_projection(x))
        joint=((b+head.gain.tanh()*p)+encoder.response_gain.tanh()*r)+(.2*pool.gain_raw.tanh())*s
        return dict(identity=b,physical=p,reference_raw=r,sequence_raw=s,joint=joint,logits=head.classify(joint))


def _raises(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected malformed/illegal source input rejection')


def check():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261015);model=Fixture().double()
    x=torch.sin(torch.arange(30*8,dtype=torch.float64).reshape(30,8)/7)
    labels=torch.arange(6).repeat_interleave(5)
    rxs=(1,3,4,6,8);ids=[(0,rxs[i%5],1,i,100+i) for i in range(30)]
    parts=model.components(x);encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
    before=copy.deepcopy(model.state_dict());count=sum(p.numel() for p in model.parameters())
    head_identity=id(head);weight_identity=id(head.weight)
    rng=torch.get_rng_state().clone();py_rng=random.getstate()
    results={mode:objective(model,parts,labels,ids,mode) for mode in ('ZERO','SELF','SWAP_PA')}
    partner=torch.tensor([5*(i//5)+(i+1)%5 for i in range(30)])
    grads_by_mode={}
    named=list(model.named_parameters())
    for mode,result in results.items():
        summary=result['summary'];assert result['loss'] is result['weighted_loss']
        assert torch.equal(result['partner_index'],partner)
        assert summary['eligible_count']==30 and summary['pair_count']==30 and summary['coverage']==1.
        assert summary['head_readout_calls']==2 and summary['head_readout_packets']==60
        assert summary['self_reconstruction_max_abs_error']==0 and summary['self_reconstruction_exact']
        assert summary['scale']==head.scale==30. and summary['normalize_eps']==1e-4 and summary['coefficient']==.1
        json.dumps(summary,allow_nan=False)
        gradients=torch.autograd.grad(result['loss'],list(model.parameters()),retain_graph=True,allow_unused=True)
        grads_by_mode[mode]=gradients
        assert all(g is None for (name,_),g in zip(named,gradients)
                   if name.endswith('cls_head.weight') or name.endswith('cls_head.gain')
                   or name.endswith('response_gain') or name.endswith('sequence_pool.gain_raw'))
    direct_self=.1*F.cross_entropy(head.classify(parts['joint']),labels)
    assert torch.allclose(results['SELF']['loss'].detach(),direct_self.detach(),atol=1e-15,rtol=1e-15)
    swap=((parts['identity']+head.gain.detach().tanh()*parts['physical'][partner])
           +encoder.response_gain.detach().tanh()*parts['reference_raw'])
    swap=swap+(.2*encoder.sequence_pool.gain_raw.detach().tanh())*parts['sequence_raw']
    direct_swap=.1*F.cross_entropy(30.*F.linear(F.normalize(swap,dim=1,eps=1e-4),
                                                F.normalize(head.weight.detach(),dim=1,eps=1e-4)),labels)
    assert torch.allclose(results['SWAP_PA']['loss'].detach(),direct_swap.detach(),atol=1e-15,rtol=1e-15)
    assert results['SWAP_PA']['summary']['swap_changed_packets']==30
    group_norms={}
    for mode in ('SELF','SWAP_PA'):
        norms={}
        for group,prefix in (('shallow','shallow.'),('PA','PA.'),('sequence','id_backbone.encoder.sequence_pool.output_'),
                             ('reference','id_backbone.encoder.response_projection.')):
            relevant=[g for (name,_),g in zip(named,grads_by_mode[mode]) if name.startswith(prefix) and g is not None]
            norms[group]=sum(float(g.detach().square().sum()) for g in relevant)**.5
            assert norms[group]>1e-8 and all(bool(torch.isfinite(g).all()) for g in relevant)
        group_norms[mode]=norms
    assert results['ZERO']['loss'].item()==0 and all(g is None or torch.count_nonzero(g)==0 for g in grads_by_mode['ZERO'])
    main_grads=torch.autograd.grad(F.cross_entropy(parts['logits'],labels),
        [head.weight,head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw],retain_graph=True)
    assert all(bool(torch.isfinite(g).all()) and g.norm()>0 for g in main_grads)
    assert sum(p.numel() for p in model.parameters())==count and id(head)==head_identity and id(head.weight)==weight_identity
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
    assert all(p.grad is None for p in model.parameters()) and torch.equal(rng,torch.get_rng_state()) and py_rng==random.getstate()
    # Reference contribution is nonzero; omitting it cannot pass SELF equality.
    no_reference=dict(parts,reference_raw=torch.zeros_like(parts['reference_raw']))
    _raises(lambda:objective(model,no_reference,labels,ids,'SELF'))
    inconsistent=dict(parts,joint=parts['joint']+.001)
    _raises(lambda:objective(model,inconsistent,labels,ids,'SELF'))
    # PA values identical across each RX cycle leave the deployed value unchanged.
    identical=dict(parts);identical['physical']=parts['physical'].reshape(6,5,160)[:,0:1].expand(-1,5,-1).reshape(30,160)
    identical['joint']=((identical['identity']+head.gain.tanh()*identical['physical'])
        +encoder.response_gain.tanh()*identical['reference_raw'])+(.2*encoder.sequence_pool.gain_raw.tanh())*identical['sequence_raw']
    same_self=objective(model,identical,labels,ids,'SELF');same_swap=objective(model,identical,labels,ids,'SWAP_PA')
    assert torch.equal(same_self['loss'],same_swap['loss']) and same_swap['summary']['swap_changed_packets']==0
    # Any incomplete cell skips the entire cohort, preserving the fixed plan.
    incomplete=list(ids);incomplete[-1]=(1,8,1,29,129)
    skipped=objective(model,parts,labels,incomplete,'SWAP_PA')
    assert skipped['loss'].item()==0 and skipped['summary']['eligible_count']==0
    assert skipped['summary']['coverage']==0 and (skipped['partner_index']==-1).all()
    assert skipped['summary']['head_readout_calls']==2
    duplicate=list(ids);duplicate[-1]=duplicate[0]
    _raises(lambda:objective(model,parts,labels,duplicate,'SELF'))
    wrong_rx=list(ids);wrong_rx[0]=(0,2,1,0,100)
    _raises(lambda:objective(model,parts,labels,wrong_rx,'SELF'))
    wrong_day=list(ids);wrong_day[0]=(0,1,4,0,100)
    _raises(lambda:objective(model,parts,labels,wrong_day,'SELF'))
    native_int=list(ids);native_int[0]=(0,1,1,0,True)
    _raises(lambda:objective(model,parts,labels,native_int,'SELF'))
    hidden=labels.clone();hidden[0]=-1
    _raises(lambda:objective(model,parts,hidden,ids,'SELF'))
    _raises(lambda:objective(model,parts,labels.float(),ids,'SELF'))
    for role in ('U_s','V','query','target'):_raises(lambda role=role:objective(model,parts,labels,ids,'SELF',role=role))
    _raises(lambda:objective(model,parts,labels,ids,'OTHER'))
    nonfinite=dict(parts);nonfinite['physical']=parts['physical'].clone();nonfinite['physical'][0,0]=float('inf')
    _raises(lambda:objective(model,nonfinite,labels,ids,'SELF'))
    badshape=dict(parts,sequence_raw=parts['sequence_raw'][:,:159])
    _raises(lambda:objective(model,badshape,labels,ids,'SELF'))
    return dict(status='PASS',torch=torch.__version__,device='cpu',dtype='float64',
        four_term_SELF_exact=True,fixed_RX_cycle_bijective=True,PA_swap_changed_packets=30,
        W_PA_SIG_reference_gains_aux_gradients_None=True,feature_gradient_norms=group_norms,
        main_W_all_gains_joint_gradient_nonzero=True,ZERO_exact_zero=True,complete_cohort_skip=True,
        matched_two_head_readouts=True,parameters_state_RNG_unchanged=True,protocol_negatives=True,
        data_reads=0,head_fit_updates=0,optimizer_updates=0,
        boundary='Synthetic feature recomposition only; no training efficacy or cross-RX generalization claim')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
