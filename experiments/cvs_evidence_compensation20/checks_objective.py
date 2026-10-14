"""Closed-form masks/stops plus actual-constructor synthetic gradient isolation."""
import copy
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .objective import readouts, objective
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


class Fixture(nn.Module):
    def __init__(self):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module()
        encoder=self.id_backbone.encoder;encoder.core=nn.Module();encoder.core.id_backbone=nn.Module()
        encoder.core.id_backbone.cls_head=ResidualPhysicalHead(160)
        encoder.response_gain=nn.Parameter(torch.full((160,),.1));encoder.sequence_pool=nn.Module()
        encoder.sequence_pool.gain_raw=nn.Parameter(torch.full((160,),.12))
        with torch.no_grad():encoder.core.id_backbone.cls_head.weight.copy_(torch.eye(6,160))


def _parts(model, *, strong=True):
    encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
    a=torch.zeros(4,160,dtype=head.weight.dtype);p=torch.zeros_like(a)
    if strong:a[:,0]=1.
    else:a[:,1]=1.;p[:,0]=6.
    a.requires_grad_();p.requires_grad_()
    r=torch.zeros_like(a,requires_grad=True);s=torch.zeros_like(a,requires_grad=True)
    joint=((a+head.gain.tanh()*p)+encoder.response_gain.tanh()*r)+(.2*encoder.sequence_pool.gain_raw.tanh())*s
    return dict(identity=a,physical=p,reference_raw=r,sequence_raw=s,joint=joint,logits=head.classify(joint))


def _raises(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected metadata/role/component rejection')


def check():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(20261016);model=Fixture().double()
    ids=[(0,rx,1,i,100+i) for i,rx in enumerate((1,3,4,6))]
    targets=torch.zeros(4,dtype=torch.long);all_q=torch.ones(4,dtype=torch.bool)
    high=_parts(model);high_result=objective(model,high,targets,all_q,ids,'EVIDENCE_STOP')
    assert high_result['loss'].item()==0 and high_result['summary']['active_packets']==0
    high_grads=torch.autograd.grad(high_result['loss'],[high['identity'],high['sequence_raw'],high['physical'],high['reference_raw']],
                                   retain_graph=True,allow_unused=True)
    assert all(g is not None and torch.count_nonzero(g)==0 for g in high_grads[:2]) and high_grads[2:]==(None,None)
    active=_parts(model,strong=False);views=readouts(model,active)
    assert torch.equal(views['self']['joint'].detach(),active['joint'].detach())
    assert (views['self']['logits'].argmax(1)==targets).all() and (views['PAweak']['logits'].argmax(1)!=targets).all()
    q=torch.tensor([True,False,False,False])
    result=objective(model,active,targets,q,ids,'EVIDENCE_STOP',role='U_s')
    assert result['summary']['accepted_count']==1 and result['summary']['active_packets']==1 and result['loss']>0
    expected=.1/(2*4)*sum(F.relu(math.log(.9)-F.log_softmax(views[view]['logits'],dim=1)[0,0]) for view in ('PAweak','REFweak'))
    assert torch.allclose(result['loss'],expected,atol=1e-15,rtol=1e-15)
    full=objective(model,active,targets,all_q,ids,'EVIDENCE_STOP',role='U_s')
    assert torch.allclose(full['loss'],4*result['loss'],atol=1e-15,rtol=1e-15)
    self_control=objective(model,active,targets,all_q,ids,'SELF_STOP')
    assert self_control['loss'].item()==0  # Nominal confidence already fulfils pi.
    all_false=objective(model,active,targets,torch.zeros(4,dtype=torch.bool),ids,'EVIDENCE_STOP',role='U_s')
    assert all_false['loss'].item()==0
    encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
    endpoints=[active['identity'],active['sequence_raw'],active['physical'],active['reference_raw'],
               head.weight,head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw]
    for mode in ('ZERO','SELF_STOP','EVIDENCE_STOP'):
        output=objective(model,active,targets,all_q,ids,mode)
        assert output['loss'] is output['weighted_loss'] and output['summary']['head_readout_calls']==3
        assert output['summary']['head_readout_packets']==12 and output['summary']['extra_IQ_forwards']==0
        gradient=torch.autograd.grad(output['loss'],endpoints,retain_graph=True,allow_unused=True)
        assert all(g is None for g in gradient[2:])
        if mode=='ZERO':assert output['loss'].item()==0 and all(g is not None and torch.count_nonzero(g)==0 for g in gradient[:2])
        json.dumps(output['summary'],allow_nan=False)
    false_grad=torch.autograd.grad(all_false['loss'],endpoints,retain_graph=True,allow_unused=True)
    assert all(g is None for g in false_grad[2:]) and all(torch.count_nonzero(g)==0 for g in false_grad[:2])
    _raises(lambda:objective(model,active,targets,q,ids,'EVIDENCE_STOP',role='L_s'))
    for role in ('V','V_s','query','target'):_raises(lambda role=role:objective(model,active,targets,all_q,ids,'SELF_STOP',role=role))
    duplicate=list(ids);duplicate[-1]=duplicate[0]
    _raises(lambda:objective(model,active,targets,all_q,duplicate,'SELF_STOP'))
    wrong_rx=list(ids);wrong_rx[0]=(0,2,1,0,100)
    _raises(lambda:objective(model,active,targets,all_q,wrong_rx,'SELF_STOP'))
    _raises(lambda:objective(model,active,targets.float(),all_q,ids,'SELF_STOP'))
    _raises(lambda:objective(model,active,targets,all_q.float(),ids,'SELF_STOP'))
    _raises(lambda:objective(model,active,torch.full_like(targets,-1),all_q,ids,'SELF_STOP'))
    _raises(lambda:objective(model,active,targets,all_q,ids,'UNKNOWN'))
    _raises(lambda:readouts(model,dict(active,joint=active['joint']+.01)))
    nonfinite=dict(active);nonfinite['physical']=active['physical'].clone();nonfinite['physical'][0,0]=float('inf')
    _raises(lambda:readouts(model,nonfinite))

    # Real constructor and same-call capture verify upstream shared parameters,
    # not merely leaf endpoint derivatives. No checkpoint/IQ source is loaded.
    from experiments.cvs_crossrx_sharpness20 import design as d, model as api
    config=d.config(next(row for row in d.rows() if row['arm']=='BASE_CE' and row['model_seed']==d.SEEDS[0]))
    actual=api.build(config,'cpu').eval().requires_grad_(True)
    with torch.no_grad():actual.id_backbone.encoder.response_gain.fill_(.08)
    generator=torch.Generator().manual_seed(2026101607)
    parts=forward(actual,torch.randn(4,2,256,generator=generator))
    y=(parts['logits'].argmax(1)+1).remainder(6)
    head=actual.id_backbone.encoder.core.id_backbone.cls_head;encoder=actual.id_backbone.encoder
    named=dict(actual.named_parameters());count=sum(p.numel() for p in actual.parameters())
    before=copy.deepcopy(actual.state_dict());rng=_rng_snapshot();flags=[(m,m.training) for m in actual.modules()]
    output=objective(actual,parts,y,all_q,ids,'EVIDENCE_STOP')
    tensors=[parts['identity'],parts['sequence_raw'],parts['physical'],parts['reference_raw'],head.weight,head.gain,
             encoder.response_gain,encoder.sequence_pool.gain_raw]
    grads=torch.autograd.grad(output['loss'],tensors,retain_graph=True,allow_unused=True)
    assert all(g is None for g in grads[2:]) and all(bool(torch.isfinite(g).all()) and g.norm()>0 for g in grads[:2])
    selected=[named['id_backbone.encoder.core.id_backbone.sinc.low_coordinate'],
              named['id_backbone.encoder.sequence_pool.output_projection.weight'],
              named['id_backbone.encoder.core.id_backbone.cls_head.pa_norm.weight'],
              named['id_backbone.encoder.response_projection.0.weight']]
    shared=torch.autograd.grad(output['loss'],selected,retain_graph=True,allow_unused=True)
    assert all(g is not None and bool(torch.isfinite(g).all()) and g.norm()>0 for g in shared[:2]) and shared[2:]==(None,None)
    main=torch.autograd.grad(F.cross_entropy(parts['logits'],y),[head.weight,head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw],
                             retain_graph=True,allow_unused=True)
    assert all(g is not None and bool(torch.isfinite(g).all()) and g.norm()>0 for g in main)
    with torch.no_grad():proxy=readouts(actual,parts)
    assert all(not proxy[view]['logits'].requires_grad for view in ('self','PAweak','REFweak'))
    assert _equal(before,actual.state_dict()) and _equal(rng,_rng_snapshot())
    assert all(m.training==flag for m,flag in flags) and all(p.grad is None for p in actual.parameters())
    assert sum(p.numel() for p in actual.parameters())==count
    return dict(status='PASS',torch=torch.__version__,device='cpu',original_joint_bitwise_exact=True,
        fixed_beta=.5,probability=.9,coefficient=.1,three_readout_equal_cost=True,
        stopped_satisfied_probability=True,violating_probability_active=True,original_q_not_regated=True,
        denominator_full_batch=True,physical_reference_W_three_gains_aux_gradients_None=True,
        actual_constructor_shared_parameter_gradients_nonzero=dict(Sinc=float(shared[0].norm()),sequence_projection=float(shared[1].norm())),
        PA_norm_reference_projection_aux_None=True,main_default_head_all_gains_joint_gradient_nonzero=True,
        ZERO_and_all_qFalse_exact_zero=True,no_grad_proxy_supported=True,state_flags_RNG_unchanged=True,
        role_ID_shape_nonfinite_negatives=True,data_reads=0,checkpoint_reads=0,optimizer_updates=0,
        boundary='Synthetic compensation loss only; pseudo-label/gate provenance is caller responsibility, no efficacy claim')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
