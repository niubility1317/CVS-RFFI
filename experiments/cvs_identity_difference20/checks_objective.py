"""Balanced source-shaped math and actual-constructor gradient checks only."""
import copy
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .objective import objective
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


def metadata():
    rows=[(side,rx,y,day) for side in (0,1) for rx in (1,3,4,6,8) for y in range(6) for day in (1,2,3)]
    side=torch.tensor([v[0] for v in rows]);rx=torch.tensor([v[1] for v in rows]);y=torch.tensor([v[2] for v in rows])
    ids=[(0,r,day,i,1000+i) for i,(_,r,_,day) in enumerate(rows)]
    return rows,y,rx,side,ids


def parts(model,rows, *, aligned):
    a=torch.zeros(180,160,dtype=torch.float64)
    for i,(side,rx,y,day) in enumerate(rows):
        r=(1,3,4,6,8).index(rx)
        if aligned:
            # Both sides have different RX offsets plus the same balanced day
            # offset. Raw subtraction exactly cancels them before normalization.
            a[i].fill_((side+1)*(r+1)*10+day);a[i,y]+=2.
        else:
            angle=0. if side==0 else math.radians(30+y*10+r*2)
            a[i,y]=math.cos(angle);a[i,y+6]=math.sin(angle)
    a.requires_grad_();p=torch.zeros_like(a,requires_grad=True);r=torch.zeros_like(a,requires_grad=True);s=torch.zeros_like(a,requires_grad=True)
    encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
    joint=((a+head.gain.tanh()*p)+encoder.response_gain.tanh()*r)+(.2*encoder.sequence_pool.gain_raw.tanh())*s
    return dict(identity=a,physical=p,reference_raw=r,sequence_raw=s,joint=joint)


def _raises(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected malformed/incomplete source cohort rejection')


def check():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(20261016);model=Fixture().double()
    rows,y,rx,side,ids=metadata();aligned=parts(model,rows,aligned=True)
    result=objective(model,aligned,y,rx,side,ids,'CROSS_DIFF')
    class_pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
    expected=torch.stack([2*F.one_hot(torch.tensor(i),160).double()-2*F.one_hot(torch.tensor(j),160).double() for i,j in class_pairs])
    assert torch.equal(result['tensors']['delta_A'],expected.expand(5,-1,-1))
    assert torch.equal(result['tensors']['delta_B'],expected.expand(5,-1,-1))
    assert result['loss'].item()==0 and result['summary']['CROSS_active300']==0
    stopped=torch.autograd.grad(result['loss'],[aligned['identity'],aligned['sequence_raw']],retain_graph=True)
    assert all(torch.count_nonzero(g)==0 for g in stopped)
    active=parts(model,rows,aligned=False);encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
    before=copy.deepcopy(model.state_dict());rng=_rng_snapshot();count=sum(p.numel() for p in model.parameters())
    outputs={mode:objective(model,active,y,rx,side,ids,mode) for mode in ('ZERO','SELF_DIFF','CROSS_DIFF')}
    endpoints=[active['identity'],active['sequence_raw'],active['physical'],active['reference_raw'],head.weight,
               head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw]
    for mode,output in outputs.items():
        assert output['loss'] is output['weighted_loss'] and output['summary']['SELF_computed_terms']==300
        assert output['summary']['CROSS_computed_terms']==300 and output['summary']['extra_IQ_forwards']==0
        assert output['summary']['default_W_used'] is False and output['summary']['SELF_joint_bitwise_exact']
        grads=torch.autograd.grad(output['loss'],endpoints,retain_graph=True,allow_unused=True)
        assert all(g is None for g in grads[2:])
        if mode=='ZERO':assert output['loss'].item()==0 and all(torch.count_nonzero(g)==0 for g in grads[:2])
        else:
            assert all(bool(torch.isfinite(g).all()) and g[side==0].norm()>0 and g[side==1].norm()>0 for g in grads[:2])
        json.dumps(output['summary'],allow_nan=False)
    cross=outputs['CROSS_DIFF'];mu_grads=torch.autograd.grad(cross['loss'],[cross['tensors']['mu_A'],cross['tensors']['mu_B']],retain_graph=True)
    assert all(g.norm()>0 for g in mu_grads)
    perm=torch.tensor([4,2,5,0,3,1]);relabeled=objective(model,active,perm[y],rx,side,ids,'CROSS_DIFF')
    assert torch.allclose(relabeled['loss'],cross['loss'],atol=2e-15,rtol=2e-15)
    # Reorder all source rows including RX blocks; fixed day order stabilizes
    # the raw cell means even when rows arrive in a different batch order.
    index=torch.arange(179,-1,-1);shuffled={key:value[index] for key,value in active.items()}
    reordered=objective(model,shuffled,y[index],rx[index],side[index],[ids[i] for i in index.tolist()],'CROSS_DIFF')
    assert torch.equal(reordered['loss'],cross['loss'])
    mapping={1:8,3:6,4:4,6:3,8:1};renamed_rx=torch.tensor([mapping[int(r)] for r in rx])
    renamed_ids=[(eq,mapping[r],day,sig,base) for eq,r,day,sig,base in ids]
    renamed=objective(model,active,y,renamed_rx,side,renamed_ids,'CROSS_DIFF')
    assert torch.allclose(renamed['loss'],cross['loss'],atol=2e-15,rtol=2e-15)
    collapsed=dict(active,identity=torch.zeros_like(active['identity'],requires_grad=True))
    collapsed['joint']=((collapsed['identity']+head.gain.tanh()*collapsed['physical'])
        +encoder.response_gain.tanh()*collapsed['reference_raw'])+(.2*encoder.sequence_pool.gain_raw.tanh())*collapsed['sequence_raw']
    nearzero=objective(model,collapsed,y,rx,side,ids,'CROSS_DIFF')
    assert nearzero['summary']['difference_norms']['nearzero_count']==150
    assert nearzero['summary']['difference_norms']['exact_zero_count']==150 and nearzero['summary']['CROSS_active300']==300
    assert abs(nearzero['loss'].item()-.09)<1e-15  # Disclose collapse, never remove these terms.
    # W does not enter the computation at all, including reconstruction.
    without_W=copy.deepcopy(model);del without_W.id_backbone.encoder.core.id_backbone.cls_head.weight
    assert torch.equal(objective(without_W,active,y,rx,side,ids,'CROSS_DIFF')['loss'],cross['loss'])
    assert _equal(before,model.state_dict()) and _equal(rng,_rng_snapshot()) and sum(p.numel() for p in model.parameters())==count
    assert all(p.grad is None for p in model.parameters())
    for role in ('U_s','V','V_s','query','target'):_raises(lambda role=role:objective(model,active,y,rx,side,ids,'CROSS_DIFF',role=role))
    duplicate=list(ids);duplicate[-1]=duplicate[0];_raises(lambda:objective(model,active,y,rx,side,duplicate,'CROSS_DIFF'))
    wrong_day=list(ids);eq,r,d,sig,base=wrong_day[-1];wrong_day[-1]=(eq,r,2,sig,base)
    _raises(lambda:objective(model,active,y,rx,side,wrong_day,'CROSS_DIFF'))
    wrong_rx=rx.clone();wrong_rx[0]=3;_raises(lambda:objective(model,active,y,wrong_rx,side,ids,'CROSS_DIFF'))
    hidden=y.clone();hidden[0]=-1;_raises(lambda:objective(model,active,hidden,rx,side,ids,'CROSS_DIFF'))
    _raises(lambda:objective(model,active,y,rx,side.bool(),ids,'CROSS_DIFF'))
    _raises(lambda:objective(model,active,y.float(),rx,side,ids,'CROSS_DIFF'))
    _raises(lambda:objective(model,{key:value[:-1] for key,value in active.items()},y[:-1],rx[:-1],side[:-1],ids[:-1],'CROSS_DIFF'))
    nonfinite=dict(active);nonfinite['identity']=active['identity'].clone();nonfinite['identity'][0,0]=float('nan')
    _raises(lambda:objective(model,nonfinite,y,rx,side,ids,'CROSS_DIFF'))
    _raises(lambda:objective(model,dict(active,joint=active['joint']+.001),y,rx,side,ids,'CROSS_DIFF'))
    _raises(lambda:objective(model,active,y,rx,side,ids,'UNKNOWN'))
    # A genuine constructor and same-call capture prove the shared feature graph.
    from experiments.cvs_crossrx_sharpness20 import design as d, model as api
    c=d.config(next(row for row in d.rows() if row['arm']=='BASE_CE' and row['model_seed']==d.SEEDS[0]))
    actual=api.build(c,'cpu').eval().requires_grad_(True)
    with torch.no_grad():actual.id_backbone.encoder.response_gain.fill_(.08)
    generator=torch.Generator().manual_seed(2026101611)
    captured=forward(actual,torch.randn(180,2,256,generator=generator))
    encoder=actual.id_backbone.encoder;head=encoder.core.id_backbone.cls_head;named=dict(actual.named_parameters())
    live_before=copy.deepcopy(actual.state_dict());live_rng=_rng_snapshot();flags=[(m,m.training) for m in actual.modules()]
    real=objective(actual,captured,y,rx,side,ids,'CROSS_DIFF')
    protected=[captured['physical'],captured['reference_raw'],head.weight,head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw]
    assert all(g is None for g in torch.autograd.grad(real['loss'],protected,retain_graph=True,allow_unused=True))
    shared=torch.autograd.grad(real['loss'],[named['id_backbone.encoder.core.id_backbone.sinc.low_coordinate'],
        named['id_backbone.encoder.sequence_pool.output_projection.weight']],retain_graph=True)
    assert all(bool(torch.isfinite(g).all()) and g.norm()>0 for g in shared)
    main=torch.autograd.grad(F.cross_entropy(captured['logits'],y),[head.weight,head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw],retain_graph=True)
    assert all(bool(torch.isfinite(g).all()) and g.norm()>0 for g in main)
    assert _equal(live_before,actual.state_dict()) and _equal(live_rng,_rng_snapshot())
    assert all(m.training==flag for m,flag in flags) and all(p.grad is None for p in actual.parameters())
    return dict(status='PASS',torch=torch.__version__,device='cpu',side_dtype='torch.long:0=A,1=B',
        raw_same_RX_offset_cancellation=True,satisfied_direction_exact_zero_and_zero_gradient=True,
        label_permutation_RX_relabeling_and_row_order_invariant=True,both_A_B_feature_gradients_attached=True,
        PA_reference_W_three_gain_aux_None=True,default_W_not_read=True,matched_SELF300_CROSS300_cost=True,
        nearzero_differences_disclosed_without_filtering=True,
        actual_constructor_shared_gradients=dict(Sinc=float(shared[0].norm()),SIG_projection=float(shared[1].norm())),
        main_default_W_three_gains_joint_gradient_nonzero=True,state_flags_RNG_unchanged=True,
        metadata_role_shape_finite_negatives=True,data_reads=0,checkpoint_reads=0,optimizer_updates=0,
        boundary='Synthetic identity-difference objective only; no hardware/DG or efficacy claim')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
