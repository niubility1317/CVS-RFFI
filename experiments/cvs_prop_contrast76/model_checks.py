"""Tensor-only real six-forward hook and exact As mask/loss checks."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from . import design as d,model,relations
from experiments.cvs_full_channel_pair68 import checks as oldchecks


class Shifted(oldchecks.ToyShifted):
    def __getattr__(self,key):
        if key=='obj':raise AttributeError(key)
        return getattr(self.obj,key)


class Toy(oldchecks.ToyModel):
    def __init__(self,c):
        self._full_channel=Shifted();self._full_channel.obj.calls=0
        self._sat_engine=oldchecks.Engine('AS_CE');self._sat_steps=[];self._full_records=[];self._feature_learning_report={}
        self.parameter=torch.tensor([.3,-.2,.7,-.5,.9,-.8],requires_grad=True);self.inputs=[]
        model.attach(self,c)
    def forward_identity_only(self,x,*args,**kwargs):
        self.inputs.append(x.detach().clone())
        maps={key:self._sat_engine.apply(SimpleNamespace(p=.2,training=True,inplace=False),
            torch.cat((x[:,:,:8]*(1+self.parameter[0]),x[:,:,:8].square()*(1+self.parameter[1])),1))
            for key in relations.LEVELS}
        logits=self._sat_engine.apply(SimpleNamespace(p=.2,training=True,inplace=False),x.mean((1,2))[:,None]*self.parameter[None,:])
        return dict(tx_logits=logits,maps=maps)


def grouped(primary,raw,meta,namespace,share):
    """No physics engine: deterministic fake records and 64 captured RX calls."""
    waves=[];records=[]
    for i in range(64):
        pid=[int(meta[key][i]) for key in model.PHYSICAL_FIELDS];session=f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
        hw=primary.ch.receiver_for_session(None,1,session);primary.sessions[session]=asdict(hw)
        scene='practical_low_suburban' if pid[2]==1 else ('practical_mid_urban' if pid[2]==2 else 'practical_low_urban')
        waves.append(raw[i]*.8+.03)
        records.append(dict(sample_id=pid,session_id=session,namespace=namespace,scene=scene,
            state_end='LOS',receiver_lock_status='locked',channel_equalization_applied=False,
            group=[pid[1],pid[2],pid[0]],group_id=session,share_propagation=share))
        primary.coverage=getattr(primary,'coverage',{})
    primary.calls+=1;primary.packets+=64
    return torch.stack(waves),records


def original(self,**kw):
    """Actual loss/routing order and ordinary penalty; tiny tensors replace CVS."""
    raw,y,domains=kw['labeled_inputs'],kw['labels'],kw['domains'];mask=(torch.arange(128)+kw['optimizer_step']).remainder(2)!=0
    ordinary=raw+.2;first=kw['labeled_output'];second=self.forward_identity_only(ordinary,domain_labels=domains)
    common=model.pair.base.terms(first['tx_logits'],second['tx_logits'],y,1.)[0]
    meta={key:value[mask] for key,value in kw['extra'][1].items()}
    sat,_=self._full_channel.apply(raw[mask],meta,epoch=(kw['optimizer_step']-1)//222+1,step=kw['optimizer_step'])
    mixed=ordinary.clone();mixed[mask]=sat
    a=self.forward_identity_only(mixed,domain_labels=domains);b=self.forward_identity_only(mixed,domain_labels=domains)
    extra=model.pair.base.terms(a['tx_logits'],b['tx_logits'],y,1.)[0]
    vals=[]
    for ordinary_maps,satellite_maps in ((first['maps'],a['maps']),(second['maps'],b['maps'])):
        value,_=model.pair.base.relations.penalty(ordinary_maps,satellite_maps,y,kw['extra'][1]['rx_i'],mask,'ORD128');vals.append(value)
    aux=torch.stack(vals).mean()
    r0=self.forward_identity_only(raw,domain_labels=domains);r1=self.forward_identity_only(raw,domain_labels=domains)
    shape=model.pair.base.terms(r0['tx_logits'],r1['tx_logits'],y,1.)[0]
    total=.5*common+.5*extra+.1*aux+.25*shape
    self._full_records.append(dict(total_L=float(total.detach()),SupCon=float(aux.detach()),SupCon_weight=.1))
    return total-kw['identity_loss'],{}


def call(toy,raw,y,meta,hook):
    first=toy.forward_identity_only(raw+.2);toy.inputs.clear();ce=torch.nn.functional.cross_entropy(first['tx_logits'],y,label_smoothing=.01)
    kw=dict(labels=y,optimizer_step=1,labeled_inputs=raw,extra=(None,meta),domains=torch.zeros(128,dtype=torch.long),
        labeled_output=first,identity_loss=ce)
    with patch.object(model.pair.original,'hook',original):delta,info=hook(toy,**kw)
    return ce+delta,info,kw


def baseline(self,**kw):
    # The original SAT18 CE-only hook, with the same grouped first generator.
    primary=self._full_channel.obj;had='apply'in vars(primary);previous=vars(primary).get('apply')
    def apply(raw,meta,epoch,step):return grouped(primary,raw,meta,f'FcrossRX/train/e{epoch}/step{step}/L_S',self._prop_share)
    primary.apply=apply
    try:return model.pair.original.satellite_hook(self,**kw)
    finally:
        if had:primary.apply=previous
        else:delattr(primary,'apply')


def run():
    negatives=0
    def bad(fn):
        nonlocal negatives
        try:fn()
        except (ValueError,TypeError,KeyError,AttributeError):negatives+=1
        else:raise AssertionError('Invalid source/auxiliary context accepted')
    configs={r['arm']:d.config(r) for r in d.rows()};assert set(configs)=={'C00','C10','C01','C11'}
    g=torch.Generator().manual_seed(7601);raw=torch.randn(128,2,256,generator=g)
    # Every As group has multiple TX; same TX also appears at other RX.
    y=(torch.arange(128)//2)%6
    meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor((1,3,4,6,8))[(torch.arange(128)//12)%5],
        day_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128)+70,base_index=torch.arange(128))
    expected_mask=(torch.arange(128)+1).remainder(2)!=0
    old_view,old_penalty,old_terms=model.pair.independent_view,model.pair.base.relations.penalty,model.pair.base.terms
    off=Toy(configs['C00']);ref=Toy(configs['C00'])
    torch.manual_seed(7611)
    with patch.object(model,'grouped_view',grouped):loss,_,_=call(off,raw,y,meta,model.hook)
    loss.backward();rng=torch.get_rng_state().clone()
    torch.manual_seed(7611);expected,_,_=call(ref,raw,y,meta,baseline);expected.backward()
    assert torch.equal(loss,expected) and torch.equal(off.parameter.grad,ref.parameter.grad)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(off.inputs)==5 and torch.equal(off.inputs[1],off.inputs[2])
    assert torch.equal(off.inputs[3],raw) and torch.equal(off.inputs[4],raw)
    assert off._prop_generated==64 and off._prop_slots==128 and off._pair_generated==64
    assert off._full_channel.packets==64 and off._full_channel.obj.calls==1
    assert off._prop_records[0]['As_loss_weighted']==0 and off._prop_records[0]['ordinary_SupCon_coefficient']==.1
    assert sum(off._prop_coverage.values())==128
    # ON differs by exactly the one extra .05 As loss on the same existing graphs.
    on=Toy(configs['C01']);payload=[];model.physics_sink(on,payload.append)
    torch.manual_seed(7611);np0=np.random.get_state()
    with patch.object(model,'grouped_view',grouped):on_loss,_,_=call(on,raw,y,meta,model.hook)
    on_loss.backward();np1=np.random.get_state()
    assert torch.equal(rng,torch.get_rng_state()) and np0[0]==np1[0] and np.array_equal(np0[1],np1[1]) and np0[2:]==np1[2:]
    rr=on._prop_records[0]
    assert rr['As_loss_raw']>0 and abs(rr['As_loss_weighted']-.05*rr['As_loss_raw'])<1e-7
    assert abs(float(on_loss.detach()-loss.detach())-rr['As_loss_weighted'])<3e-7
    assert len(rr['As_details'])==4 and all(v['valid_anchors']>0 for v in rr['As_details'])
    assert rr['ordinary_SupCon_loss']==off._prop_records[0]['ordinary_SupCon_loss']
    assert abs(on._full_records[-1]['total_L']-float(on_loss.detach()))<3e-7
    assert on.parameter.grad is not None and bool(torch.isfinite(on.parameter.grad).all())
    assert payload[0]['labels_L_As']==y[expected_mask].tolist() and len(payload[0]['view0'])==len(payload[0]['view1'])==64
    assert all(b['channel_generated'] is False for b in payload[0]['view1'])
    model.physics_sink(on);assert on not in model._PHYSICS_SINKS
    for arm in ('C10','C11'):
        toy=Toy(configs[arm]);torch.manual_seed(7611)
        with patch.object(model,'grouped_view',grouped):v,_,_=call(toy,raw,y,meta,model.hook)
        assert toy._prop_records[0]['share_propagation'] is True and len(toy.inputs)==5
    # Formula masks explicitly ignore day for positives, restrict negatives to group.
    yy=torch.tensor([0,0,1,1,0,2]);rx=torch.tensor([1,3,1,3,1,1]);day=torch.tensor([1,2,1,2,3,1]);eq=torch.ones(6,dtype=torch.long)
    pos,neg,den,valid=relations.masks(yy,rx,day,eq)
    assert pos[0,1] and not pos[0,4] and neg[0,2] and neg[0,5] and not neg[0,3]
    maps={k:torch.randn(6,4,8,generator=g,requires_grad=True) for k in relations.LEVELS}
    aux,details=relations.penalty(maps,yy,rx,day,eq)
    gradients=torch.autograd.grad(aux,tuple(maps.values()));assert all(float(v.abs().sum())>0 for v in gradients)
    zero,details=relations.penalty(maps,yy,torch.ones(6,dtype=torch.long),day,eq)
    assert float(zero)==0 and all(v['valid_anchors']==0 and v['zero_anchors']==6 for v in details)
    for vals in ((yy-6,rx,day,eq),(yy,rx*0,day,eq),(yy,rx,day*0,eq),(yy,rx,day,eq*0)):
        bad(lambda vals=vals:relations.penalty(maps,*vals))
    invalid={k:v.clone() for k,v in maps.items()};invalid[relations.LEVELS[0]][0,0,0]=float('nan')
    bad(lambda:relations.penalty(invalid,yy,rx,day,eq))
    # No-positive ON keeps both satellite CE terms and completes the same calls.
    single=Toy(configs['C01']);single_meta=dict(meta,rx_i=torch.ones(128,dtype=torch.long))
    with patch.object(model,'grouped_view',grouped):zero_loss,_,_=call(single,raw,y,single_meta,model.hook)
    assert single._prop_records[0]['As_loss_raw']==0 and single._prop_records[0]['As_supervised']==128
    assert len(single.inputs)==5 and zero_loss.requires_grad
    failure=Toy(configs['C01']);saved_apply=failure._full_channel.obj.apply;failure._full_channel.obj.apply=saved_apply
    def broken(*args,**kwargs):raise ValueError('Synthetic first channel failure')
    with patch.object(model,'grouped_view',broken):bad(lambda:call(failure,raw,y,meta,model.hook))
    assert failure._full_channel.obj.apply is saved_apply and not failure._sat_engine.active
    assert model.pair.independent_view is old_view and model.pair.base.relations.penalty is old_penalty and model.pair.base.terms is old_terms
    assert 'apply' not in vars(off._full_channel.obj) and 'forward_identity_only' not in vars(off)
    illegal=dict(meta,day_i=torch.zeros(128,dtype=torch.long))
    with patch.object(model,'grouped_view',grouped):bad(lambda:call(Toy(configs['C00']),raw,y,illegal,model.hook))
    return dict(status='PASS',focused_negatives=negatives,OFF_original_SAT18_loss_grad_RNG_exact=True,
        six_L_calls=True,As_IQ_repeated=True,ON_live_As_graphs=True,As_coefficient=.05,ordinary_coefficient=.1,
        crossRX_anyday_positives=True,within_RX_day_eq_negatives=True,zero_aux_no_CE_drop=True,
        primary_generations64=True,logical_As_slots128=True,true_L_labels_in_sink=True,scoped_failure_restores=True,
        no_global_RNG_draws=True,CVS_NN=0,real_IQ=0,checkpoint_reads=0,optimizer_updates=0,target_access=False)
