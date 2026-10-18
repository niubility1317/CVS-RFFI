"""Pure CPU tensor checks of the exact C00 hook plus As-only target entropy."""
from copy import deepcopy
import math
import random
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,entropy,model
from experiments.cvs_prop_contrast76 import model_checks as previous


def quality_grouped(*args,**kwargs):
    value=previous.grouped(*args,**kwargs)
    for i,record in enumerate(value[1]):
        record['signal_to_added_noise_db']=(-20.,-10.,0.,3.,8.,15.,30.,100.)[i%8]
    return value


def run():
    negatives=0
    def bad(fn):
        nonlocal negatives
        try:fn()
        except (ValueError,TypeError,KeyError,AttributeError):negatives+=1
        else:raise AssertionError('Invalid or hidden source input accepted')
    def same_np(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
    configs={r['arm']:d.config(r) for r in d.rows()};assert set(configs)=={'Q','R'}
    ns='FcrossRX/train/e21/step4441/L_S'
    snr=[float('-inf'),-100.,-20.,-10.,0.,10.,100.,float('inf')]*8
    eps=[entropy.epsilon_from_snr(v) for v in snr]
    assert eps[0]==.2 and abs(eps[4]-.105)<1e-15 and eps[7]==.01
    assert all(.01<=v<=.2 for v in eps) and all(a>=b for a,b in zip(eps[:8],eps[1:8]))
    for value in (-20.,0.,10.,100.):
        assert abs(entropy.epsilon_from_snr(value)-(.01+.19/(1+10**(value/10))))<1e-15
    for invalid in (None,float('nan'),True,'10'):
        bad(lambda value=invalid:entropy.epsilon_from_snr(value))
    py0=random.getstate();np0=np.random.get_state();t0=torch.get_rng_state().clone()
    q=entropy.assignment(snr,ns,'Q');r=entropy.assignment(snr,ns,'R')
    assert q['permutation']==list(range(64)) and q['epsilon_assigned']==q['epsilon_original']
    assert r==entropy.assignment(snr,ns,'R') and sorted(r['epsilon_assigned'])==sorted(q['epsilon_original'])
    assert r['permutation']!=list(range(64)) and r['epsilon_assigned']==[r['epsilon_original'][i]for i in r['permutation']]
    assert r['signal_to_added_noise_db'][0]=='-inf' and r['signal_to_added_noise_db'][7]=='+inf'
    assert py0==random.getstate() and same_np(np0,np.random.get_state()) and torch.equal(t0,torch.get_rng_state())
    for values,namespace,mode in ((snr[:63],ns,'Q'),(snr,ns,'U'),(snr,'target/L_S','Q')):
        bad(lambda values=values,namespace=namespace,mode=mode:entropy.assignment(values,namespace,mode))
    g=torch.Generator().manual_seed(7701)
    z0=torch.randn(128,6,dtype=torch.float64,generator=g,requires_grad=True)
    z1=torch.randn(128,6,dtype=torch.float64,generator=g,requires_grad=True)
    y=(torch.arange(128)//2)%6;mask=(torch.arange(128)+1).remainder(2)!=0
    ee=torch.full((64,),.01,dtype=torch.float64)
    parity=entropy.paired_delta(z0[mask],z1[mask],y[mask],ee)
    nominal=.125*(F.cross_entropy(z0[mask],y[mask],label_smoothing=.01)+F.cross_entropy(z1[mask],y[mask],label_smoothing=.01))
    assert torch.equal(parity['weighted_delta'],torch.zeros((),dtype=torch.float64))
    nominal_g=torch.autograd.grad(nominal,(z0,z1),retain_graph=True)
    restored_g=torch.autograd.grad(nominal+parity['weighted_delta'],(z0,z1),retain_graph=True)
    assert all(torch.equal(a,b)for a,b in zip(nominal_g,restored_g))
    ee=torch.tensor(q['epsilon_original'],dtype=torch.float64)
    delta=entropy.paired_delta(z0[mask],z1[mask],y[mask],ee)
    observed=nominal+delta['weighted_delta']
    direct=.125*(entropy.variable_ce(z0[mask],y[mask],ee)+entropy.variable_ce(z1[mask],y[mask],ee))
    observed_g=torch.autograd.grad(observed,(z0,z1),retain_graph=True)
    direct_g=torch.autograd.grad(direct,(z0,z1))
    assert torch.allclose(observed,direct,atol=1e-15,rtol=0)
    assert all(torch.allclose(a,b,atol=1e-17,rtol=1e-13)for a,b in zip(observed_g,direct_g))
    assert all(bool((v[~mask]==0).all())for v in observed_g)
    for yy,epsilon in ((y[mask]-6,ee),(y[mask],ee.requires_grad_()),(y[mask],torch.zeros_like(ee))):
        bad(lambda yy=yy,epsilon=epsilon:entropy.variable_ce(z0[mask],yy,epsilon))
    raw=torch.randn(128,2,256,generator=g)
    meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor((1,3,4,6,8))[(torch.arange(128)//12)%5],
        day_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128)+70,base_index=torch.arange(128))
    def toy(arm):return model.attach(previous.Toy(d.control_config()),configs[arm])
    baseline=previous.Toy(d.control_config());default=toy('Q')
    torch.manual_seed(7711)
    with patch.object(model.control,'grouped_view',quality_grouped):
        ref_loss,_,_=previous.call(baseline,raw,y,meta,model.control.hook)
    ref_loss.backward();ref_rng=torch.get_rng_state().clone()
    torch.manual_seed(7711)
    with patch.object(model.control,'grouped_view',quality_grouped),patch.object(entropy,'epsilon_from_snr',lambda value:.01):
        default_loss,_,_=previous.call(default,raw,y,meta,model.hook)
    default_loss.backward()
    assert torch.equal(default_loss,ref_loss) and torch.equal(default.parameter.grad,baseline.parameter.grad)
    assert torch.equal(torch.get_rng_state(),ref_rng) and default._prop_records[-1]['entropy_weighted_delta']==0
    records={};payloads={}
    for arm in ('Q','R'):
        current=toy(arm);payload=[];model.physics_sink(current,payload.append)
        torch.manual_seed(7711);py0=random.getstate();np0=np.random.get_state()
        with patch.object(model.control,'grouped_view',quality_grouped):
            loss,_,call_kw=previous.call(current,raw,y,meta,model.hook)
        loss.backward();record=current._prop_records[-1];e=record['entropy'];native=current._full_records[-1]
        assert torch.equal(torch.get_rng_state(),ref_rng) and py0==random.getstate() and same_np(np0,np.random.get_state())
        assert len(current.inputs)==5 and torch.equal(current.inputs[1],current.inputs[2])
        assert all(torch.equal(a,b)for a,b in zip(current.inputs,baseline.inputs))
        assert torch.equal(current.inputs[3],raw) and torch.equal(current.inputs[4],raw)
        assert current._prop_generated==64 and current._prop_slots==128 and current._pair_generated==64
        assert current._full_channel.packets==64 and current._full_channel.obj.calls==1 and sum(current._prop_coverage.values())==128
        assert record['As_loss_weighted']==0 and record['As_SupCon_coefficient']==0 and record['ordinary_SupCon_coefficient']==.1
        assert record['ordinary_SupCon_loss']==baseline._prop_records[-1]['ordinary_SupCon_loss']
        assert record['As_supervised']==128 and e['As_CE_passes']==2 and e['epsilon_same_both_passes'] and e['all_labeled_CE_retained']
        assert abs(e['weighted_delta']-.125*(e['new_CE0']+e['new_CE1']-e['nominal_CE0']-e['nominal_CE1']))<1e-7
        assert abs(float(loss.detach()-ref_loss.detach())-e['weighted_delta'])<5e-7
        assert abs(native['total_L']-float(loss.detach()))<5e-7 and abs(e['base_total_L']-float(ref_loss.detach()))<5e-7
        assert current.parameter.grad is not None and bool(torch.isfinite(current.parameter.grad).all())
        assert not torch.equal(current.parameter.grad,baseline.parameter.grad)
        assert len(payload)==1 and payload[0]['labels_L_As']==y[mask].tolist() and payload[0]['arm']==arm
        assert payload[0]['entropy']['epsilon_assigned']==[payload[0]['entropy']['epsilon_original'][i]for i in payload[0]['entropy']['permutation']]
        assert payload[0]['entropy']['signal_to_added_noise_db']==[v['signal_to_added_noise_db']for v in payload[0]['view0']]
        assert all(v['channel_generated']is False for v in payload[0]['view1'])
        assert current._entropy_records is current._prop_records and current._prop_arm=='C00'
        records[arm]=record;payloads[arm]=payload[0];model.physics_sink(current)
    assert records['Q']['entropy']['epsilon_mean']==records['R']['entropy']['epsilon_mean']
    assert payloads['Q']['entropy']['epsilon_original']==payloads['R']['entropy']['epsilon_original']
    assert payloads['Q']['entropy']['epsilon_assigned']!=payloads['R']['entropy']['epsilon_assigned']
    old_grouped,old_terms=model.control.grouped_view,model.pair.as_ce_terms
    failure=toy('Q');sentinel=lambda value:None;model.control.physics_sink(failure,sentinel)
    saved_apply=failure._full_channel.obj.apply;failure._full_channel.obj.apply=saved_apply
    def missing(*args,**kwargs):return previous.grouped(*args,**kwargs)
    with patch.object(model.control,'grouped_view',missing):bad(lambda:previous.call(failure,raw,y,meta,model.hook))
    assert model.control._PHYSICS_SINKS.get(failure)is sentinel and failure._full_channel.obj.apply is saved_apply
    assert not failure._sat_engine.active and model.control.grouped_view is old_grouped and model.pair.as_ce_terms is old_terms
    assert model.pair.independent_view is previous.model.pair.independent_view and 'forward_identity_only'not in vars(failure)
    model.control.physics_sink(failure)
    invalid=dict(meta,day_i=torch.zeros(128,dtype=torch.long))
    with patch.object(model.control,'grouped_view',quality_grouped):
        bad(lambda:previous.call(toy('Q'),raw,y,invalid,model.hook))
        bad(lambda:model.hook(toy('Q'),**dict(call_kw,labels=y-6)))
    return dict(status='PASS',focused_negatives=negatives,epsilon_formula_and_infinite_limits=True,
        exact_epsilon_multiset_private_R_permutation=True,Q_physical_assignment=True,two_As_passes_same_epsilon=True,
        default_epsilon_C00_loss_grad_RNG_exact=True,As_only_actual_coefficient=.125,total_As_weight=.25,
        six_L_calls=True,As_IQ_repeated=True,primary_generations64=True,logical_As_slots128=True,
        ordinary_raw_inputs_loss_and_RNG_unchanged=True,actual_total_reconstructable=True,
        assigned_CE_live_gradients=True,true_L_labels_and_native_SNR_in_sink=True,scoped_failure_restores=True,
        CVS_NN=0,real_IQ=0,checkpoint_reads=0,optimizer_updates=0,target_access=False)


def boundary_run():
    """Only the affected zero-power resolver and composed-hook path."""
    import json
    negatives=0
    def bad(fn):
        nonlocal negatives
        try:fn()
        except (ValueError,TypeError,KeyError):negatives+=1
        else:raise AssertionError('Invalid missing-ratio boundary accepted')
    finite=dict(signal_to_added_noise_db=0.)
    zero_signal=dict(signal_to_added_noise_db=None,measured_added_noise_power_pre_agc=.3)
    zero_noise=dict(signal_to_added_noise_db=None,measured_added_noise_power_pre_agc=0.)
    assert entropy.resolve_snr(finite)==(0.,'finite')
    assert entropy.resolve_snr(zero_signal)==(float('-inf'),'zero_signal')
    assert entropy.resolve_snr(zero_noise)==(float('inf'),'zero_added_noise')
    for record in ({},{'signal_to_added_noise_db':None},
            dict(zero_signal,measured_added_noise_power_pre_agc=-.1),
            dict(zero_signal,measured_added_noise_power_pre_agc=float('nan')),
            dict(zero_signal,measured_added_noise_power_pre_agc=float('inf'))):
        bad(lambda record=record:entropy.resolve_snr(record))
    ns='FcrossRX/train/e21/step4441/L_S'
    mapping=entropy.from_records([zero_signal,zero_noise]+[finite]*62,ns,'Q')
    assert mapping['epsilon_original'][:2]==[.2,.01]
    assert mapping['signal_to_added_noise_db_native'][:2]==[None,None]
    assert mapping['signal_to_added_noise_db'][:2]==['-inf','+inf']
    assert mapping['snr_zero_signal_count']==mapping['snr_zero_added_noise_count']==1 and mapping['snr_finite_count']==62
    json.dumps(mapping,allow_nan=False)
    def with_boundary(*args,**kwargs):
        value=quality_grouped(*args,**kwargs)
        value[1][0].update(zero_signal);value[1][1].update(zero_noise)
        return value
    configs={r['arm']:d.config(r)for r in d.rows()}
    gen=torch.Generator().manual_seed(7722);raw=torch.randn(128,2,256,generator=gen);y=(torch.arange(128)//2)%6
    meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor((1,3,4,6,8))[(torch.arange(128)//12)%5],
        day_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128)+70,base_index=torch.arange(128))
    reference=previous.Toy(d.control_config());torch.manual_seed(7723)
    with patch.object(model.control,'grouped_view',with_boundary):ref,_,_=previous.call(reference,raw,y,meta,model.control.hook)
    ref_rng=torch.get_rng_state().clone()
    result={}
    for arm in ('Q','R'):
        current=model.attach(previous.Toy(d.control_config()),configs[arm]);payload=[];model.physics_sink(current,payload.append)
        torch.manual_seed(7723);py=random.getstate();np0=np.random.get_state()
        with patch.object(model.control,'grouped_view',with_boundary):loss,_,_=previous.call(current,raw,y,meta,model.hook)
        loss.backward();e=current._prop_records[-1]['entropy'];p=payload[0]['entropy'];result[arm]=p
        assert e['snr_zero_signal_count']==e['snr_zero_added_noise_count']==1 and e['snr_finite_count']==62
        assert e['epsilon_max']==.2 and e['epsilon_min']==.01 and e['As_CE_passes']==2
        assert e['epsilon_multiset_preserved'] and e['epsilon_same_both_passes']
        assert current._prop_generated==64 and current._prop_slots==128 and len(current.inputs)==5
        assert all(torch.equal(a,b)for a,b in zip(reference.inputs,current.inputs))
        assert torch.equal(ref_rng,torch.get_rng_state()) and py==random.getstate()
        np1=np.random.get_state();assert np0[0]==np1[0] and np.array_equal(np0[1],np1[1]) and np0[2:]==np1[2:]
        assert abs(float((loss-ref).detach())-e['weighted_delta'])<5e-7
        assert abs(current._full_records[-1]['total_L']-e['base_total_L']-e['weighted_delta'])<1e-12
        assert current.parameter.grad is not None and bool(torch.isfinite(current.parameter.grad).all())
        assert p['signal_to_added_noise_db_native'][:2]==[None,None] and p['signal_to_added_noise_db'][:2]==['-inf','+inf']
        assert p['epsilon_original'][:2]==[.2,.01]
        json.dumps(payload[0],allow_nan=False);model.physics_sink(current)
    assert result['Q']['epsilon_original']==result['R']['epsilon_original']
    assert sorted(result['Q']['epsilon_assigned'])==sorted(result['R']['epsilon_assigned'])
    assert result['Q']['epsilon_assigned']!=result['R']['epsilon_assigned']
    return dict(status='PASS',focused_boundary_negatives=negatives,native_None_zero_signal_epsilon=.2,
        native_None_zero_added_noise_epsilon=.01,original_None_preserved=True,JSON_allow_nan_false=True,
        both_actual_composed_hooks=True,physical64_logical128=True,no_CE_drop=True,
        live_corrected_loss_gradient=True,ordinary_inputs_and_global_RNG_unchanged=True,
        CVS_NN=0,real_IQ=0,checkpoint_reads=0,optimizer_updates=0,target_access=False)
