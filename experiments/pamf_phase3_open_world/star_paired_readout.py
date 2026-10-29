"""Fixed source-only paired evidence readers on frozen F0 P1/P0 descriptors.

Each fresh 200->16->1 MLP follows exactly 1000 CPU AdamW updates. Step200 is
an intermediate snapshot of that same trajectory. The control masks P0 before
normalization, training and scoring; F0, its six-class head and G are absent.
"""
from __future__ import annotations

import numpy as np

from . import source_open_readout as core
from . import source_domain_oe_readout as domain
from . import star_consensus_readout as calibration


RUN_ID='20261029-phase3-star-paired-evidence-s2026092701-r01'
SCHEMA='pamf_phase3_star_paired_prediction_v1'
STATE_SCHEMA='pamf_phase3_star_paired_readouts_v1'
CALIBRATION_STATUS='STAR_PAIRED_THRESHOLDS_FROZEN'
METHODS=('energy','p1_mlp16_step200','paired_mlp16_step200','p1_mlp16_step1000','paired_mlp16_step1000')
LEARNED_METHODS=METHODS[1:]
PRIMARY=METHODS[-1]
PRIMARY_INDEX=4
PRIMARY_PATH='P1'
RECEIVERS=calibration.RECEIVERS
SCENES=calibration.SCENES
PAIR_DIM=200
MODEL_SEED=2026092701
STEPS=1000
CHECKPOINTS=(200,1000)
ARMS=('p1_only','paired')
SHAPES=dict(weight1=(16,200),bias1=(16,),weight2=(1,16),bias2=(1,))


def pair_descriptors(p1, p0):
    first=core._matrix(p1,'same-record P1 consensus',100,fp32=True)
    second=core._matrix(p0,'same-record P0 consensus',100,fp32=True)
    if len(first)!=len(second):raise ValueError('P1/P0 descriptor row alignment differs')
    return core._fp32(np.concatenate((first,second),axis=1),'paired consensus')


def _pair(value):
    return core._matrix(value,'paired P1/P0 consensus',PAIR_DIM,fp32=True)


def _masked(value, arm):
    result=np.array(value,copy=True)
    if arm=='p1_only':result[:,100:]=0.
    return result


def _canonical(values, domains):
    # Generalize the existing domain/descriptor ordering from 100 to 200.
    keys=tuple(values[:,i] for i in range(PAIR_DIM-1,-1,-1))+(domains,)
    order=np.lexsort(keys)
    return values[order],domains[order]


def _normalizer(known, oe, wk, wo):
    train=np.concatenate((known,oe),axis=0);weights=np.r_[.6*wk,.4*wo]
    with np.errstate(over='ignore',invalid='ignore'):
        mean=(train*weights[:,None]).sum(0)/weights.sum()
        variance=(np.square(train-mean)*weights[:,None]).sum(0)/weights.sum()
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError('Nonfinite source role/domain weighted population statistics')
    scale=np.sqrt(np.maximum(variance,0.));scale=np.where(scale<=1e-8,1.,scale)
    return dict(mean=mean,scale=scale),core._fp32((train-mean)/scale,'normalized paired source descriptors')


def fit_readouts(known_pair, oe_pair, known_domains, oe_domains, model_seed=MODEL_SEED):
    if type(model_seed) is not int or model_seed!=MODEL_SEED:
        raise ValueError('Only the fixed source paired readout seed is allowed')
    known,oe=_pair(known_pair),_pair(oe_pair)
    kd=domain._domains(known_domains,len(known),'known');od=domain._domains(oe_domains,len(oe),'OE')
    import torch
    normalizers,models,losses={},{},{}
    first_initial=None;initial_equal=True
    with torch.random.fork_rng(devices=[]),torch.enable_grad():
        for arm in ARMS:
            k,kdomain=_canonical(_masked(known,arm),kd)
            o,odomain=_canonical(_masked(oe,arm),od)
            kcounts,wk=domain._role_weights(kdomain);ocounts,wo=domain._role_weights(odomain)
            normalizers[arm],normalized=_normalizer(k,o,wk,wo)
            x=torch.tensor(normalized.tolist(),dtype=torch.float32,device='cpu')
            kw=torch.tensor(wk.tolist(),dtype=torch.float32,device='cpu')
            ow=torch.tensor(wo.tolist(),dtype=torch.float32,device='cpu')
            torch.random.default_generator.manual_seed(model_seed)
            parameters={key:torch.nn.Parameter(torch.zeros(shape,dtype=torch.float32,device='cpu')) for key,shape in SHAPES.items()}
            torch.nn.init.xavier_uniform_(parameters['weight1']);torch.nn.init.xavier_uniform_(parameters['weight2'])
            initial={key:value.detach().clone() for key,value in parameters.items()}
            if first_initial is None:first_initial=initial
            else:initial_equal=all(torch.equal(first_initial[key],value) for key,value in initial.items())
            optimizer=torch.optim.AdamW(list(parameters.values()),lr=.001,weight_decay=.0001,
                betas=(.9,.999),eps=1e-8,foreach=False)
            trace=[]
            for step in range(1,STEPS+1):
                optimizer.zero_grad(set_to_none=True)
                hidden=torch.relu(x@parameters['weight1'].T+parameters['bias1'])
                g=(hidden@parameters['weight2'].T+parameters['bias2']).reshape(-1)
                loss=.5*(kw*torch.nn.functional.softplus(-g[:len(k)])).sum()+.5*(ow*torch.nn.functional.softplus(g[len(k):])).sum()
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite domain-balanced paired readout loss')
                trace.append(float(loss.detach().item()));loss.backward();optimizer.step()
                if step in CHECKPOINTS:
                    name=('p1' if arm=='p1_only' else 'paired')+'_mlp16_step'+str(step)
                    models[name]={key:core._fp32(value.detach().cpu().tolist(),name+' '+key) for key,value in parameters.items()}
                if step%200==0:
                    print('STAR_PAIRED_READOUT_PROGRESS',arm,step,STEPS,'loss_before_update',trace[-1],flush=True)
            losses[arm]=trace
    if not initial_equal:raise ValueError('Both source arms must start from identical matrices')
    state=dict(schema=STATE_SCHEMA,pair_dim=PAIR_DIM,method_names=list(LEARNED_METHODS),
        normalizers=normalizers,models={name:models[name] for name in LEARNED_METHODS},training=dict(
            model_seed=model_seed,steps_per_arm=STEPS,total_optimizer_updates=2*STEPS,checkpoint_steps=list(CHECKPOINTS),
            checkpoint_updates={name:200 if name.endswith('200') else 1000 for name in LEARNED_METHODS},
            parameter_counts={name:sum(value.size for value in models[name].values()) for name in LEARNED_METHODS},
            losses=losses,known_count=len(known),oe_count=len(oe),domain_count=15,
            domain_counts=dict(known=kcounts.tolist(),oe=ocounts.tolist()),
            normalizer_role_prior=dict(known=.6,oe=.4),loss_role_prior=dict(known=.5,oe=.5),
            normalizer_variance='weighted population variance',normalizer_prior_origin='fixed original source L6300/OE4200 role ratio',
            loss_record='full-batch role/domain-balanced softplus before each update',
            optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8,foreach=False),
            fresh_optimizer_per_arm=True,seed_reset_per_arm=True,initial_parameters_equal=initial_equal,
            checkpoint_semantics='post-update snapshots of the same 1000-step trajectory; no additional fit',
            canonical_order='domain then 200D descriptor lexicographic; control P0 masked first',
            control_P0_masked_before_normalization=True,source_membership_supervision=True,oe_transmitter_labels_used=False,
            encoder_updates=0,identity_head_updates=0,DURA_updates=0,source_V_reads=0,novel_reads=0,target_reads=0,
            device='cpu',dropout=0.))
    _state(state)
    return state


def _state(state):
    if (not isinstance(state,dict) or set(state)!={'schema','pair_dim','method_names','normalizers','models','training'}
            or state['schema']!=STATE_SCHEMA or type(state['pair_dim']) is not int or state['pair_dim']!=PAIR_DIM
            or state['method_names']!=list(LEARNED_METHODS) or not isinstance(state['training'],dict)
            or not isinstance(state['normalizers'],dict) or set(state['normalizers'])!=set(ARMS)
            or not isinstance(state['models'],dict) or set(state['models'])!=set(LEARNED_METHODS)):
        raise ValueError('Frozen paired readout state schema differs')
    for arm,normalizer in state['normalizers'].items():
        if not isinstance(normalizer,dict) or set(normalizer)!={'mean','scale'}:raise ValueError('Two source-only normalizer arrays required')
        for key in ('mean','scale'):
            values=core._array(normalizer[key],(200,),arm+' '+key)
            if values.dtype!=np.float64:raise ValueError('FP64 weighted source normalizer arrays required')
        if np.any(normalizer['scale']<=0):raise ValueError('Positive source normalizer scales required')
    control,paired=(state['normalizers'][arm] for arm in ARMS)
    if (np.any(control['mean'][100:]!=0) or np.any(control['scale'][100:]!=1)
            or any(not np.array_equal(control[key][:100],paired[key][:100]) for key in ('mean','scale'))):
        raise ValueError('The P1-only normalizer must exclude P0 and match the shared P1 coordinates')
    for name,model in state['models'].items():
        if not isinstance(model,dict) or set(model)!=set(SHAPES):raise ValueError('Frozen paired MLP parameter keys differ')
        for key,shape in SHAPES.items():core._array(model[key],shape,name+' '+key,fp32=True)
    return state['normalizers'],state['models']


def score_readouts(pair, state):
    values=_pair(pair);normalizers,models=_state(state);prepared={}
    for arm in ARMS:
        normalizer=normalizers[arm]
        prepared[arm]=core._fp32((_masked(values,arm)-normalizer['mean'])/normalizer['scale'],'normalized '+arm)
    columns=[]
    for name in LEARNED_METHODS:
        x=prepared['paired' if name.startswith('paired_') else 'p1_only'];model=models[name]
        hidden=np.maximum(x@model['weight1'].T+model['bias1'],np.float32(0.))
        columns.append((hidden@model['weight2'].T+model['bias2']).reshape(-1))
    result=np.column_stack(columns).astype(np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite paired readout forward')
    return result


def score_compact(pair, energy, state):
    scores=score_readouts(pair,state);values=np.asarray(energy)
    if values.shape!=(len(scores),) or values.dtype.kind!='f' or not np.isfinite(values).all():
        raise ValueError('Aligned finite original F0 P1 energy vector required')
    return np.column_stack((values,scores)).astype(np.float64,copy=False)


def calibrate_known(ids, y, closed, baseline, scores, rx, scenario):
    thresholds,report=calibration.calibrate_known(ids,y,closed,baseline,scores,rx,scenario,method_names=METHODS)
    report.update(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,primary_method=PRIMARY,primary_path=PRIMARY_PATH)
    return thresholds,report
