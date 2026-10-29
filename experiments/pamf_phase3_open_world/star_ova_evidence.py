"""Shared class-conditional evidence readers on frozen same-record P1/P0.

The OVA arm borrows a hard-negative objective, not the OVANet method: F0,
its identity head and G remain frozen; no open-set entropy minimization runs.
Only source L labels, source OE membership and source RX/day domains enter.
"""
from __future__ import annotations

import numpy as np

from . import source_open_readout as core
from . import source_domain_oe_readout as domain


RUN_ID='20261029-phase3-star-ova-evidence-s2026092701-r01'
DESCRIPTOR_SCHEMA='class_conditional_paired_branch_evidence_v1'
STATE_SCHEMA='pamf_phase3_star_ova_readouts_v1'
DESCRIPTOR_DIM=106
CLASS_COUNT=6
MODEL_NAMES=('membership','ova_hard')
MODEL_SEED=2026092701
STEPS=1000
SHAPES=dict(weight1=(32,106),bias1=(32,),weight2=(1,32),bias2=(1,))


def _other_max(values):
    return np.where(np.eye(CLASS_COUNT,dtype=bool)[None,:,:],-np.inf,values[:,None,:]).max(axis=2)


def _evidence(values):
    other=_other_max(values)
    return np.stack((values,other,values-other),axis=2)


def _path_descriptor(branches, logits, geometry):
    values=core._branches(branches);fitted=core._geometry(geometry)
    z=core._matrix(logits,'complete frozen F0 six-class logits',CLASS_COUNT,fp32=True)
    if len(z)!=len(values['fused160']):raise ValueError('Branch/head row alignment differs')
    columns=[]
    for name in core.DIMENSIONS:
        b=fitted[name];u=core._unit((values[name]-b['mean'])/b['scale']);p=b['prototypes']
        cosine=np.clip(np.einsum('nd,kd->nk',u,p,optimize=False),-1.,1.)
        # Same squared shared-whitener distances as source_consensus.descriptor.
        white=u@b['whitener'].T
        centers=np.einsum('nd,kd->nk',p,b['whitener'],optimize=False)
        distance=(np.einsum('nd,nd->n',white,white)[:,None]
            +np.einsum('kd,kd->k',centers,centers)[None,:]
            -2.*np.einsum('nd,kd->nk',white,centers,optimize=False))
        columns.extend((_evidence(cosine),_evidence(-np.maximum(distance,0.))))
    top=z.max(axis=1);exponential=np.exp(z-top[:,None])
    denominator=np.sort(exponential,axis=1).sum(axis=1)
    other=_other_max(z);energy=top+np.log(denominator)
    columns.append(np.stack((exponential/denominator[:,None],z,other,z-other,
                            np.repeat(energy[:,None],CLASS_COUNT,axis=1)),axis=2))
    return np.concatenate(columns,axis=2)


def descriptor(p1branches,p1logits,p1geometry,p0branches,p0logits,p0geometry):
    first=_path_descriptor(p1branches,p1logits,p1geometry)
    second=_path_descriptor(p0branches,p0logits,p0geometry)
    if len(first)!=len(second):raise ValueError('Same-physical-record P1/P0 rows differ')
    return core._fp32(np.concatenate((first,second),axis=2),'class-conditional P1/P0 evidence')


def _desc(value):
    array=np.asarray(value)
    if (array.ndim!=3 or not len(array) or array.shape[1:]!=(CLASS_COUNT,DESCRIPTOR_DIM)
            or array.dtype!=np.float32 or not np.isfinite(array).all()):
        raise ValueError('Nonempty finite FP32[N,6,106] evidence required')
    return array.astype(np.float64)


def _labels(value,count,name,*,all_classes=False):
    labels=np.asarray(value)
    if (labels.shape!=(count,) or labels.dtype.kind not in 'iu' or not np.isin(labels,range(CLASS_COUNT)).all()
            or (all_classes and set(np.unique(labels).tolist())!=set(range(CLASS_COUNT)))):
        raise ValueError('Aligned six-class integer source '+name+' required')
    return labels.astype(np.int64,copy=False)


def _canonical(values,domains,y=None,closed=None):
    flat=values.reshape(len(values),-1)
    # Visible L labels only break exact evidence ties; they never route query.
    tie=() if y is None else (closed,y)
    order=np.lexsort(tie+tuple(flat[:,i] for i in range(flat.shape[1]-1,-1,-1))+(domains,))
    return values[order],domains[order],None if y is None else y[order],None if closed is None else closed[order]


def _normalizer(known,oe,wk,wo):
    train=np.concatenate((known,oe));weights=np.r_[.6*wk,.4*wo]
    with np.errstate(over='ignore',invalid='ignore'):
        # Six class slots share one population normalizer with uniform 1/6 mass.
        mean=(np.sort(train,axis=1).mean(axis=1)*weights[:,None]).sum(0)/weights.sum()
        variance=(np.sort(np.square(train-mean),axis=1).mean(axis=1)*weights[:,None]).sum(0)/weights.sum()
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError('Nonfinite role/domain/class weighted source statistics')
    scale=np.sqrt(np.maximum(variance,0.));scale=np.where(scale<=1e-8,1.,scale)
    return mean,scale,core._fp32((train-mean)/scale,'normalized source class evidence')


def _loss_vectors(known,oe,y,closed,arm):
    import torch
    softplus=torch.nn.functional.softplus;row=torch.arange(len(known),device='cpu')
    if arm=='membership':known_loss=softplus(-known[row,closed])
    elif arm=='ova_hard':
        negatives=known.masked_fill(torch.arange(CLASS_COUNT,device='cpu')[None,:]==y[:,None],-torch.inf)
        known_loss=.5*(softplus(-known[row,y])+softplus(negatives).max(dim=1).values)
    else:raise ValueError('Only membership and ova_hard reader arms are defined')
    return known_loss,softplus(oe).max(dim=1).values


def fit(known_desc,oe_desc,known_y,known_closed,known_domains,oe_domains,model_seed=MODEL_SEED):
    if type(model_seed) is not int or model_seed!=MODEL_SEED:raise ValueError('Fixed source model seed required')
    known,oe=_desc(known_desc),_desc(oe_desc)
    y=_labels(known_y,len(known),'L labels',all_classes=True)
    closed=_labels(known_closed,len(known),'original P1 closed predictions')
    kd=domain._domains(known_domains,len(known),'known');od=domain._domains(oe_domains,len(oe),'OE')
    known,kd,y,closed=_canonical(known,kd,y,closed);oe,od,_,_=_canonical(oe,od)
    kcounts,wk=domain._role_weights(kd);ocounts,wo=domain._role_weights(od)
    mean,scale,normalized=_normalizer(known,oe,wk,wo)
    import torch
    x=torch.tensor(normalized.tolist(),dtype=torch.float32,device='cpu')
    kw=torch.tensor(wk.tolist(),dtype=torch.float32,device='cpu');ow=torch.tensor(wo.tolist(),dtype=torch.float32,device='cpu')
    ty=torch.tensor(y.tolist(),dtype=torch.int64,device='cpu');tc=torch.tensor(closed.tolist(),dtype=torch.int64,device='cpu')
    models,losses={},{};first_initial=None;initial_equal=True
    with torch.random.fork_rng(devices=[]),torch.enable_grad():
        for arm in MODEL_NAMES:
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
                g=(hidden@parameters['weight2'].T+parameters['bias2'])[:,:,0]
                kl,ol=_loss_vectors(g[:len(known)],g[len(known):],ty,tc,arm)
                loss=.5*(kw*kl).sum()+.5*(ow*ol).sum()
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite domain-balanced class-conditional source loss')
                trace.append(float(loss.detach().item()));loss.backward();optimizer.step()
                if step%100==0:
                    print('STAR_OVA_READOUT_PROGRESS',arm,step,STEPS,'loss_before_update',trace[-1],flush=True)
            models[arm]={key:core._fp32(value.detach().cpu().tolist(),arm+' '+key) for key,value in parameters.items()}
            losses[arm]=trace
    if not initial_equal:raise ValueError('Source reader arms must start from identical matrices')
    state=dict(schema=STATE_SCHEMA,descriptor_schema=DESCRIPTOR_SCHEMA,descriptor_dim=DESCRIPTOR_DIM,
        class_count=CLASS_COUNT,model_names=list(MODEL_NAMES),mean=mean,scale=scale,models=models,training=dict(
            model_seed=model_seed,steps_per_arm=STEPS,total_optimizer_updates=2*STEPS,losses=losses,
            parameter_counts={name:sum(v.size for v in model.values()) for name,model in models.items()},
            known_count=len(known),oe_count=len(oe),domain_count=15,
            domain_counts=dict(known=kcounts.tolist(),oe=ocounts.tolist()),
            normalizer_role_prior=dict(known=.6,oe=.4),normalizer_class_prior=1/6,
            normalizer_variance='role/domain/class-weighted population variance',shared_normalizer=True,
            loss_role_prior=dict(known=.5,oe=.5),loss_record='full-batch loss before each update',
            known_loss=dict(membership='softplus(-g[original_P1_closed])',
                ova_hard='.5*(softplus(-g[source_y])+max_other_class softplus(g))'),
            oe_loss='max_all_six_classes softplus(g)',oe_transmitter_labels_used=False,
            optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8,foreach=False),
            fresh_optimizer_per_arm=True,seed_reset_per_arm=True,initial_parameters_equal=initial_equal,
            canonical_order='domain then flattened descriptor lexicographic; visible L y/closed break exact ties',
            source_L_labels_used=True,source_V_reads=0,novel_reads=0,target_reads=0,
            encoder_updates=0,identity_head_updates=0,DURA_updates=0,
            shared_class_conditional_weights=True,independent_identity_heads=0,device='cpu',dropout=0.))
    _state(state)
    return state


def _state(state):
    if (not isinstance(state,dict) or set(state)!={'schema','descriptor_schema','descriptor_dim','class_count','model_names','mean','scale','models','training'}
            or state['schema']!=STATE_SCHEMA or state['descriptor_schema']!=DESCRIPTOR_SCHEMA
            or type(state['descriptor_dim']) is not int or state['descriptor_dim']!=DESCRIPTOR_DIM
            or type(state['class_count']) is not int or state['class_count']!=CLASS_COUNT
            or state['model_names']!=list(MODEL_NAMES) or not isinstance(state['training'],dict)
            or not isinstance(state['models'],dict) or set(state['models'])!=set(MODEL_NAMES)):
        raise ValueError('Frozen class-conditional readout state schema differs')
    for key in ('mean','scale'):
        if core._array(state[key],(DESCRIPTOR_DIM,),key).dtype!=np.float64:raise ValueError('FP64 source normalizer required')
    if np.any(state['scale']<=0):raise ValueError('Positive source normalizer scales required')
    for name,model in state['models'].items():
        if not isinstance(model,dict) or set(model)!=set(SHAPES):raise ValueError('Shared 106->32->1 model keys differ')
        for key,shape in SHAPES.items():core._array(model[key],shape,name+' '+key,fp32=True)
    return state['mean'],state['scale'],state['models']


def score(desc,state):
    values=_desc(desc);mean,scale,models=_state(state)
    x=core._fp32((values-mean)/scale,'normalized class-conditional query evidence');columns=[]
    for name in MODEL_NAMES:
        model=models[name]
        # Fixed reduction order keeps class-column permutations exact in FP32;
        # BLAS row blocking can otherwise change the last few bits by slot.
        hidden=np.maximum(np.einsum('ncd,hd->nch',x,model['weight1'],optimize=False)+model['bias1'],np.float32(0.))
        columns.append((np.einsum('nch,oh->nco',hidden,model['weight2'],optimize=False)+model['bias2'])[:,:,0])
    result=np.stack(columns,axis=1).astype(np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite class-conditional readout forward')
    return result
