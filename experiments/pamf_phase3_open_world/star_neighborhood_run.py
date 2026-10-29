"""Frozen source prior, L-only anchors, then all-U neighborhoods and future Q."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from . import star_neighborhood_core as core
from . import star_consensus_run as base
from . import star_paired_run as paired
from . import star_class_risk_run as class_run
from . import star_class_risk as class_core
from . import star_relative_source as relative_source
from . import star_relative_target as public
from . import unlabeled_registration_core as reg

RUN_ID='20261029-phase3-star-neighborhood-s2026092701-r01'
SOURCE_SCHEMA='pamf_phase3_star_neighborhood_source_v1'
SOURCE_STATUS='STAR_NEIGHBORHOOD_SOURCE_FROZEN'
SCHEMA='pamf_phase3_star_neighborhood_target_v1'
STATUS='STAR_NEIGHBORHOOD_PREDICTIONS_COMPLETE'
REGISTRY_STATUS='STAR_NEIGHBORHOOD_REGISTRY_FROZEN'
STATE_SCHEMA=core.STATE_SCHEMA
CLASS_COMMIT='bb507e2b6f4d66bd808a485d70d64d25c3556da3'
STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')
SOURCE_L_COUNT=6300
U_COUNT=384
Q_COUNT=864
SOURCE_FIELDS={'run_id','code_commit','source_config','L_cache_root','paired_prediction_root','class_risk_prediction_root','output_root','batch_size'}
TARGET_FIELDS={'run_id','code_commit','source_root','U_public_root','U_capsule_id','U_split_id','Q_public_root','Q_capsule_id','Q_split_id','output_root','batch_size','dura_checkpoint','dura_release'}
STATE_FIELDS={'schema','cluster_embedding','cluster_logits','cluster_margin','cluster_radius','anonymous_indices','anonymous_penalty'}
DIAG_FIELDS={'ids','cluster_labels','assignments','candidate_mask','cluster_sizes','anonymous_ids'}
PRED_FIELDS={'ids','U_ids','state_names','prediction','closed_prediction','baseline_closed_prediction','knownness'}|{'joint_margins_'+s for s in STATES}
_write,_npz=base._write,base._npz
_read=base.runner._read
_same=relative_source._same
_snapshot,_check_models=base._snapshot,base._check_models
_source_backend,_source_smoke=base._backend,base._smoke
close_role=base.close_role


def _header(c,fields):
    if (not isinstance(c,dict) or set(c)!=fields or c.get('run_id')!=RUN_ID or not isinstance(c.get('code_commit'),str)
            or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None or type(c.get('batch_size')) is not int or c['batch_size']!=128):
        raise ValueError('Fixed independent neighborhood configuration and batch128 required')


def validate_source_config(c):
    _header(c,SOURCE_FIELDS)
    if any(not isinstance(c[k],str) or not c[k].strip() for k in SOURCE_FIELDS-{'source_config','batch_size'}):raise ValueError('Nonempty source inputs required')
    if not isinstance(c['source_config'],dict) or c['source_config'].get('run_id')!=RUN_ID or c['source_config'].get('row_id')!='star-neighborhood-source-s2026092701':
        raise ValueError('The original F0 constructor metadata must belong to this source stage')
    mapped=dict(run_id=base.RUN_ID,code_commit=c['code_commit'],source_config=dict(c['source_config'],run_id=base.RUN_ID,row_id='star-consensus-s2026092701'),
        cache_root=str(Path(c['L_cache_root']).parent),output_root=c['output_root'],batch_size=c['batch_size'])
    base.validate_config(mapped)
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in ('L_cache_root','paired_prediction_root','class_risk_prediction_root')):
        raise ValueError('Source output cannot overwrite a frozen parent/input')
    return c


def validate_target_config(c):
    _header(c,TARGET_FIELDS)
    if any(not isinstance(c[k],str) or not c[k].strip() for k in TARGET_FIELDS-{'batch_size'}):raise ValueError('Public handles, source bundle and frozen G paths required')
    for role,parent in (('U','discovery'),('Q','evaluation')):
        root=Path(c[role+'_public_root'])
        if root.name!='public' or root.parent.name!=parent:raise ValueError('Separate discovery U and future Q public roles required')
    if c['U_capsule_id']==c['Q_capsule_id'] or c['U_split_id']==c['Q_split_id']:raise ValueError('U/Q handles must be independent')
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in ('source_root','U_public_root','Q_public_root')):
        raise ValueError('Target output cannot overwrite source/public inputs')
    return c


def _groups():return np.asarray([(y,r,d) for y in range(6) for r in base.previous.RECEIVERS for d in (1,2,3)],np.int64)


def _source_prior(c):
    cr=Path(c['class_risk_prediction_root']);cm=_read(cr/'complete.json')
    if (cm.get('status')!=class_run.STATUS or cm.get('schema')!=class_run.SCHEMA or cm.get('run_id')!=class_run.RUN_ID
            or cm.get('code_commit')!=CLASS_COMMIT or cm.get('parent_run_id')!=paired.RUN_ID or cm.get('parent_code_commit')!=class_run.PARENT_COMMIT
            or cm.get('source_only') is not True or cm.get('classification_path')!='P1' or cm.get('use_predicted_class_only') is not True
            or any(cm.get(k)!=0 for k in ('target_reads','new_encoder_updates','new_identity_head_updates','new_readout_updates'))):
        raise ValueError('The exact frozen class-risk source prior is required before model/source IO')
    pm=class_run._parent(c['paired_prediction_root']);pr=Path(c['paired_prediction_root'])
    cal=_read(cr/'calibration/complete.json');main=cal.get('methods',{}).get('predicted_class',{});groups=main.get('groups',{})
    expected={'overall'}|{'rx/'+str(r) for r in base.previous.RECEIVERS}|{'scene/'+s for s in base.previous.SCENES}|{'rxscene/'+str(r)+'/'+s for r in base.previous.RECEIVERS for s in base.previous.SCENES}
    if (cal.get('status')!=class_run.CALIBRATION_STATUS or cal.get('code_commit')!=CLASS_COMMIT or cal.get('run_id')!=class_run.RUN_ID
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False or main.get('constraints_passed') is not True
            or set(groups)!=expected or any(row.get('constraints_passed') is not True for row in groups.values())):
        raise ValueError('All 42 source-known protection groups must pass; source unknown success is not this joint candidate gate')
    with np.load(cr/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'global','conditional'}:raise ValueError('Only the fixed class-risk scalar state is allowed')
        glob=z['global'].copy();cond=z['conditional'].copy()
    if (glob.shape!=(1,) or glob.dtype!=np.float64 or cond.shape!=(6,) or cond.dtype!=np.float64
            or not np.isfinite(glob).all() or not np.isfinite(cond).all()
            or cal.get('global_threshold')!=float(glob[0]) or not np.array_equal(np.asarray(cal.get('conditional_thresholds')),cond)
            or float(glob[0])!=class_run._parent_threshold(pr,pm)):
        raise ValueError('The copied class-risk thresholds and paired global ancestor differ')
    readouts=paired.load_readouts(pr/'readouts.npz',pr/'readouts.json');paired._training_complete(readouts)
    return dict(geometry={p:relative_source._load_geometry(pr/'geometry'/(p+'.npz')) for p in base.PATHS},readouts=readouts,
        thresholds=dict(schema=class_core.STATE_SCHEMA,global_threshold=float(glob[0]),conditional_thresholds=cond),
        marker=dict(parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_known_42_passed=True,
            paired_run_id=paired.RUN_ID,paired_code_commit=class_run.PARENT_COMMIT,class_risk_run_id=class_run.RUN_ID,class_risk_code_commit=CLASS_COMMIT),
        parent_paths=dict(paired=str(pr),class_risk=str(cr)))


def _save_prior(root,prior):
    (root/'geometry').mkdir(parents=True)
    for path in base.PATHS:relative_source._save_geometry(root/'geometry'/(path+'.npz'),prior['geometry'][path])
    paired._save_readouts(root/'readouts.npz',prior['readouts']);_write(root/'readouts.json',dict(training=prior['readouts']['training']))
    _npz(root/'thresholds.npz',{'global':np.asarray([prior['thresholds']['global_threshold']],np.float64),'conditional':prior['thresholds']['conditional_thresholds']})


def _load_L(path,identity,dura):return base._role(path,'L',identity,dura)


def _prior_evidence(b1,l1,b0,l0,prior):
    desc1=base.core.descriptor(b1,l1,prior['geometry']['P1']);desc0=base.core.descriptor(b0,l0,prior['geometry']['P0'])
    pair=paired.core.pair_descriptors(desc1,desc0)
    raw=paired.core.score_readouts(pair,prior['readouts'])[:,3]
    margin=class_core.apply(raw,l1.argmax(axis=1),prior['thresholds'])[:,1]
    return reg.embedding(b1,prior['geometry']['P1']),l1,margin


def _cache_evidence(backend,role,start,end,prior,device):
    b1,l1=base._encode(backend['reader'],role['restored'][start:end],device)
    b0,l0=base._encode(backend['reader'],role['received'][start:end],device)
    return _prior_evidence(b1,l1,b0,l0,prior)


def source(c,device='cuda:0'):
    validate_source_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source bundle required')
    prior=_source_prior(c);before=copy.deepcopy(prior);backend,_=_source_backend(c['source_config'],device);snapshots=_snapshot(backend)
    if backend['reader_identity']!=prior['marker']['parent_F0_identity']:raise ValueError('Frozen F0 source/cache prior identity differs')
    smoke=_source_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual source checkpoint smoke required')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);role=None
    try:
        role=_load_L(c['L_cache_root'],backend['reader_identity'],prior['marker']['dura_identity'])
        if len(role['ids'])!=SOURCE_L_COUNT:raise ValueError('Only the original source L count may form anchors')
        sums=np.zeros((90,688));margin_sum=np.zeros(90);counts=np.zeros(90,np.int64)
        for start in range(0,SOURCE_L_COUNT,c['batch_size']):
            end=min(start+c['batch_size'],SOURCE_L_COUNT);embedding,logits,margin=_cache_evidence(backend,role,start,end,prior,device)
            m=role['metadata'];slots=m['y'][start:end]*15+np.searchsorted(base.previous.RECEIVERS,m['rx_i'][start:end])*3+m['day_i'][start:end]-1
            np.add.at(sums,slots,embedding);np.add.at(margin_sum,slots,margin);np.add.at(counts,slots,1)
            print('STAR_NEIGHBORHOOD_SOURCE_PROGRESS',end,SOURCE_L_COUNT,flush=True)
        if np.any(counts==0):raise ValueError('All 90 legal source groups must contribute anchors')
        anchors=dict(embedding=(sums/counts[:,None]).astype(np.float32),margin=margin_sum/counts,group_keys=_groups())
        if not _same(prior,before):raise ValueError('Frozen source prior changed while aggregating L')
        _check_models(snapshots);_save_prior(out/'prior',prior);_npz(out/'anchors.npz',anchors)
        done=dict(prior['marker'])
        done.update(dict(status=SOURCE_STATUS,schema=SOURCE_SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'],config=c,model_config=c['source_config'],
            source_only=True,target_reads=0,L_count=SOURCE_L_COUNT,group_count=90,group_counts=counts.tolist(),
            source_known_42_passed=True,individual_source_bank_saved=False,source_L_forward_rows=2*SOURCE_L_COUNT,synthetic_F0_forward_rows=2,
            source_OE_reads=0,source_V_reads=0,source_novel_reads=0,new_neural_updates=0,G_forward_rows=0,channel_generations=0,
            readout_models_stored=4,readout_models_scored_per_batch=4,selected_method='paired_mlp16_step1000',
            model_parameters_buffers_unchanged=True,prior_state_unchanged=True,anchors_ref='anchors.npz',final_phase3_complete=False))
        _write(out/'complete.json',done);print(SOURCE_STATUS,flush=True);return done
    finally:
        if role is not None:close_role(role)


def load_source(root):
    root=Path(root);done=_read(root/'complete.json')
    if (done.get('status')!=SOURCE_STATUS or done.get('schema')!=SOURCE_SCHEMA or done.get('run_id')!=RUN_ID
            or done.get('source_known_42_passed') is not True or done.get('group_count')!=90 or done.get('L_count')!=SOURCE_L_COUNT
            or done.get('new_neural_updates')!=0 or done.get('target_reads')!=0 or done.get('individual_source_bank_saved') is not False):
        raise ValueError('Complete frozen source prior and aggregate-only anchors required')
    with np.load(root/'anchors.npz',allow_pickle=False) as z:
        if set(z.files)!={'embedding','margin','group_keys'}:raise ValueError('Only source group aggregate anchors may deploy')
        anchors={k:z[k].copy() for k in z.files}
    if (anchors['embedding'].shape!=(90,688) or anchors['embedding'].dtype!=np.float32 or anchors['margin'].shape!=(90,)
            or anchors['margin'].dtype!=np.float64 or not np.isfinite(anchors['embedding']).all() or not np.isfinite(anchors['margin']).all()
            or not np.array_equal(anchors['group_keys'],_groups()) or anchors['group_keys'].dtype!=np.int64):raise ValueError('Finite canonical ninety group aggregates required')
    with np.load(root/'prior/thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'global','conditional'}:raise ValueError('Frozen source thresholds required')
        glob=z['global'].copy();cond=z['conditional'].copy()
    if glob.shape!=(1,) or glob.dtype!=np.float64 or cond.shape!=(6,) or cond.dtype!=np.float64 or not np.isfinite(glob).all() or not np.isfinite(cond).all():raise ValueError('Small finite frozen scalar threshold state required')
    return dict(marker=done,anchors=anchors,geometry={p:relative_source._load_geometry(root/'prior/geometry'/(p+'.npz')) for p in base.PATHS},
        readouts=paired.load_readouts(root/'prior/readouts.npz',root/'prior/readouts.json'),
        thresholds=dict(schema=class_core.STATE_SCHEMA,global_threshold=float(glob[0]),conditional_thresholds=cond))


def _target_backend(bundle,c,device):
    model=bundle['marker']['model_config']
    spec=dict(reader_config={k:model[k] for k in ('base_release','frozen_sources','source_output','model_seed')},source_contract=model['source_contract'],
        dura_checkpoint=c['dura_checkpoint'],dura_release=c['dura_release'])
    backend=relative_source._backend(spec,device,with_dura=True)
    if backend['reader_identity']!=bundle['marker']['parent_F0_identity'] or backend['dura_identity']!=bundle['marker']['dura_identity']:
        raise ValueError('Target must use the exact original F0 and cached-path frozen DURA')
    return backend


def _target_evidence(backend,values,prior,device):
    import torch
    from .source_open_export import _tensor
    x=_tensor(values,device);b0,l0=base._encode(backend['reader'],x,device)
    with torch.no_grad():restored=backend['dura'][1](x)[0].detach().clone()
    b1,l1=base._encode(backend['reader'],restored,device)
    return _prior_evidence(b1,l1,b0,l0,prior)


def _target_smoke(prior,backend,device):
    from .star_channel import observe
    mixture=_read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json');time=np.arange(256)
    values=[observe(np.stack((np.cos(w*time),np.sin(w*time))),physical_id='neighborhood-synthetic-'+str(i),receiver='1-19',day='2021_03_08',
        episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture)[0] for i,w in enumerate((.07,.11))]
    z,l,m=_target_evidence(backend,np.asarray(values,np.float32),prior,device)
    if z.shape!=(2,688) or l.shape!=(2,6) or m.shape!=(2,) or not all(np.isfinite(v).all() for v in (z,l,m)):raise ValueError('Actual F0/G paired-prior synthetic smoke failed')
    return dict(status='PASS',actual_F0_G_checkpoints=True,synthetic_G10_rows=2,F0_forward_rows=4,G_forward_rows=2,U_reads=0,Q_reads=0,truth_read=False)


def _open(c,role):return public._open(dict(c,U_expected_count=U_COUNT,Q_expected_count=Q_COUNT),role)


def _frozen(bundle,before,snapshots,state=None,state_before=None):
    _check_models(snapshots)
    if not _same(bundle,before):raise ValueError('Frozen source bundle changed on target')
    if state is not None and not _same(state,state_before):raise ValueError('Frozen U registry changed on future Q')


def _save_state(path,state):core._state(state);_npz(path,{k:np.asarray(v) for k,v in state.items()})


def _load_state(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=STATE_FIELDS:raise ValueError('Only cluster aggregates may be saved in the registry')
        state={k:z[k].copy() for k in z.files}
    if state['schema'].shape!=() or state['schema'].dtype.kind!='U':raise ValueError('Registry schema scalar differs')
    state['schema']=state['schema'].item();core._state(state);return state


def predict(c,device='cuda:0'):
    validate_target_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable target output required')
    bundle=load_source(c['source_root'])
    if bundle['marker']['code_commit']!=c['code_commit']:raise ValueError('Source and target code identity differ')
    before=copy.deepcopy(bundle);backend=_target_backend(bundle,c,device);snapshots=_snapshot(backend)
    smoke=_target_smoke(bundle,backend,device);_frozen(bundle,before,snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Real F0/G synthetic smoke must precede U')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_,Uids,iq=_open(c,'U')
    try:
        zs=[];ls=[];ms=[]
        for start in range(0,len(Uids),c['batch_size']):
            z,l,m=_target_evidence(backend,iq[start:start+c['batch_size']],bundle,device);zs.append(z);ls.append(l);ms.append(m)
        zU,lU,mU=np.concatenate(zs),np.concatenate(ls),np.concatenate(ms)
    finally:
        if isinstance(iq,np.memmap):iq._mmap.close()
    state,diag=core.fit(zU,lU,mU,Uids,bundle['anchors']['embedding'],bundle['anchors']['margin']);core._state(state)
    u_fusion=int(core.fusion_mask(zU,state).sum());k=len(state['anonymous_indices']);adapt=out/'adaptation';adapt.mkdir()
    _save_state(adapt/'state.npz',state)
    diagnostics=dict(ids=Uids,cluster_labels=diag['cluster_labels'],assignments=diag['assignments'],candidate_mask=diag['candidate_mask'],
        cluster_sizes=diag['cluster_sizes'],anonymous_ids=np.asarray(['anon-'+str(i+1).zfill(4) for i in range(k)],dtype=str))
    _npz(adapt/'discovery_diagnostics.npz',diagnostics)
    summary={key:value for key,value in diag.items() if key not in ('cluster_labels','assignments','candidate_mask','cluster_sizes')}
    _write(adapt/'complete.json',dict(status=REGISTRY_STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'],U_count=len(Uids),
        Q_reads=0,truth_read=False,query_fit=False,prototype_counts=dict(DA0=k,DA1=k),discovery_diagnostics=summary,U_fusion_count=u_fusion))
    state_before=copy.deepcopy(state);_frozen(bundle,before,snapshots)
    del zU,lU,mU,zs,ls,ms
    # This is the first Q metadata/IQ open. State is immutable for all future rows.
    _,idsQ,iq=_open(c,'Q')
    try:
        if set(Uids.tolist())&set(idsQ.tolist()):raise ValueError('U/Q physical roles overlap')
        n=len(idsQ);pred=np.empty((4,n),np.int64);closed=np.empty((2,n),np.int64);baseline=np.empty(n,np.int64);known=np.empty((4,n))
        joint={s:np.empty((n,6+(k if s.endswith('REG1') else 0))) for s in STATES};q_fusion=0
        for start in range(0,n,c['batch_size']):
            stop=min(start+c['batch_size'],n);z,l,m=_target_evidence(backend,iq[start:stop],bundle,device);baseline[start:stop]=l.argmax(1)
            q_fusion+=int(core.fusion_mask(z,state).sum())
            for index,name in enumerate(STATES):
                row=core.infer(z,l,m,state,adapt=name.startswith('DA1'),registered=name.endswith('REG1'))
                pred[index,start:stop]=row['prediction'];closed[index%2,start:stop]=row['closed'];joint[name][start:stop]=row['joint_margins']
                known[index,start:stop]=row['joint_margins'].max(axis=1)
            _frozen(bundle,before,snapshots,state,state_before);print('STAR_NEIGHBORHOOD_Q_PROGRESS',stop,n,flush=True)
        p=dict(ids=idsQ,U_ids=Uids,state_names=np.asarray(STATES),prediction=pred,closed_prediction=closed,baseline_closed_prediction=baseline,knownness=known,
            **{'joint_margins_'+s:joint[s] for s in STATES})
        _npz(out/'predictions.npz',p)
        done=dict(status=STATUS,schema=SCHEMA,config=c,run_id=RUN_ID,code_commit=c['code_commit'],condition='P1',view_family='practical_cross_receiver',
            states=list(STATES),primary_state='DA1_REG1',classification_path='P1',rejection_input_paths=['P1','P0'],count=n,U_count=len(Uids),
            prototype_counts=dict(DA0=k,DA1=k),DA1_fusion_applied=q_fusion>0,U_fusion_count=u_fusion,Q_fusion_count=q_fusion,fusion_weight=.5,
            U_day='2021-03-08',Q_day='2021-03-15',truth_read=False,query_fit=False,target_labels_read=0,formal_target_reads=0,
            registry_frozen_before_Q=True,source_frozen_before_U=True,model_parameters_buffers_unchanged=True,frozen_source_aggregates_unchanged=True,
            frozen_target_aggregates_unchanged=True,source_IQ_reads=0,source_loader_reads=0,new_encoder_updates=0,new_identity_head_updates=0,
            new_readout_updates=0,new_threshold_calibrations=0,all_U_rows_used=True,U_pre_rejection_applied=False,
            readout_models_stored=4,readout_models_scored_per_batch=4,selected_method='paired_mlp16_step1000',
            F0_parameters=232993,G_parameters=266951,encoder_models=1,all_new_IQ_satellite=True,auxiliary_E51_read=False,
            prediction_ref='predictions.npz',registry_ref='adaptation/state.npz',diagnostics_ref='adaptation/discovery_diagnostics.npz',
            knownness_scope='actual joint maximum per state; REG1 includes anonymous slots',final_phase3_complete=False)
        _write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        if isinstance(iq,np.memmap):iq._mmap.close()


def _predictions(root):
    root=Path(root);done=_read(root/'complete.json');c=done.get('config');validate_target_config(c)
    if (done.get('status')!=STATUS or done.get('schema')!=SCHEMA or done.get('run_id')!=RUN_ID or done.get('code_commit')!=c['code_commit']
            or done.get('states')!=list(STATES) or done.get('count')!=Q_COUNT or done.get('U_count')!=U_COUNT
            or done.get('truth_read') is not False or done.get('query_fit') is not False or done.get('registry_frozen_before_Q') is not True
            or any(done.get(k)!=0 for k in ('target_labels_read','formal_target_reads','source_IQ_reads','new_encoder_updates','new_identity_head_updates','new_readout_updates','new_threshold_calibrations'))):
        raise ValueError('All four frozen neighborhood predictions must complete before scoring')
    registry=_read(root/'adaptation/complete.json')
    if registry.get('status')!=REGISTRY_STATUS or registry.get('Q_reads')!=0 or registry.get('truth_read') is not False or registry.get('code_commit')!=done['code_commit']:
        raise ValueError('Registry must freeze before any future Q read')
    state=_load_state(root/'adaptation/state.npz');_,k=core._state(state)
    if done.get('prototype_counts')!=dict(DA0=k,DA1=k) or registry.get('prototype_counts')!=done['prototype_counts']:raise ValueError('Shared anonymous registry count differs')
    with np.load(root/'predictions.npz',allow_pickle=False) as z:
        if set(z.files)!=PRED_FIELDS:raise ValueError('Exact four-state prediction fields required')
        p={key:z[key].copy() for key in z.files}
    with np.load(root/'adaptation/discovery_diagnostics.npz',allow_pickle=False) as z:
        if set(z.files)!=DIAG_FIELDS:raise ValueError('Exact unlabeled discovery diagnostics required')
        diag={key:z[key].copy() for key in z.files}
    for key,n in (('ids',Q_COUNT),('U_ids',U_COUNT)):
        if p[key].shape!=(n,) or p[key].dtype.kind!='U' or len(set(p[key].tolist()))!=n or np.any(np.char.strip(p[key])==''):raise ValueError('Unique opaque public IDs required')
    if set(p['ids'].tolist())&set(p['U_ids'].tolist()) or not np.array_equal(diag['ids'],p['U_ids']):raise ValueError('U/Q identities overlap or diagnostics misalign')
    if (p['state_names'].tolist()!=list(STATES) or p['prediction'].shape!=(4,Q_COUNT) or p['prediction'].dtype!=np.int64
            or p['closed_prediction'].shape!=(2,Q_COUNT) or p['closed_prediction'].dtype!=np.int64 or p['baseline_closed_prediction'].shape!=(Q_COUNT,)
            or p['baseline_closed_prediction'].dtype!=np.int64 or p['knownness'].shape!=(4,Q_COUNT) or p['knownness'].dtype!=np.float64
            or not np.isfinite(p['knownness']).all() or np.any((p['closed_prediction']<0)|(p['closed_prediction']>=6))
            or not np.array_equal(p['closed_prediction'][0],p['baseline_closed_prediction'])):raise ValueError('Aligned per-DA original-head identities and joint evidence required')
    for index,name in enumerate(STATES):
        a=p['joint_margins_'+name];width=6+(k if name.endswith('REG1') else 0)
        if (a.shape!=(Q_COUNT,width) or a.dtype!=np.float64 or not np.isfinite(a).all() or not np.array_equal(a.max(1),p['knownness'][index])
            or not np.array_equal(np.where(a.max(1)>=0,a.argmax(1),-1),p['prediction'][index]) or not np.array_equal(a[:,:6].argmax(1),p['closed_prediction'][index%2])):
            raise ValueError('Joint competition, evidence and actual decisions differ')
    if (diag['cluster_labels'].shape!=(U_COUNT,) or diag['cluster_labels'].dtype!=np.int64 or diag['assignments'].shape!=(U_COUNT,)
            or diag['assignments'].dtype!=np.int64 or diag['candidate_mask'].shape!=(U_COUNT,) or diag['candidate_mask'].dtype!=bool
            or np.any((diag['assignments']<-1)|(diag['assignments']>=k)) or not np.array_equal(diag['candidate_mask'],diag['assignments']>=0)
            or diag['cluster_sizes'].shape!=(len(state['cluster_embedding']),) or diag['cluster_sizes'].dtype!=np.int64
            or diag['anonymous_ids'].shape!=(k,) or diag['anonymous_ids'].dtype.kind!='U'):raise ValueError('U-only shared anonymous assignment schema differs')
    return done,p,diag,state


def main(argv=None):
    print(json.dumps(dict(event='STAR_NEIGHBORHOOD_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for name in ('source','target'):
        command=sub.add_parser(name);command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);c=_read(args.config);done=source(c,args.device) if args.action=='source' else predict(c,args.device)
    print(json.dumps(dict(status=done['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
