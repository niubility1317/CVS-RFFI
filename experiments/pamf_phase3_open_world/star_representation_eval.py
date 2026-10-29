"""Frozen post-training source statistics and truth-blind star predictions.

Only cache L, V and source-heldout novel are opened. L fits all geometry and
score normalization; known V selects global thresholds against ORIGINAL F0
cache baselines. The novel role is opened only after both paths are frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import source_open_readout as geometry_core
from . import source_practical_readout as risk
from . import star_relative_core as core
from . import star_relative_source as serialization
from .development_transfer import _snapshot_model, _check_model

RUN_ID = '20261029-phase3-star-representation-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_representation_eval_v1'
STATUS = 'STAR_REPRESENTATION_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS = 'STAR_REPRESENTATION_SOURCE_THRESHOLDS_FROZEN'
CACHE_ROLE_SCHEMA = 'pamf_phase3_star_representation_cache_role_v1'
CACHE_ROLE_STATUS = 'STAR_REPRESENTATION_ROLE_COMPLETE'
ARMS = ('anchor','oe','oe_vic')
PATHS = ('P0','P1')
METHODS = core.METHODS
PRIMARY_METHOD = 'energy'
PRIMARY_PATH = 'P1'
PRIMARY_ARM = 'oe_vic'
COUNTS = dict(L=6300,V=27000,novel=3150)
RECEIVERS, SCENES = serialization.RECEIVERS, serialization.SCENES
FIELDS = {'run_id','code_commit','arm','train_config','checkpoint','cache_root','output_root','batch_size'}
TRAIN_FIELDS = {'run_id','code_commit','arm','native_config','L_root','OE_root','output_root','steps',
                'lr','L_seed','OE_seed','view_seed'}
KNOWN_FIELDS = {'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed'}
NOVEL_FIELDS = {'ids','rx_i','day_i','scores','closed','fused_unit'}
_write, _read, _npz = serialization._write, serialization._read, serialization._npz
_save_geometry, _save_metric = serialization._save_geometry, serialization._save_metric


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or config.get('arm') not in ARMS or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or any(not isinstance(config[k],str) or not config[k].strip() for k in ('checkpoint','cache_root','output_root'))):
        raise ValueError('Only the fixed source-only eight-field evaluation configuration is accepted')
    train=config['train_config']
    if (not isinstance(train,dict) or set(train)!=TRAIN_FIELDS or any(train.get(k)!=config[k] for k in ('run_id','code_commit','arm'))
            or train.get('steps')!=2220):
        raise ValueError('The matching arm/run/code and 2220-step source training configuration is required')
    return config


def _backend(config,device):
    from .star_representation_train import load_trained
    model=load_trained(config['train_config'],config['checkpoint'],device)
    identity=getattr(model,'_star_representation_identity',None)
    if (not isinstance(identity,dict) or any(identity.get(k)!=config[k] for k in ('run_id','arm','code_commit','checkpoint'))
            or identity.get('updates')!=2220 or identity.get('cumulative_updates')!=13320):
        raise ValueError('Only this arm actual frozen trained checkpoint identity may be evaluated')
    return model,model.id_backbone.encoder,identity


def _load_role(root,role):
    from .star_representation_cache import load_role
    return load_role(Path(root)/role,role)


def _close_role(bundle):
    from .star_representation_cache import close_role
    close_role(bundle)


def _ids(ids,n):
    return (ids.shape==(n,) and ids.dtype.kind=='U' and len(set(ids.tolist()))==n
            and not np.any(np.char.strip(ids)==''))


def _validate_role(bundle,role):
    if role not in COUNTS or not isinstance(bundle,dict):
        raise ValueError('Evaluation consumes only L, V and source-heldout novel, never OE')
    n=COUNTS[role];manifest=bundle.get('manifest',{})
    allowed={'manifest','ids','received','restored','metadata'}|({'teacher_logits'} if role=='L' else {'baseline_closed'} if role=='V' else set())
    if (set(bundle)!=allowed or manifest.get('schema')!=CACHE_ROLE_SCHEMA or manifest.get('status')!=CACHE_ROLE_STATUS
            or manifest.get('run_id')!=RUN_ID or manifest.get('role')!=role or manifest.get('count')!=n
            or manifest.get('iq_shape')!=[n,2,256] or manifest.get('paths')!=list(PATHS)
            or manifest.get('source_only') is not True or manifest.get('all_new_IQ_satellite') is not True
            or not _ids(np.asarray(bundle['ids']),n)):
        raise ValueError('Matching completed star-only cache role, unique IDs and fixed count required')
    for key in ('received','restored'):
        a=bundle[key]
        if a.shape!=(n,2,256) or a.dtype!=np.float32:raise ValueError('Cache observation shape/dtype differs')
    meta=bundle['metadata'];fields={'rx_i','day_i','domain'}|({'y'} if role!='novel' else set())|({'scenario'} if role=='V' else set())
    if not isinstance(meta,dict) or set(meta)!=fields:raise ValueError('Novel labels/scenes/private truth are forbidden in evaluation')
    for key in fields-{'scenario'}:
        if meta[key].shape!=(n,) or meta[key].dtype!=np.int64:raise ValueError('Aligned int64 source role metadata required')
    if (not np.isin(meta['rx_i'],RECEIVERS).all() or not np.isin(meta['day_i'],(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(RECEIVERS,meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Only original source RX/day/domain metadata may enter source evaluation')
    if role!='novel' and (np.any((meta['y']<0)|(meta['y']>=6)) or set(meta['y'].tolist())!=set(range(6))):
        raise ValueError('Source known labels must contain the old six classes only')
    if role=='L' and (bundle['teacher_logits'].shape!=(n,6) or bundle['teacher_logits'].dtype!=np.float32):
        raise ValueError('The complete L cache contract is required; teacher logits are not evaluation inputs')
    if role=='V':
        baseline=bundle['baseline_closed'];scene=meta['scenario']
        if (baseline.shape!=(n,2) or baseline.dtype!=np.int64 or np.any((baseline<0)|(baseline>=6))
                or scene.shape!=(n,) or scene.dtype.kind!='U' or not np.isin(scene,SCENES).all()):
            raise ValueError('Original F0 per-path closed baselines and known V scenarios are required')
    return bundle


def _encode(reader,values,device):
    import torch
    from .source_open_export import _tensor,extract_branches
    with torch.no_grad():
        x=_tensor(values,device)
        branches,logits=extract_branches(reader,x)
        return ({name:np.asarray(value.detach().cpu().tolist(),np.float32) for name,value in branches.items()},
                np.asarray(logits.detach().cpu().tolist(),np.float32))


def _branches(reader,values,device,batch_size):
    pieces={};raw=[]
    for start in range(0,len(values),batch_size):
        branches,logits=_encode(reader,values[start:start+batch_size],device)
        for name,value in branches.items():pieces.setdefault(name,[]).append(value)
        raw.append(logits)
    return {k:np.concatenate(v) for k,v in pieces.items()},np.concatenate(raw)


def _check_states(states,before):
    if not serialization._same(states,before):raise ValueError('L-frozen geometry, metric or thresholds changed after fit')


def calibrate_known(ids,y,closed,baseline,scores,rx,scenario):
    """One maximal legal scalar per score; ORIGINAL F0 baselines, intact ties."""
    ids,y,closed,baseline,scores,rx,scenario=map(np.asarray,(ids,y,closed,baseline,scores,rx,scenario));n=len(ids)
    if (not _ids(ids,n) or scores.shape!=(n,7) or scores.dtype.kind!='f' or not np.isfinite(scores).all()
            or any(a.shape!=(n,) or a.dtype.kind not in 'iu' for a in (y,closed,baseline,rx))
            or any(np.any((a<0)|(a>=6)) for a in (y,closed,baseline)) or not np.isin(rx,RECEIVERS).all()
            or scenario.shape!=(n,) or scenario.dtype.kind!='U' or not np.isin(scenario,SCENES).all()):
        raise ValueError('Only aligned source known V labels, new closed and original F0 baseline may calibrate')
    masks=list(serialization._masks(rx,scenario));coverage={key:set(y[mask].tolist())==set(range(6)) for key,mask in masks}
    methods={};thresholds=[]
    for j,method in enumerate(METHODS):
        values=scores[:,j]
        local={key:risk._local_maximum(y[mask],closed[mask],baseline[mask],values[mask]) for key,mask in masks if coverage[key]}
        feasible=all(coverage.values()) and all(row[1] for row in local.values())
        threshold=min(row[0] for row in local.values()) if feasible else float(values.min())
        groups={}
        for key,mask in masks:
            groups[key]=dict(risk._risk_group(y[mask],closed[mask],baseline[mask],values[mask],threshold),certified=True) if coverage[key] else dict(
                certified=False,constraints_passed=False,reason='missing_known_class',known_count=int(mask.sum()),
                class_counts=np.bincount(y[mask],minlength=6).tolist())
        levels=np.unique(values);i=int(np.searchsorted(levels,threshold))
        with np.errstate(over='ignore'):following=float(levels[i+1]) if i+1<len(levels) else float(np.nextafter(levels[-1],np.inf))
        next_pass=feasible and np.isfinite(following) and all(risk._risk_group(y[mask],closed[mask],baseline[mask],values[mask],following)['constraints_passed'] for _,mask in masks)
        methods[method]=dict(threshold=float(threshold),no_rejection_feasible=bool(feasible),zero_rejection_fallback=not feasible,
            constraints_passed=all(row['constraints_passed'] for row in groups.values()),all_groups_class_complete=all(coverage.values()),
            groups=groups,local_maximum_thresholds={key:value[0] for key,value in local.items()},
            next_candidate_threshold=following if np.isfinite(following) else None,next_candidate_passed=bool(next_pass))
        thresholds.append(threshold)
    return np.asarray(thresholds,np.float64),dict(schema=SCHEMA,status=CALIBRATION_STATUS,method_names=list(METHODS),
        known_count=n,group_count=42,methods=methods,constraints_passed=all(row['constraints_passed'] for row in methods.values()),
        source_only=True,target_reads=0,source_V_labels_used=True,source_V_metric_fit=False,source_V_score_normalization_fit=False,
        unknown_inputs_used=False,baseline='cache original F0 same-input same-path complete head',
        limits=dict(known_macro_frr=.05,original_closed_minus_new_open=.02),
        acceptance_rule='score >= maximal legal global scalar; intact ties; infeasible/missing coverage accepts all and stays false')


def _predict_role(reader,bundle,states,device,batch_size,keep_features=False):
    n=len(bundle['ids']);scores=np.empty((n,2,7),np.float64);closed=np.empty((n,2),np.int64)
    fused=np.empty((n,2,160),np.float32) if keep_features else None
    for pi,path in enumerate(PATHS):
        values=bundle['received' if path=='P0' else 'restored'];state=states[path]
        for start in range(0,n,batch_size):
            end=min(start+batch_size,n);branches,logits=_encode(reader,values[start:end],device)
            if logits.shape!=(end-start,6) or not np.isfinite(logits).all():raise ValueError('Complete six-class head predictions required')
            score=core.all_scores(branches,logits,state['geometry'],state['metric'])
            if score.shape!=(end-start,7) or not np.isfinite(score).all():raise ValueError('Seven finite frozen scores required')
            scores[start:end,pi]=score;closed[start:end,pi]=logits.argmax(axis=1)
            if keep_features:
                x=np.asarray(branches['fused160'],np.float64);norm=np.linalg.norm(x,axis=1,keepdims=True)
                if np.any(norm<1e-12):raise ValueError('Nonzero source-novel fused embeddings required for retrieval diagnostic')
                fused[start:end,pi]=(x/norm).astype(np.float32)
        print('STAR_REPRESENTATION_EVAL_PROGRESS',bundle['manifest']['role'],path,n,flush=True)
    return scores,closed,fused


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable arm source-evaluation root is required')
    started=time.perf_counter();model,reader,identity=_backend(config,device);snapshot=_snapshot_model(model)
    output.mkdir(parents=True,exist_ok=False);states={};cache_identities=None;role_ids={}
    bundle=_load_role(config['cache_root'],'L')
    try:
        _validate_role(bundle,'L');manifest=bundle['manifest'];role_ids['L']=set(bundle['ids'].tolist())
        cache_identities={key:manifest[key] for key in ('parent_F0_identity','dura_identity')}
        for path in PATHS:
            values=bundle['received' if path=='P0' else 'restored']
            branches,_=_branches(reader,values,device,config['batch_size'])
            geometry=geometry_core.fit_geometry(branches,bundle['metadata']['y'])
            metric=core.fit_metric(branches,bundle['metadata']['y'],geometry)
            states[path]=dict(geometry=geometry,metric=metric)
            row=output/'by_path'/path;row.mkdir(parents=True)
            _save_geometry(row/'geometry.npz',geometry);_save_metric(row/'metric.npz',metric)
            _check_model(model,snapshot)
        del branches
    finally:_close_role(bundle)
    frozen_L=copy.deepcopy(states);thresholds={}
    bundle=_load_role(config['cache_root'],'V')
    try:
        _validate_role(bundle,'V');meta=bundle['metadata'];role_ids['V']=set(bundle['ids'].tolist())
        if role_ids['L']&role_ids['V'] or any(bundle['manifest'].get(k)!=v for k,v in cache_identities.items()):
            raise ValueError('L/V IDs must be separate and frozen F0/G cache identities must agree')
        scores,closed,_=_predict_role(reader,bundle,states,device,config['batch_size'])
        _check_states(states,frozen_L);_check_model(model,snapshot)
        for pi,path in enumerate(PATHS):
            thresholds[path],calibration=calibrate_known(bundle['ids'],meta['y'],closed[:,pi],bundle['baseline_closed'][:,pi],
                scores[:,pi],meta['rx_i'],meta['scenario'])
            row=output/'by_path'/path;(row/'calibration').mkdir()
            with (row/'thresholds.npy').open('xb') as stream:np.save(stream,thresholds[path],allow_pickle=False)
            _write(row/'calibration/complete.json',dict(calibration,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],path=path))
        _npz(output/'known_predictions.npz',dict(ids=bundle['ids'],y=meta['y'],rx_i=meta['rx_i'],day_i=meta['day_i'],
            scenario=meta['scenario'],scores=scores,closed=closed,baseline_closed=bundle['baseline_closed']))
    finally:_close_role(bundle)
    _check_states(states,frozen_L);_check_model(model,snapshot)
    frozen_deployment=copy.deepcopy(dict(states=states,thresholds=thresholds))
    # First novel access: both path geometries, score scales and all 14 taus are frozen and saved.
    bundle=_load_role(config['cache_root'],'novel')
    try:
        _validate_role(bundle,'novel');meta=bundle['metadata']
        if set(bundle['ids'].tolist())&(role_ids['L']|role_ids['V']) or any(bundle['manifest'].get(k)!=v for k,v in cache_identities.items()):
            raise ValueError('Novel physical IDs must be separate and match the same F0/G observation cache')
        scores,closed,fused=_predict_role(reader,bundle,states,device,config['batch_size'],True)
        _check_states(dict(states=states,thresholds=thresholds),frozen_deployment);_check_model(model,snapshot)
        _npz(output/'novel_predictions.npz',dict(ids=bundle['ids'],rx_i=meta['rx_i'],day_i=meta['day_i'],
            scores=scores,closed=closed,fused_unit=fused))
    finally:_close_role(bundle)
    result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],model_identity=identity,
        path_names=list(PATHS),method_names=list(METHODS),primary_method=PRIMARY_METHOD,primary_path=PRIMARY_PATH,
        deployment_primary_arm=PRIMARY_ARM,known_count=COUNTS['V'],novel_count=COUNTS['novel'],L_physical_count=COUNTS['L'],
        source_only=True,target_reads=0,target_labels_read=0,source_novel_truth_reads=0,source_U_records_consumed=0,
        source_OE_records_consumed=0,eval_neural_updates=0,G_forward_rows=0,
        F_forward_rows=2*sum(COUNTS.values()),same_cached_received_restored_for_all_methods=True,
        source_L_labels_used_for_statistics=True,source_V_labels_used_for_thresholds=True,source_V_metric_fit=False,
        model_parameters_buffers_unchanged=True,L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
        baseline='cache original F0 same-input same-path complete head',cache_identities=cache_identities,
        novel_fused_features_scope='source truth-side cross-RX 1NN diagnostic only; not a deployed source bank',
        source_bank_deployed=False,source_IQ_saved=False,final_phase3_complete=False,A_N=None,H_OW=None,
        seconds=time.perf_counter()-started)
    _write(output/'complete.json',result);print(STATUS,config['arm'],flush=True)
    return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_REPRESENTATION_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('run');command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);run(_read(args.config),args.device);return 0


if __name__=='__main__':raise SystemExit(main())
