"""Frozen r02 OVA evidence with source-OE risk-budget thresholds only.

The checkpoint smoke precedes new source rows. Known V and OE may calibrate;
parent novel arrays are opened only after both methods' thresholds freeze.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import star_joint_ova_eval as parent
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261030-phase3-star-oe-risk-budget-s2026092701-r01'
PARENT_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
PARENT_COMMIT='5aa041122396d48c49b8cb3d5912649bdc3217e1'
SCHEMA='pamf_phase3_star_oe_risk_prediction_v1'
STATUS='STAR_OE_RISK_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_OE_RISK_THRESHOLDS_FROZEN'
ARMS=parent.ARMS
METHODS=('common_tail','oe_risk_budget')
PRIMARY_ARM='joint';PRIMARY_METHOD='oe_risk_budget'
FIELDS={'run_id','code_commit','arm','parent_eval_config','parent_prediction_root','OE_root','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS=parent.KNOWN_FIELDS,parent.NOVEL_FIELDS
OE_FIELDS={'ids','rx_i','day_i','domain','closed','score','default_logits','aux_logits'}
COUNTS=dict(known=27000,novel=3150,OE=6134)
RECEIVERS,SCENES,GROUP_NAMES=parent.RECEIVERS,parent.SCENES,parent.common.GROUP_NAMES
_read,_write,_npz,_ids=parent._read,parent._write,parent._npz,parent._ids


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID or config.get('arm') not in ARMS
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('parent_prediction_root','OE_root','output_root'))):
        raise ValueError('Only the fixed eight-field source OE risk configuration is allowed; no truth or target inputs')
    old=config['parent_eval_config'];parent.validate_config(old)
    if (old['run_id']!=PARENT_RUN or old['code_commit']!=PARENT_COMMIT or old['arm']!=config['arm']
            or Path(old['output_root']).resolve()!=Path(config['parent_prediction_root']).resolve()
            or Path(old['train_config']['OE_root']).resolve()!=Path(config['OE_root']).resolve()):
        raise ValueError('Exact frozen r02 arm/code, saved prediction root and original source OE handle required')
    out=Path(config['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) for p in (config['parent_prediction_root'],config['OE_root'],old['train_config']['output_root'])):
        raise ValueError('New output cannot overwrite parent predictions, OE cache or trained state')
    return config


def _parent_marker(config):
    d=_read(Path(config['parent_prediction_root'])/'complete.json')
    if (d.get('schema')!=parent.SCHEMA or d.get('status')!=parent.STATUS or d.get('run_id')!=PARENT_RUN
            or d.get('code_commit')!=PARENT_COMMIT or d.get('arm')!=config['arm'] or d.get('method_names')!=list(parent.METHODS)
            or d.get('classification_path')!='P1' or d.get('source_only') is not True or d.get('target_reads')!=0
            or d.get('source_novel_truth_reads')!=0 or d.get('known_count')!=COUNTS['known'] or d.get('novel_count')!=COUNTS['novel']
            or any(d.get(k) is not True for k in ('all_thresholds_frozen_before_novel','model_parameters_buffers_unchanged','aux_parameters_buffers_unchanged'))):
        raise ValueError('Completed frozen source-only r02 parent predictions required')
    return d


def _parent_calibration(root,marker):
    root=Path(root);cal=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
    report=cal.get('methods',{}).get('ova_max',{})
    if (cal.get('schema')!=parent.SCHEMA or cal.get('status')!=parent.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit','arm')) or cal.get('known_only') is not True
            or cal.get('group_count')!=42 or tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or not np.array_equal(tau,cal.get('conditional_thresholds'))
            or not np.array_equal(tau[2],report.get('conditional_thresholds')) or set(report.get('groups',{}))!=set(GROUP_NAMES)):
        raise ValueError('Actual r02 ova_max thresholds/report must be reused exactly, never recalibrated')
    return tau,deepcopy(report)


def _load_parent_predictions(role,root,thresholds):
    if role not in ('known','novel'):raise ValueError('Only fixed public known/novel prediction arrays may be reused')
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS;n=COUNTS[role]
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Exact public parent fields required; private novel labels forbidden')
        d={k:z[k].copy() for k in z.files}
    if not _ids(d['ids'],n):raise ValueError('Unique aligned parent opaque IDs required')
    for k in ('default_logits','aux_logits'):
        if d[k].shape!=(n,6) or d[k].dtype!=np.float32 or not np.isfinite(d[k]).all():raise ValueError('Finite public parent six-logit evidence required')
    raw,closed=parent.scalars(d['default_logits'],d['aux_logits'])
    if (d['closed'].shape!=(n,) or d['closed'].dtype!=np.int64 or not np.array_equal(d['closed'],closed)
            or d['scores'].shape!=(n,3) or d['scores'].dtype!=np.float64 or not np.isfinite(d['scores']).all()
            or not np.allclose(d['scores'],raw-thresholds[:,closed].T,rtol=1e-12,atol=1e-12)):
        raise ValueError('Parent default-head identity and frozen energy/routed/max margins must agree')
    for k,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
        if d[k].shape!=(n,) or d[k].dtype!=np.int64 or not np.isin(d[k],roster).all():raise ValueError('Original source RX/day metadata required')
    if role=='known':
        if (any(d[k].shape!=(n,) or d[k].dtype!=np.int64 or not np.isin(d[k],range(6)).all() for k in ('y','baseline_closed'))
                or d['scenario'].shape!=(n,) or d['scenario'].dtype.kind!='U' or not np.isin(d['scenario'],SCENES).all()):
            raise ValueError('Known V labels, original F0 baseline and source scenes required')
    else:
        for k in ('candidate_fused_unit','baseline_fused_unit'):
            if d[k].shape!=(n,160) or d[k].dtype!=np.float32 or not np.isfinite(d[k]).all():raise ValueError('Parent public source structure features required')
    return d


def _load_OE(root):
    from .star_representation_cache import load_role
    return load_role(root,'OE')


def _close_OE(bundle):
    from .star_representation_cache import close_role
    close_role(bundle)


def _OE(bundle,marker):
    m=bundle['manifest'];meta=bundle['metadata'];n=COUNTS['OE'];identities=marker['cache_identities']
    if (set(meta)!={'rx_i','day_i','domain'} or m.get('role')!='OE' or m.get('count')!=n
            or m.get('parent_F0_identity')!=identities['parent_F0_identity'] or m.get('dura_identity')!=identities['dura_identity']
            or not _ids(bundle['ids'],n) or bundle['restored'].shape!=(n,2,256) or bundle['restored'].dtype!=np.float32):
        raise ValueError('Only the matching completed public P1 source OE role is permitted; no TX/K')
    if any(meta[k].shape!=(n,) or meta[k].dtype!=np.int64 for k in meta):raise ValueError('Aligned integer source OE metadata required')
    rx,day=meta['rx_i'],meta['day_i']
    if (not np.isin(rx,RECEIVERS).all() or not np.isin(day,(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(RECEIVERS,rx)*3+day-1)
            or set(meta['domain'].tolist())!=set(range(15))):raise ValueError('All fifteen source RX/day domains required')


def _calibrate(*args):
    from .star_oe_risk_budget import calibrate
    return calibrate(*args)


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('New immutable source OE-risk output required')
    started=time.perf_counter();marker=_parent_marker(config);old=config['parent_eval_config']
    model,reader,aux,identity=parent._backend(old,device)
    if identity!=marker.get('model_identity'):raise ValueError('Actual frozen checkpoint must match the exact saved r02 model identity')
    snapshots=(_snapshot_model(model),_snapshot_model(aux))
    def check():_check_model(model,snapshots[0]);_check_model(aux,snapshots[1])
    smoke=parent._smoke(reader,aux,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Real-checkpoint no-query synthetic smoke must pass before source rows')
    output.mkdir(parents=True);_write(output/'smoke.json',smoke)
    bundle=_load_OE(config['OE_root'])
    try:
        _OE(bundle,marker);old_tau,common=_parent_calibration(config['parent_prediction_root'],marker)
        known=_load_parent_predictions('known',config['parent_prediction_root'],old_tau)
        if set(known['ids'].tolist())&set(bundle['ids'].tolist()):raise ValueError('Source OE and known V physical IDs must be disjoint')
        z,g,_=parent._predict(reader,aux,bundle['restored'],device,config['batch_size']);check();raw,closed=parent.scalars(z,g)
        meta=bundle['metadata'];oe=dict(ids=bundle['ids'].copy(),rx_i=meta['rx_i'].copy(),day_i=meta['day_i'].copy(),domain=meta['domain'].copy(),
            closed=closed,score=raw[:,2],default_logits=z,aux_logits=g)
    finally:_close_OE(bundle)
    raw_known=known['aux_logits'].astype(np.float64).max(1)
    state,report=_calibrate(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw_known,known['rx_i'],known['scenario'],
        oe['ids'],oe['closed'],oe['score'],oe['domain'])
    selected=np.asarray(state['conditional_thresholds'])
    if (selected.shape!=(6,) or selected.dtype!=np.float64 or not np.isfinite(selected).all()
            or not np.array_equal(selected,report.get('conditional_thresholds')) or set(report.get('groups',{}))!=set(GROUP_NAMES)):
        raise ValueError('Six finite OE-budget thresholds and all42 actual risk reports required')
    thresholds=np.stack((old_tau[2],selected));check();_npz(output/'thresholds.npz',dict(conditional=thresholds))
    cal=output/'calibration';cal.mkdir()
    _write(cal/'complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],
        method_names=list(METHODS),conditional_thresholds=thresholds.tolist(),methods=dict(common_tail=common,oe_risk_budget=report),group_count=42,
        source_V_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,known_only=False,
        novel_inputs_used=False,target_reads=0,source_only=True,source_V_neural_fit=False,neural_updates=0))
    _npz(output/'OE_scores.npz',oe)
    known['scores']=raw_known[:,None]-thresholds[:,known['closed']].T;_npz(output/'known_predictions.npz',known)
    frozen=thresholds.copy();check();print(CALIBRATION_STATUS,config['arm'],flush=True)
    novel=_load_parent_predictions('novel',config['parent_prediction_root'],old_tau)
    if set(novel['ids'].tolist())&(set(known['ids'].tolist())|set(oe['ids'].tolist())):raise ValueError('Source novel, V and OE physical roles must be disjoint')
    novel['scores']=novel['aux_logits'].astype(np.float64).max(1)[:,None]-thresholds[:,novel['closed']].T
    check()
    if not np.array_equal(frozen,thresholds):raise ValueError('Thresholds changed after freeze')
    _npz(output/'novel_predictions.npz',novel)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],method_names=list(METHODS),
        primary_arm=PRIMARY_ARM,primary_method=PRIMARY_METHOD,classification_path='P1',score_threshold=0.,source_only=True,target_reads=0,
        parent_run_id=PARENT_RUN,parent_code_commit=PARENT_COMMIT,parent_model_identity=identity,parent_prediction_root=config['parent_prediction_root'],
        known_count=COUNTS['known'],novel_count=COUNTS['novel'],OE_count=COUNTS['OE'],all_thresholds_frozen_before_novel=True,
        model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True,neural_updates=0,F_forward_rows=COUNTS['OE']+2,G_forward_rows=0,
        real_P1_forward_rows=COUNTS['OE'],synthetic_P0_forward_rows=2,source_L_reads=0,source_U_reads=0,source_novel_truth_reads=0,
        new_V_forward_rows=0,new_novel_forward_rows=0,new_original_F0_forward_rows=0,source_OE_membership_used=True,source_OE_TX_or_K_used=False,
        source_V_neural_fit=False,cache_identities=marker['cache_identities'],final_phase3_complete=False,A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(output/'complete.json',result);print(STATUS,config['arm'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_OE_RISK_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0')
    a=parser.parse_args(argv);run(_read(a.config),a.device);return 0


if __name__=='__main__':raise SystemExit(main())
