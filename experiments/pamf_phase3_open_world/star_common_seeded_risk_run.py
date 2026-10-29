"""Common-initialized source risk budgets using frozen public evidence only.

Apart from two synthetic checkpoint smoke rows, this process consumes no IQ
and performs no neural forward. Novel public arrays remain unopened until
the three policies' thresholds and known predictions have been written.
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

from . import star_oe_risk_run as previous

parent=previous.parent
RUN_ID='20261030-phase3-star-common-seeded-risk-s2026092701-r01'
SCHEMA='pamf_phase3_star_common_seeded_risk_prediction_v1'
STATUS='STAR_COMMON_SEEDED_RISK_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_COMMON_SEEDED_RISK_THRESHOLDS_FROZEN'
PARENT_RUN=previous.PARENT_RUN;PARENT_COMMIT=previous.PARENT_COMMIT
OE_PARENT_RUN='20261030-phase3-star-oe-risk-budget-s2026092701-r01'
OE_PARENT_COMMIT='585e10b07be1f6821edea08b4853582b92512aff'
ARMS=previous.ARMS
METHODS=('common_tail','zero_initialized_budget','common_initialized_budget')
PRIMARY_ARM='joint';PRIMARY_METHOD='common_initialized_budget'
FIELDS={'run_id','code_commit','arm','parent_eval_config','parent_prediction_root','OE_prediction_root','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,OE_FIELDS=previous.KNOWN_FIELDS,previous.NOVEL_FIELDS,previous.OE_FIELDS
COUNTS=previous.COUNTS
RECEIVERS,SCENES,GROUP_NAMES=previous.RECEIVERS,previous.SCENES,previous.GROUP_NAMES
_read,_write,_npz,_ids=previous._read,previous._write,previous._npz,previous._ids
_load_parent_predictions=previous._load_parent_predictions
_snapshot_model,_check_model=previous._snapshot_model,previous._check_model


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID or config.get('arm') not in ARMS
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('parent_prediction_root','OE_prediction_root','output_root'))):
        raise ValueError('Only eight public-evidence fields are allowed; no IQ, truth or target handle')
    old=config['parent_eval_config'];parent.validate_config(old)
    if (old['run_id']!=PARENT_RUN or old['code_commit']!=PARENT_COMMIT or old['arm']!=config['arm']
            or Path(old['output_root']).resolve()!=Path(config['parent_prediction_root']).resolve()):
        raise ValueError('Exact original r02 source arm/code and saved prediction root required')
    output=Path(config['output_root']).resolve()
    if any(output.is_relative_to(Path(p).resolve()) for p in (config['parent_prediction_root'],config['OE_prediction_root'],old['train_config']['output_root'])):
        raise ValueError('New output cannot overwrite parent public evidence or checkpoint')
    return config


def _load_OE_predictions(root,marker,identity,common):
    """Read only previous marker/calibration/thresholds/OE_scores, never novel."""
    root=Path(root);done=_read(root/'complete.json')
    if (done.get('schema')!=previous.SCHEMA or done.get('status')!=previous.STATUS or done.get('run_id')!=OE_PARENT_RUN
            or done.get('code_commit')!=OE_PARENT_COMMIT or done.get('arm')!=marker['arm']
            or done.get('method_names')!=list(previous.METHODS) or done.get('classification_path')!='P1'
            or done.get('source_only') is not True or done.get('target_reads')!=0 or done.get('neural_updates')!=0
            or done.get('source_novel_truth_reads')!=0 or done.get('parent_run_id')!=PARENT_RUN or done.get('parent_code_commit')!=PARENT_COMMIT
            or done.get('parent_model_identity')!=identity or done.get('cache_identities')!=marker.get('cache_identities')
            or any(done.get(k)!=COUNTS[v] for k,v in (('known_count','known'),('novel_count','novel'),('OE_count','OE')))
            or any(done.get(k) is not True for k in ('all_thresholds_frozen_before_novel','model_parameters_buffers_unchanged','aux_parameters_buffers_unchanged'))):
        raise ValueError('Matching completed source-only OE parent and exact original checkpoint identity required')
    cal=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('The two saved parent conditional policies are required')
        tau=z['conditional'].copy()
    report=cal.get('methods',{}).get('oe_risk_budget',{})
    if (cal.get('schema')!=previous.SCHEMA or cal.get('status')!=previous.CALIBRATION_STATUS
            or any(cal.get(k)!=done[k] for k in ('run_id','code_commit','arm')) or cal.get('method_names')!=list(previous.METHODS)
            or cal.get('group_count')!=42 or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or cal.get('source_OE_membership_used') is not True or cal.get('source_OE_TX_or_K_used') is not False
            or cal.get('novel_inputs_used') is not False or tau.shape!=(2,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or not np.array_equal(tau,cal.get('conditional_thresholds')) or not np.array_equal(tau[0],common)
            or not np.array_equal(tau[1],report.get('conditional_thresholds')) or set(report.get('groups',{}))!=set(GROUP_NAMES)):
        raise ValueError('Exact saved common/zero thresholds and source-only calibration required')
    with np.load(root/'OE_scores.npz',allow_pickle=False) as z:
        if set(z.files)!=OE_FIELDS:raise ValueError('Only public OE scores, IDs and source domains may be read; no TX/K')
        data={k:z[k].copy() for k in z.files}
    n=COUNTS['OE']
    if not _ids(data['ids'],n):raise ValueError('Complete unique source OE opaque IDs required')
    for k in ('default_logits','aux_logits'):
        if data[k].shape!=(n,6) or data[k].dtype!=np.float32 or not np.isfinite(data[k]).all():raise ValueError('Finite frozen OE logits required')
    if (data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64 or not np.array_equal(data['closed'],data['default_logits'].argmax(1))
            or data['score'].shape!=(n,) or data['score'].dtype!=np.float64 or not np.isfinite(data['score']).all()
            or not np.array_equal(data['score'],data['aux_logits'].astype(np.float64).max(1))):
        raise ValueError('OE raw max and unchanged default identity must agree with public logits')
    if any(data[k].shape!=(n,) or data[k].dtype!=np.int64 for k in ('rx_i','day_i','domain')):
        raise ValueError('Aligned source RX/day/domain metadata required')
    rx,day=data['rx_i'],data['day_i']
    if (not np.isin(rx,RECEIVERS).all() or not np.isin(day,(1,2,3)).all()
            or not np.array_equal(data['domain'],np.searchsorted(RECEIVERS,rx)*3+day-1) or set(data['domain'].tolist())!=set(range(15))):
        raise ValueError('All fifteen original source RX/day domains required')
    return data,tau[1],deepcopy(report)


def _calibrate(*args,initial_thresholds):
    from .star_oe_risk_budget import calibrate
    return calibrate(*args,initial_thresholds=initial_thresholds)


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('New immutable common-seeded source output required')
    started=time.perf_counter();marker=previous._parent_marker(config)
    model,reader,aux,identity=parent._backend(config['parent_eval_config'],device)
    if identity!=marker.get('model_identity'):raise ValueError('Actual r02 checkpoint identity differs from saved parent model')
    snapshots=(_snapshot_model(model),_snapshot_model(aux))
    def check():_check_model(model,snapshots[0]);_check_model(aux,snapshots[1])
    smoke=parent._smoke(reader,aux,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual frozen-checkpoint synthetic smoke must pass before source records')
    output.mkdir(parents=True);_write(output/'smoke.json',smoke)
    old_tau,common_report=previous._parent_calibration(config['parent_prediction_root'],marker);common=old_tau[2].copy()
    oe,zero,zero_report=_load_OE_predictions(config['OE_prediction_root'],marker,identity,common)
    known=_load_parent_predictions('known',config['parent_prediction_root'],old_tau)
    if set(known['ids'].tolist())&set(oe['ids'].tolist()):raise ValueError('Known V and OE physical roles must be disjoint')
    raw=known['aux_logits'].astype(np.float64).max(1)
    state,report=_calibrate(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw,known['rx_i'],known['scenario'],
        oe['ids'],oe['closed'],oe['score'],oe['domain'],initial_thresholds=common.copy())
    selected=np.asarray(state['conditional_thresholds'])
    if (selected.shape!=(6,) or selected.dtype!=np.float64 or not np.isfinite(selected).all() or np.any(selected<common)
            or not np.array_equal(selected,report.get('conditional_thresholds')) or set(report.get('groups',{}))!=set(GROUP_NAMES)):
        raise ValueError('Six finite thresholds monotone from common and all42 source risk groups required')
    check();thresholds=np.stack((common,zero,selected));_npz(output/'thresholds.npz',dict(conditional=thresholds))
    folder=output/'calibration';folder.mkdir()
    _write(folder/'complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],
        method_names=list(METHODS),conditional_thresholds=thresholds.tolist(),methods=dict(common_tail=common_report,zero_initialized_budget=zero_report,common_initialized_budget=report),
        group_count=42,source_V_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,novel_inputs_used=False,target_reads=0,source_only=True,
        known_only=False,source_V_neural_fit=False,neural_updates=0,monotone_thresholds_from_common=True))
    _npz(output/'OE_scores.npz',oe);known['scores']=raw[:,None]-thresholds[:,known['closed']].T;_npz(output/'known_predictions.npz',known)
    frozen=thresholds.copy();check();print(CALIBRATION_STATUS,config['arm'],flush=True)
    novel=_load_parent_predictions('novel',config['parent_prediction_root'],old_tau)
    if set(novel['ids'].tolist())&(set(known['ids'].tolist())|set(oe['ids'].tolist())):raise ValueError('Source novel/V/OE physical IDs must be disjoint')
    novel['scores']=novel['aux_logits'].astype(np.float64).max(1)[:,None]-thresholds[:,novel['closed']].T;check()
    if not np.array_equal(thresholds,frozen):raise ValueError('Thresholds changed after freeze')
    _npz(output/'novel_predictions.npz',novel)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],method_names=list(METHODS),
        primary_arm=PRIMARY_ARM,primary_method=PRIMARY_METHOD,classification_path='P1',score_threshold=0.,source_only=True,target_reads=0,
        parent_run_id=PARENT_RUN,parent_code_commit=PARENT_COMMIT,parent_model_identity=identity,parent_prediction_root=config['parent_prediction_root'],
        OE_parent_run_id=OE_PARENT_RUN,OE_parent_code_commit=OE_PARENT_COMMIT,OE_prediction_root=config['OE_prediction_root'],
        known_count=COUNTS['known'],novel_count=COUNTS['novel'],OE_count=COUNTS['OE'],all_thresholds_frozen_before_novel=True,
        model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True,neural_updates=0,G_forward_rows=0,F_forward_rows=2,
        real_P1_forward_rows=0,synthetic_P0_forward_rows=2,source_L_reads=0,source_U_reads=0,source_novel_truth_reads=0,
        new_OE_forward_rows=0,new_V_forward_rows=0,new_novel_forward_rows=0,new_original_F0_forward_rows=0,real_IQ_reads=0,
        monotone_thresholds_from_common=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,cache_identities=marker['cache_identities'],
        final_phase3_complete=False,A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(output/'complete.json',result);print(STATUS,config['arm'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_COMMON_SEEDED_RISK_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);run(_read(a.config),a.device);return 0


if __name__=='__main__':raise SystemExit(main())
