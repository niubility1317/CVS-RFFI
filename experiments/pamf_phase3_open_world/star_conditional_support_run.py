"""Source-only predicted-identity local support from frozen original-F0 features.

Real IQ is never loaded here. The prior source prediction contains the original
features/logits; only lawful source-L label metadata is added to its bank.
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

from . import star_local_support_run as base
from . import star_class_risk as risk

RUN_ID='20261030-phase3-star-conditional-support-s2026092701-r01'
SCHEMA='pamf_phase3_star_conditional_support_prediction_v1'
STATUS='STAR_CONDITIONAL_SUPPORT_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CONDITIONAL_SUPPORT_THRESHOLDS_FROZEN'
PARENT_RUN=base.RUN_ID
PARENT_COMMIT='84cdc012f7d9641b283c407f6048bfe4e6e21cf6'
PARENT_SCHEMA=base.SCHEMA
PARENT_STATUS=base.STATUS
METHODS=('pooled_relative','predicted_identity_relative')
PRIMARY_METHOD='predicted_identity_relative'
FIELDS={'run_id','code_commit','source_config','parent_prediction_root','cache_root','output_root','batch_size'}
SOURCE_FIELDS=base.SOURCE_FIELDS
COUNTS=base.COUNTS
RECEIVERS,SCENES=base.RECEIVERS,base.SCENES
KNOWN_FIELDS=base.KNOWN_FIELDS
NOVEL_FIELDS=base.NOVEL_FIELDS
BANK_FIELDS=base.BANK_FIELDS|{'L_y'}
_backend,_smoke,_snapshot,_check_models,_identity=base._backend,base._smoke,base._snapshot,base._check_models,base._identity
_same,_write,_npz,_read=base._same,base._write,base._npz,base._read


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or not isinstance(config.get('source_config'),dict) or set(config['source_config'])!=SOURCE_FIELDS
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('parent_prediction_root','cache_root','output_root'))):
        raise ValueError('Only seven source-only conditional-support configuration fields are allowed')
    source=config['source_config'];base.base.runner.validate_config('calibrate',dict(source,retention=.99))
    if (source['run_id']!=RUN_ID or source['row_id']!='star-conditional-support-s2026092701' or source['code_commit']!=config['code_commit']
            or source['output_root']!=config['output_root'] or source['batch_size']!=128 or source['model_seed']!=2026092701
            or source['registered_classes']!=base.base.native.CLASSES):
        raise ValueError('This independent run and original full-six source F0 are required')
    out=Path(config['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) for p in (config['parent_prediction_root'],config['cache_root'],source['source_output'],source['source_cache'])):
        raise ValueError('Output cannot overwrite parent predictions or original source/cache inputs')
    return config


def _load_npz(path,fields):
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=fields:raise ValueError('Exact public feature fields required; private or added fields forbidden')
        return {k:data[k].copy() for k in fields}


def _parent(config,identity):
    parent=_read(Path(config['parent_prediction_root'])/'complete.json')
    if (parent.get('schema')!=PARENT_SCHEMA or parent.get('status')!=PARENT_STATUS or parent.get('run_id')!=PARENT_RUN
            or parent.get('code_commit')!=PARENT_COMMIT or parent.get('model_identity')!=identity
            or parent.get('cache_root')!=config['cache_root'] or parent.get('classification_path')!='P1'
            or parent.get('teacher_run_id')!=base.TEACHER_RUN or parent.get('teacher_code_commit')!=base.TEACHER_COMMIT
            or parent.get('source_only') is not True or parent.get('default_head_replaced') is not False
            or any(parent.get(k)!=0 for k in ('target_reads','source_novel_truth_reads','neural_updates','G_updates','G_forward_rows','source_U_reads'))
            or any(parent.get(k) is not True for k in ('source_bank_frozen_before_V','source_bank_unchanged','model_parameters_buffers_unchanged','all_thresholds_frozen_before_novel'))
            or parent.get('fresh_closed_vs_cached_baseline_changed_count')!=0
            or any(parent.get(k)!=COUNTS[role] for k,role in (('known_count','V'),('novel_count','novel'),('bank_L_count','L'),('bank_OE_count','OE')))):
        raise ValueError('Complete matching original-F0 source-only local-support parent required')
    dura=parent.get('dura_identity',{})
    if dura.get('deployment_id')!='dura_v1' or dura.get('parameters')!=266951:
        raise ValueError('Matching frozen DURA v1 source feature ancestry required')
    handles=parent.get('cache_identities',{})
    if (set(handles)!=set(COUNTS) or any(handles[r].get('role')!=r or handles[r].get('count')!=n
            or handles[r].get('parent_F0_identity')!=identity or handles[r].get('dura_identity')!=dura for r,n in COUNTS.items())):
        raise ValueError('Matching four validated source role handles required')
    return parent


def _bank(config,parent):
    bank=_load_npz(Path(config['parent_prediction_root'])/'source_bank.npz',base.BANK_FIELDS)
    for role in ('L','OE'):
        n=COUNTS[role];ids=bank[role+'_ids'];features=bank[role+'_unit_features'];rx=bank[role+'_rx_i']
        if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n
                or features.shape!=(n,160) or features.dtype!=np.float32 or not np.isfinite(features).all()
                or not np.allclose(np.linalg.norm(features.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
                or rx.shape!=(n,) or rx.dtype!=np.int64 or set(rx.tolist())!=set(RECEIVERS)):
            raise ValueError('Exact parent bank IDs, original-F0 unit features and source RX required')
    if set(bank['L_ids'].tolist())&set(bank['OE_ids'].tolist()):raise ValueError('Parent L and OE IDs must be disjoint')
    cache=Path(config['cache_root'])/'L';manifest=_read(cache/'manifest.json')
    if manifest!=parent['cache_identities']['L']:raise ValueError('Source-L label metadata must use the same validated role')
    ids=np.load(cache/'ids.npy',allow_pickle=False)
    metadata=_load_npz(cache/'metadata.npz',{'y','rx_i','day_i','domain'});y=metadata['y']
    if (not np.array_equal(ids,bank['L_ids']) or y.shape!=(COUNTS['L'],) or y.dtype!=np.int64
            or not np.isin(y,range(6)).all() or any(np.count_nonzero(y==c)<8 for c in range(6))
            or not np.array_equal(metadata['rx_i'],bank['L_rx_i'])):
        raise ValueError('Legal six-class source-L labels and RX must align to the parent bank ID order')
    bank['L_y']=y
    return bank


def _control(config):
    root=Path(config['parent_prediction_root']);cal=_read(root/'calibration/complete.json')
    tau=_load_npz(root/'thresholds.npz',{'conditional'})['conditional']
    if (cal.get('schema')!=PARENT_SCHEMA or cal.get('status')!=base.CALIBRATION_STATUS or cal.get('run_id')!=PARENT_RUN
            or cal.get('code_commit')!=PARENT_COMMIT or cal.get('method_names')!=list(base.METHODS) or cal.get('group_count')!=42
            or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or not np.array_equal(np.asarray(cal.get('conditional_thresholds')),tau)):
        raise ValueError('All parent thresholds must be frozen with the original calibration identity')
    control=cal.get('methods',{}).get('pooled_relative',{})
    if not isinstance(control,dict) or not np.array_equal(np.asarray(control.get('conditional_thresholds')),tau[1]):
        raise ValueError('Exact frozen parent pooled-relative control report and six thresholds required')
    return tau[1].copy(),deepcopy(control)


def _public(config,role,seen):
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS
    data=_load_npz(Path(config['parent_prediction_root'])/(role+'_predictions.npz'),fields)
    n=COUNTS['V' if role=='known' else 'novel'];ids=data['ids'];unit=data['unit_features'];logits=data['default_logits'];closed=data['closed']
    if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or seen&set(ids.tolist())
            or unit.shape!=(n,160) or unit.dtype!=np.float32 or not np.isfinite(unit).all()
            or not np.allclose(np.linalg.norm(unit.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
            or logits.shape!=(n,6) or logits.dtype!=np.float32 or not np.isfinite(logits).all()
            or closed.shape!=(n,) or closed.dtype!=np.int64 or not np.array_equal(closed,logits.argmax(1))
            or any(data[k].shape!=(n,) or data[k].dtype!=np.int64 for k in ('rx_i','day_i'))
            or not np.isin(data['rx_i'],RECEIVERS).all() or not np.isin(data['day_i'],(1,2,3)).all()
            or any(data[k].shape!=(n,3) or data[k].dtype!=np.float64 or not np.isfinite(data[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Original parent features/default-head identity and source public metadata required')
    if role=='known' and (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or not np.isin(data['y'],range(6)).all()
            or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
            or data['baseline_closed'].shape!=(n,) or data['baseline_closed'].dtype!=np.int64
            or not np.array_equal(data['baseline_closed'],closed)):
        raise ValueError('Unchanged parent original-F0 P1 baseline and known-only calibration labels required')
    return data


def _fit(L_features,L_y,OE_features):
    from .star_conditional_support_core import fit
    return fit(L_features,L_y,OE_features)


def _score(features,closed,state):
    from .star_conditional_support_core import score
    return score(features,closed,state)


def _raw(data,state,control_tau):
    started=time.perf_counter();raw=_score(data['unit_features'],data['closed'],state);seconds=time.perf_counter()-started
    if (raw.shape!=(len(data['ids']),2) or raw.dtype!=np.float64 or not np.isfinite(raw).all()
            or not np.allclose(raw[:,0],data['raw_scores'][:,1],rtol=0,atol=1e-12)):
        raise ValueError('Two finite support scores and exact matched parent pooled-relative raw control required')
    margin=raw[:,0]-control_tau[data['closed']]
    if (not np.allclose(margin,data['scores'][:,1],rtol=0,atol=1e-12)
            or not np.array_equal(margin>=0,data['scores'][:,1]>=0)):
        raise ValueError('Parent pooled-relative thresholds, margins and acceptance must reproduce without refit')
    raw[:,0]=data['raw_scores'][:,1]
    return raw,seconds


def _export(data,raw,tau):
    result={k:v for k,v in data.items() if k not in ('raw_scores','scores')}
    margins=raw-tau[:,data['closed']].T;margins[:,0]=data['scores'][:,1]
    result.update(raw_scores=raw,scores=margins)
    return result


def run(config,device='cuda:0'):
    validate_config(config);out=Path(config['output_root'])
    if out.exists():raise FileExistsError('A new immutable source conditional-support output root is required')
    started=time.perf_counter();backend,_=_backend(config['source_config'],device);identity=backend['reader_identity'];_identity(identity,config['source_config'])
    snapshots=_snapshot(backend);smoke=_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual original-F0 checkpoint smoke must pass before parent feature consumption')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke)
    parent=_parent(config,identity);bank=_bank(config,parent);control_tau,control_report=_control(config)
    state=_fit(bank['L_unit_features'],bank['L_y'],bank['OE_unit_features']);frozen=deepcopy(state)
    def check():
        _check_models(snapshots)
        if not _same(state,frozen):raise ValueError('Frozen conditional support bank changed during calibration/prediction')
    check();_npz(out/'source_bank.npz',bank);seen=set(bank['L_ids'].tolist())|set(bank['OE_ids'].tolist())
    known=_public(config,'known',seen);seen.update(known['ids'].tolist());raw,cpu_known=_raw(known,state,control_tau);check()
    values,report=risk.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw[:,1],known['rx_i'],known['scenario'])
    tau=np.stack((control_tau,values['conditional_thresholds'])).astype(np.float64)
    if tau.shape!=(2,6) or not np.isfinite(tau).all():raise ValueError('Twelve finite frozen source-known thresholds required')
    methods=dict(pooled_relative=control_report,predicted_identity_relative=report['methods']['predicted_class'])
    check();_npz(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
        method_names=list(METHODS),conditional_thresholds=tau.tolist(),methods=methods,group_count=42,source_only=True,target_reads=0,
        primary_threshold_fit_known_only=True,control_thresholds_reused=True,source_L_TX_labels_used=True,source_OE_TX_or_K_used=False))
    _npz(out/'known_predictions.npz',_export(known,raw,tau));frozen_tau=tau.copy();check();print(CALIBRATION_STATUS,flush=True)
    novel=_public(config,'novel',seen);raw,cpu_novel=_raw(novel,state,control_tau);check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Thresholds changed after novel access')
    _npz(out/'novel_predictions.npz',_export(novel,raw,tau))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],model_identity=identity,dura_identity=parent['dura_identity'],
        parent_run_id=PARENT_RUN,parent_code_commit=PARENT_COMMIT,parent_prediction_root=config['parent_prediction_root'],cache_root=config['cache_root'],
        teacher_run_id=base.TEACHER_RUN,teacher_code_commit=base.TEACHER_COMMIT,teacher_root=parent['teacher_root'],cache_identities=parent['cache_identities'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,classification_path='P1',score_threshold=0.,parameters=232993,default_head_replaced=False,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],bank_L_count=COUNTS['L'],bank_OE_count=COUNTS['OE'],source_only=True,target_reads=0,
        source_novel_truth_reads=0,source_U_reads=0,F_forward_rows=2,synthetic_P0_forward_rows=2,G_forward_rows=0,new_V_forward_rows=0,new_novel_forward_rows=0,
        source_L_forward_rows=0,source_OE_forward_rows=0,neural_updates=0,G_updates=0,real_IQ_values_read=0,source_L_metadata_labels_read=COUNTS['L'],
        parent_public_features_reused=True,control_scores_thresholds_acceptance_reproduced=True,source_bank_frozen_before_V=True,source_bank_unchanged=True,
        model_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True,source_L_TX_labels_used=True,source_OE_TX_or_K_used=False,
        cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],fresh_closed_vs_cached_baseline_changed_count=0,
        fresh_closed_correct_count=int(np.count_nonzero(known['closed']==known['y'])),cached_baseline_correct_count=int(np.count_nonzero(known['baseline_closed']==known['y'])),
        source_bank_is_deployment_state=False,source_free_export_required=True,eligible_primary=False,target_eligible=False,
        final_phase3_complete=False,A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_CONDITIONAL_SUPPORT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);run(_read(a.config),a.device);return 0


if __name__=='__main__':raise SystemExit(main())
