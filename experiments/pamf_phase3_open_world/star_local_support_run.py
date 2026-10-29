"""Frozen original-F0 local support from original source teacher features.

Only cached P1 V/novel receive new real neural forwards. L/OE contribute
original-F0 teacher features and source RX metadata, never IQ values here.
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

from . import star_consensus_run as base
from . import star_class_risk as risk

RUN_ID='20261030-phase3-star-local-support-s2026092701-r01'
SCHEMA='pamf_phase3_star_local_support_prediction_v1'
STATUS='STAR_LOCAL_SUPPORT_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_LOCAL_SUPPORT_THRESHOLDS_FROZEN'
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
TEACHER_COMMIT='5aa041122396d48c49b8cb3d5912649bdc3217e1'
TEACHER_SCHEMA='pamf_phase3_star_joint_ova_teacher_v1'
TEACHER_STATUS='STAR_JOINT_OVA_TEACHER_FROZEN'
METHODS=('pooled_known','pooled_relative','rx_consensus_relative')
PRIMARY_METHOD='rx_consensus_relative'
FIELDS={'run_id','code_commit','source_config','cache_root','teacher_root','output_root','batch_size'}
SOURCE_FIELDS=base.runner.COMMON|{'source_cache'}
COUNTS=base.COUNTS
RECEIVERS,SCENES=base.previous.RECEIVERS,base.previous.SCENES
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','unit_features','default_logits','closed','baseline_closed','raw_scores','scores'}
NOVEL_FIELDS=KNOWN_FIELDS-{'y','scenario','baseline_closed'}
BANK_FIELDS={'L_ids','L_unit_features','L_rx_i','OE_ids','OE_unit_features','OE_rx_i'}
_backend,_smoke,_snapshot,_check_models,_encode=base._backend,base._smoke,base._snapshot,base._check_models,base._encode
_load_role,close_role=base._role,base.close_role
_write,_npz,_read=base._write,base._npz,base.runner._read
_same=base.previous._same


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or not isinstance(config.get('source_config'),dict) or set(config['source_config'])!=SOURCE_FIELDS
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('cache_root','teacher_root','output_root'))):
        raise ValueError('Only the seven source-only local-support configuration fields are allowed')
    source=config['source_config'];base.runner.validate_config('calibrate',dict(source,retention=.99))
    if (source['run_id']!=RUN_ID or source['row_id']!='star-local-support-s2026092701' or source['code_commit']!=config['code_commit']
            or source['output_root']!=config['output_root'] or source['batch_size']!=128 or source['model_seed']!=2026092701
            or source['registered_classes']!=base.native.CLASSES):
        raise ValueError('This independent run and the original full-six source F0 are required')
    out=Path(config['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) for p in (config['cache_root'],config['teacher_root'],source['source_output'],source['source_cache'])):
        raise ValueError('Output cannot overwrite original source, teacher or cache inputs')
    return config


def _identity(identity,source):
    required=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
    if (any(identity.get(k)!=v for k,v in required.items())
            or any(identity.get(k)!=source[k] for k in ('source_output','source_contract','base_release'))):
        raise ValueError('Only the original F0 E50 identity may generate local-support evidence')


def _unit(value):
    x=np.asarray(value)
    if x.ndim!=2 or x.shape[1]!=160 or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError('Finite original-F0 FP32 fused160 features required')
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(norm<1e-12):raise ValueError('Nonzero original-F0 features required')
    return (x/norm).astype(np.float32)


def _teacher_bank(config,identity):
    root=Path(config['teacher_root']);teacher=_read(root/'complete.json')
    if (teacher.get('schema')!=TEACHER_SCHEMA or teacher.get('status')!=TEACHER_STATUS or teacher.get('run_id')!=TEACHER_RUN
            or teacher.get('code_commit')!=TEACHER_COMMIT or teacher.get('parent_F0_identity')!=identity
            or teacher.get('model_parameters')!=232993 or teacher.get('input_path')!='cached_P1_only' or teacher.get('feature')!='unit_fused160'
            or teacher.get('source_only') is not True or any(teacher.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads',
                'original_U_s_reads','OE_truth_reads','encoder_updates','identity_head_updates','G_updates','G_forward_rows'))):
        raise ValueError('Matching complete r02 ORIGINAL-F0 teacher required; no candidate features or target ancestry')
    bank={};handles={};seen=set();dura=teacher.get('dura_identity')
    for role in ('L','OE'):
        data=_load_role(Path(config['cache_root'])/role,role,identity,dura)
        try:
            n=COUNTS[role];manifest=data['manifest'];handle=teacher.get('cache_handles',{}).get(role,{})
            if (manifest.get('role')!=role or manifest.get('count')!=n or manifest.get('parent_F0_identity')!=identity
                    or manifest.get('dura_identity')!=dura or teacher.get('source_'+role+'_count')!=n
                    or handle.get('role')!=role or handle.get('count')!=n or handle.get('parent_F0_identity')!=identity
                    or handle.get('dura_identity')!=dura):
                raise ValueError('Teacher and public source role/model/DURA handles differ')
            with np.load(root/(role+'.npz'),allow_pickle=False) as stored:
                if set(stored.files)!={'ids','unit_features'}:raise ValueError('Original teacher IDs and unit features only')
                ids=stored['ids'].copy();features=stored['unit_features'].copy()
            if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or not np.array_equal(ids,data['ids'])
                    or features.shape!=(n,160) or features.dtype!=np.float32 or not np.isfinite(features).all()
                    or not np.allclose(np.linalg.norm(features.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):
                raise ValueError('Exact source role teacher ID order and finite FP32 unit features required')
            rx=np.array(data['metadata']['rx_i'],copy=True)
            if rx.shape!=(n,) or rx.dtype!=np.int64 or set(rx.tolist())!=set(RECEIVERS):raise ValueError('All five original source RX groups required')
            if seen&set(ids.tolist()):raise ValueError('L and OE source physical IDs must be disjoint')
            seen.update(ids.tolist());bank.update({role+'_ids':ids,role+'_unit_features':features,role+'_rx_i':rx});handles[role]=manifest
        finally:close_role(data)
    return bank,handles,dura


def _fit(L_features,L_rx,OE_features,OE_rx):
    from .star_local_support_core import fit
    return fit(L_features,L_rx,OE_features,OE_rx)


def _score(features,state):
    from .star_local_support_core import score
    return score(features,state)


def _predict(reader,data,state,device,batch_size,role):
    units=[];logits=[];scores=[];values=data['restored'];cpu_seconds=0.
    for start in range(0,len(data['ids']),batch_size):
        branches,z=_encode(reader,values[start:start+batch_size],device);u=_unit(branches['fused160'])
        if z.shape!=(len(u),6) or z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('Finite original default-six logits required')
        score_started=time.perf_counter();raw=_score(u,state);cpu_seconds+=time.perf_counter()-score_started
        if raw.shape!=(len(u),3) or raw.dtype!=np.float64 or not np.isfinite(raw).all():raise ValueError('Three finite FP64 local-support scores required')
        units.append(u);logits.append(z);scores.append(raw)
        print('STAR_LOCAL_SUPPORT_PROGRESS',role,start+len(u),len(data['ids']),flush=True)
    return np.concatenate(units),np.concatenate(logits),np.concatenate(scores),cpu_seconds


def run(config,device='cuda:0'):
    validate_config(config);out=Path(config['output_root'])
    if out.exists():raise FileExistsError('A new immutable source local-support output root is required')
    started=time.perf_counter();backend,_=_backend(config['source_config'],device);identity=backend['reader_identity'];_identity(identity,config['source_config'])
    snapshots=_snapshot(backend);smoke=_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual original-F0 checkpoint smoke must pass before source inputs')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke)
    bank,handles,dura=_teacher_bank(config,identity)
    state=_fit(bank['L_unit_features'],bank['L_rx_i'],bank['OE_unit_features'],bank['OE_rx_i']);frozen=deepcopy(state)
    def check():
        _check_models(snapshots)
        if not _same(state,frozen):raise ValueError('Frozen source bank changed during V/novel prediction')
    check();_npz(out/'source_bank.npz',bank);seen=set(bank['L_ids'].tolist())|set(bank['OE_ids'].tolist());V=novel=None
    try:
        V=_load_role(Path(config['cache_root'])/'V','V',identity,dura);handles['V']=V['manifest']
        if V['manifest'].get('parent_F0_identity')!=identity:raise ValueError('Known V original-F0 identity differs from teacher/model')
        if seen&set(V['ids'].tolist()):raise ValueError('Known V cannot overlap the source bank')
        seen.update(V['ids'].tolist());u,z,raw,cpu_V=_predict(backend['reader'],V,state,device,config['batch_size'],'V');check()
        closed=z.argmax(1).astype(np.int64);baseline=np.array(V['baseline_closed'][:,1],copy=True);meta=V['metadata'];thresholds=[];methods={}
        changed=int(np.count_nonzero(closed!=baseline));fresh_correct=int(np.count_nonzero(closed==meta['y']));cached_correct=int(np.count_nonzero(baseline==meta['y']))
        for i,name in enumerate(METHODS):
            values,report=risk.calibrate_known(V['ids'],meta['y'],closed,baseline,raw[:,i],meta['rx_i'],meta['scenario'])
            thresholds.append(values['conditional_thresholds']);methods[name]=report['methods']['predicted_class']
        thresholds=np.asarray(thresholds,np.float64);check()
        if thresholds.shape!=(3,6) or not np.isfinite(thresholds).all():raise ValueError('Eighteen finite source-known thresholds required')
        _npz(out/'thresholds.npz',dict(conditional=thresholds));(out/'calibration').mkdir()
        _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
            method_names=list(METHODS),conditional_thresholds=thresholds.tolist(),methods=methods,group_count=42,known_only=True,unknown_inputs_used=False,
            source_only=True,target_reads=0,source_bank_uses_OE_membership=True,source_bank_uses_TX_labels=False,source_V_neural_fit=False))
        _npz(out/'known_predictions.npz',dict(ids=V['ids'],y=meta['y'],rx_i=meta['rx_i'],day_i=meta['day_i'],scenario=meta['scenario'],
            unit_features=u,default_logits=z,closed=closed,baseline_closed=baseline,raw_scores=raw,scores=raw-thresholds[:,closed].T))
        close_role(V);V=None;frozen_thresholds=thresholds.copy();check();print(CALIBRATION_STATUS,flush=True)
        novel=_load_role(Path(config['cache_root'])/'novel','novel',identity,dura);handles['novel']=novel['manifest']
        if novel['manifest'].get('parent_F0_identity')!=identity:raise ValueError('Novel original-F0 identity differs from teacher/model')
        if seen&set(novel['ids'].tolist()):raise ValueError('Source novel cannot overlap source bank or known V')
        u,z,raw,cpu_novel=_predict(backend['reader'],novel,state,device,config['batch_size'],'novel');closed=z.argmax(1).astype(np.int64);check()
        if not np.array_equal(thresholds,frozen_thresholds):raise ValueError('Thresholds changed after novel access')
        _npz(out/'novel_predictions.npz',dict(ids=novel['ids'],rx_i=novel['metadata']['rx_i'],day_i=novel['metadata']['day_i'],
            unit_features=u,default_logits=z,closed=closed,raw_scores=raw,scores=raw-thresholds[:,closed].T))
        close_role(novel);novel=None
    finally:
        for data in (V,novel):
            if data is not None:close_role(data)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],model_identity=identity,dura_identity=dura,
        teacher_run_id=TEACHER_RUN,teacher_code_commit=TEACHER_COMMIT,teacher_root=config['teacher_root'],cache_root=config['cache_root'],cache_identities=handles,
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,classification_path='P1',score_threshold=0.,parameters=232993,default_head_replaced=False,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],bank_L_count=COUNTS['L'],bank_OE_count=COUNTS['OE'],source_only=True,target_reads=0,
        source_novel_truth_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,F_forward_rows=COUNTS['V']+COUNTS['novel']+2,
        new_V_forward_rows=COUNTS['V'],new_novel_forward_rows=COUNTS['novel'],synthetic_P0_forward_rows=2,source_L_forward_rows=0,source_OE_forward_rows=0,
        cpu_readout_seconds=cpu_V+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],
        source_U_reads=0,received_P0_values_read=0,source_bank_frozen_before_V=True,source_bank_unchanged=True,model_parameters_buffers_unchanged=True,
        all_thresholds_frozen_before_novel=True,source_bank_uses_OE_membership=True,source_bank_uses_TX_labels=False,
        fresh_closed_vs_cached_baseline_changed_count=changed,fresh_closed_correct_count=fresh_correct,cached_baseline_correct_count=cached_correct,
        final_phase3_complete=False,A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_LOCAL_SUPPORT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);run(_read(a.config),a.device);return 0


if __name__=='__main__':raise SystemExit(main())
