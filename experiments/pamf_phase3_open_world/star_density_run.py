"""Source-only aggregate density evidence; original F0 identity stays frozen.

L/OE features and V/novel original-head predictions reuse completed G10 P1
artifacts. Only the two synthetic no-query smoke rows receive new F forwards.
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
from . import star_protected_open_cache as cache
from . import star_local_support_run as reference

RUN_ID='20261030-phase3-star-density-s2026092701-r01'
SCHEMA='pamf_phase3_star_density_prediction_v1'
STATUS='STAR_DENSITY_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_DENSITY_THRESHOLDS_FROZEN'
METHODS=('known_only','density_ratio')
PRIMARY_METHOD='density_ratio'
FIELDS={'run_id','code_commit','source_config','feature_root','reference_prediction_root','output_root','batch_size'}
SOURCE_FIELDS=base.runner.COMMON|{'source_cache'}
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
RECEIVERS,SCENES=reference.RECEIVERS,reference.SCENES
REFERENCE_RUN=reference.RUN_ID
REFERENCE_CODE='84cdc012f7d9641b283c407f6048bfe4e6e21cf6'
REFERENCE_SCHEMA,REFERENCE_STATUS=reference.SCHEMA,reference.STATUS
KNOWN_FIELDS=reference.KNOWN_FIELDS.copy()
NOVEL_FIELDS=reference.NOVEL_FIELDS.copy()
REFERENCE_KNOWN_FIELDS=KNOWN_FIELDS-{'scores','raw_scores'}
REFERENCE_NOVEL_FIELDS=NOVEL_FIELDS-{'scores','raw_scores'}
_backend,_smoke,_snapshot,_check_models=base._backend,base._smoke,base._snapshot,base._check_models
_write,_npz,_read,_same=base._write,base._npz,base.runner._read,base.previous._same
load_features=cache.load_features


def _core():
    from . import star_density_core
    return star_density_core


def _fit_known(*args):return _core().fit_known(*args)
def _fit_background(*args):return _core().fit_background(*args)
def _density_score(state,x,method):return _core().score(state,x,method)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or not isinstance(c.get('source_config'),dict) or set(c['source_config'])!=SOURCE_FIELDS
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('feature_root','reference_prediction_root','output_root'))):
        raise ValueError('Exact seven-field source density configuration required; no truth or target inputs')
    s=c['source_config'];base.runner.validate_config('calibrate',dict(s,retention=.99))
    if (s['run_id']!=RUN_ID or s['row_id']!='star-density-s2026092701' or s['code_commit']!=c['code_commit']
            or s['output_root']!=c['output_root'] or s['batch_size']!=128 or s['model_seed']!=2026092701
            or s['registered_classes']!=base.native.CLASSES):
        raise ValueError('This independent source density run and original full-six F0 are required')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out)
            for p in (c['feature_root'],c['reference_prediction_root'],s['source_output'],s['source_cache'])):
        raise ValueError('Output must be separate from all immutable source inputs')
    return c


def _feature(root,role,identity):
    a=load_features(root,role);m=a['manifest'];n=COUNTS[role];meta=a['metadata']
    if (m.get('parent_F0_identity')!=identity or set(meta)!=({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
            or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n
            or any(v.shape!=(n,) or v.dtype!=np.int64 for v in meta.values())
            or not np.isin(meta['rx_i'],RECEIVERS).all() or not np.isin(meta['day_i'],(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(RECEIVERS,meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Only original-F0 source L labels and public L/OE RX/day metadata are allowed')
    if role=='L' and (not np.isin(meta['y'],range(6)).all() or set(meta['y'].tolist())!=set(range(6))):
        raise ValueError('The original six source known classes are required')
    _core().unit(a['raw_original_fused160'])
    return a


def _reference_marker(root,identity,feature_manifest):
    m=_read(Path(root)/'complete.json')
    required=dict(schema=REFERENCE_SCHEMA,status=REFERENCE_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,
        classification_path='P1',source_only=True,target_reads=0,source_novel_truth_reads=0,source_U_reads=0,
        neural_updates=0,default_head_replaced=False,model_parameters_buffers_unchanged=True,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],model_identity=identity,dura_identity=feature_manifest['dura_identity'])
    if any(m.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete fixed local-support original-F0 P1 predictions with matching ancestry required')
    handles=m.get('cache_identities',{})
    for role in COUNTS:
        h=handles.get(role,{})
        if (h.get('role')!=role or h.get('count')!=COUNTS[role] or h.get('parent_F0_identity')!=identity
                or h.get('dura_identity')!=m['dura_identity'] or h.get('source_only') is not True
                or h.get('restored_input_path')!='P1'
                or (role in ('L','OE') and h!=feature_manifest['source_manifests'][role])):
            raise ValueError('Feature fit roles and reference predictions must share original F0/DURA source handles')
    return m


def _reference_data(root,role):
    """Read only reusable F0 evidence. Old scores and thresholds are never read."""
    fields=REFERENCE_KNOWN_FIELDS if role=='known' else REFERENCE_NOVEL_FIELDS
    expected=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=expected:raise ValueError('Only the existing exact public local-support prediction layout is allowed')
        a={k:z[k].copy() for k in fields}
    return validate_rows(a,role,False)


def validate_rows(a,role,with_scores=True):
    n=COUNTS['V' if role=='known' else 'novel']
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS
    if not with_scores:fields=fields-{'raw_scores','scores'}
    if (set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n
            or np.any(np.char.strip(a['ids'])=='') or a['unit_features'].shape!=(n,160) or a['unit_features'].dtype!=np.float32
            or not np.isfinite(a['unit_features']).all()
            or not np.allclose(np.linalg.norm(a['unit_features'].astype(np.float64),axis=1),1.,rtol=0.,atol=2e-6)
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],roster).all() for k,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))))):
        raise ValueError('Aligned opaque IDs, unit F0 features and unchanged default-head identity required; no replacement forward')
    if role=='known' and (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all()
            or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],SCENES).all()
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):
        raise ValueError('Visible known-V labels and the unchanged original F0 P1 baseline required')
    if with_scores and any(a[k].shape!=(n,2) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('scores','raw_scores')):
        raise ValueError('Two finite FP64 policy scores and conditional margins required')
    return a


def _score_rows(x,states):
    started=time.perf_counter()
    scores=np.column_stack([_density_score(states[name],x,name) for name in METHODS])
    if scores.shape!=(len(x),2) or scores.dtype!=np.float64 or not np.isfinite(scores).all():
        raise ValueError('Finite known-density and density-ratio outputs required')
    return scores,time.perf_counter()-started


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable density output root required')
    started=time.perf_counter();backend,_=_backend(c['source_config'],device)
    identity=backend['reader_identity'];reference._identity(identity,c['source_config'])
    before=_snapshot(backend);smoke=_smoke(backend,device);_check_models(before)
    if smoke.get('status')!='PASS':raise ValueError('Actual original-F0 no-query checkpoint smoke must pass before source reads')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_write(out/'resolved_config.json',c)
    L=_feature(c['feature_root'],'L',identity);fm=L['manifest']
    states={};diagnostics={};fit_started=time.perf_counter()
    states['known_only'],diagnostics['known_only']=_fit_known(L['raw_original_fused160'],L['metadata']['y'],L['metadata']['rx_i'],L['metadata']['day_i'])
    fit_seconds=time.perf_counter()-fit_started
    _core().validate_state(states['known_only'],'known_only');known_before=deepcopy(states['known_only'])
    # The no-OE control is already fitted and frozen before any OE values open.
    OE=_feature(c['feature_root'],'OE',identity)
    if OE['manifest']!=fm or set(L['ids'].tolist())&set(OE['ids'].tolist()):raise ValueError('Matching, physically disjoint L/OE feature roles required')
    fit_started=time.perf_counter()
    states['density_ratio'],diagnostics['density_ratio']=_fit_background(states['known_only'],OE['raw_original_fused160'],OE['metadata']['rx_i'],OE['metadata']['day_i'])
    fit_seconds+=time.perf_counter()-fit_started
    if not _same(states['known_only'],known_before):raise ValueError('Background fit changed the no-OE control')
    _core().validate_state(states['density_ratio'],'density_ratio')
    seen=set(L['ids'].tolist())|set(OE['ids'].tolist());del L,OE
    frozen=deepcopy(states)
    def check():
        _check_models(before)
        if not _same(states,frozen):raise ValueError('Source-fitted density aggregate state changed during V/novel evaluation')
    models=out/'models';models.mkdir()
    for method in METHODS:_npz(models/(method+'.npz'),states[method])
    _write(models/'complete.json',dict(status='STAR_DENSITY_MODELS_FROZEN',run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),diagnostics=diagnostics,fit_roles=dict(known_only=['L'],density_ratio=['L','OE']),
        source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_fit=False,novel_fit=False))
    ref=_reference_marker(c['reference_prediction_root'],identity,fm)
    known=_reference_data(c['reference_prediction_root'],'known')
    if seen&set(known['ids'].tolist()):raise ValueError('Known calibration cannot overlap source fit records')
    seen.update(known['ids'].tolist())
    raw,cpu_known=_score_rows(known['unit_features'],states);check()
    thresholds=[];globals_=[];methods={};controls={}
    for j,name in enumerate(METHODS):
        state,report=risk.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw[:,j],known['rx_i'],known['scenario'])
        thresholds.append(state['conditional_thresholds']);globals_.append(state['global_threshold'])
        methods[name]=report['methods']['predicted_class'];controls[name]=report['methods']['global_control']
    tau=np.asarray(thresholds,np.float64);global_tau=np.asarray(globals_,np.float64);check()
    if tau.shape!=(2,6) or not np.isfinite(tau).all():raise ValueError('Both policies require six finite predicted-class thresholds')
    _npz(out/'thresholds.npz',dict(conditional=tau,**{'global':global_tau}));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,known_only=True,
        unknown_inputs_used=False,source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,
        conditional_thresholds=tau.tolist(),global_thresholds=global_tau.tolist(),methods=methods,global_controls=controls))
    known.update(raw_scores=raw,scores=raw-tau[:,known['closed']].T);_npz(out/'known_predictions.npz',known)
    frozen_tau=tau.copy();check();print(CALIBRATION_STATUS,flush=True)
    # First novel-feature access follows BOTH frozen models, thresholds and V predictions.
    novel=_reference_data(c['reference_prediction_root'],'novel')
    if seen&set(novel['ids'].tolist()):raise ValueError('Novel source records cannot overlap fitting or calibration')
    raw,cpu_novel=_score_rows(novel['unit_features'],states);check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Known-only thresholds changed after novel access')
    novel.update(raw_scores=raw,scores=raw-tau[:,novel['closed']].T);_npz(out/'novel_predictions.npz',novel)
    state_bytes={name:sum(v.nbytes for v in state.values()) for name,state in states.items()}
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='P1',score_threshold=0.,model_identity=identity,dura_identity=fm['dura_identity'],source_feature_manifest=fm,
        reference_run_id=REFERENCE_RUN,reference_code_commit=REFERENCE_CODE,reference_prediction_root=c['reference_prediction_root'],
        reference_prediction_identity=ref,known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,source_novel_truth_reads=0,target_reads=0,original_U_s_reads=0,real_IQ_reads=0,
        real_F_forward_rows=0,synthetic_F_forward_rows=2,G_forward_rows=0,G_checkpoint_loaded=False,
        neural_updates=0,native_updates=0,identity_head_updates=0,G_updates=0,default_head_replaced=False,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        source_V_model_fit=False,source_V_threshold_fit=True,source_novel_fit_reads=0,OE_TX_or_K_used=False,
        known_only_uses_OE=False,density_ratio_uses_OE_membership=True,old_scores_or_thresholds_read=False,
        source_aggregate_bytes_by_method=state_bytes,threshold_bytes=int(tau.nbytes),cpu_fit_seconds=fit_seconds,
        cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],cpu_readout_policy_rows=2*(COUNTS['V']+COUNTS['novel']),
        final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def main(argv=None):
    print(json.dumps(dict(event='STAR_DENSITY_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);run(_read(a.config),a.device)


if __name__=='__main__':main()
