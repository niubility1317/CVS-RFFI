"""Received-only open representation; immutable P1 default-head identity."""
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
import torch
from . import star_signal_open_eval as previous
from . import star_macro12_run as macro
from . import star_macro12_risk as risk
from . import star_representation_cache as IQ

RUN_ID='20261031-phase3-star-raw-open-s2026092701-r01'
SCHEMA='pamf_phase3_star_raw_open_prediction_v1'
STATUS='STAR_RAW_OPEN_PREDICTIONS_COMPLETE'
MODELS_STATUS='STAR_RAW_OPEN_MODELS_FROZEN'
DENSITY_STATUS=MODELS_STATUS
CALIBRATION_STATUS='STAR_RAW_OPEN_THRESHOLDS_FROZEN'
ARMS=('E_RAW',)
PRIMARY_ARM=ARMS[0]
METHODS=('P1_known12','P1_ratio12','P0_known12','P0_ratio12')
PRIMARY_METHOD=METHODS[2]
PRIMARY_INDEX=2
REFERENCE_RUN=macro.RUN_ID
REFERENCE_CODE='26fbecff1df7e5476bcc5f459d8367c6fa49fd36'
SIGNAL_REFERENCE_RUN=previous.RUN_ID
SIGNAL_REFERENCE_CODE='dc01a402582f93e452dbaa3ef59a9fbb44151818'
FIELDS={'run_id','code_commit','arm','train_config','reference_prediction_root','signal_reference_root','output_root','batch_size'}
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
KNOWN_FIELDS=previous.KNOWN_FIELDS|{'raw_signal_features'}
NOVEL_FIELDS=previous.NOVEL_FIELDS|{'raw_signal_features'}
_read,_write,_npz,_same,_core=previous._read,previous._write,previous._npz,previous._same,previous._core
_snapshot,_unchanged,_project,_smoke=previous._snapshot,previous._unchanged,previous._project,previous._smoke


def _train():
    from . import star_raw_open_train
    return star_raw_open_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm')!=PRIMARY_ARM
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or not isinstance(c.get('train_config'),dict)
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('reference_prediction_root','signal_reference_root','output_root'))):
        raise ValueError('Exact eight-field source P0 evaluation config required; no truth/target')
    t=_train().validate_config(c['train_config'])
    if any(t[k]!=c[k] for k in ('run_id','code_commit','arm')) or t['control_root']!=c['reference_prediction_root']:raise ValueError('Actual P0 training row and fixed macro12 control differ')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in (t['IQ_root'],t['output_root'],c['reference_prediction_root'],c['signal_reference_root'])):raise ValueError('Immutable output must be separate from every original input')
    return c


def _identity(i,c):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,open_signal_updates=2220,native_updates=0,original_parent_updates=11100,signal_feature_cumulative_updates=13320,
        signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=567714,
        identity_head_replaced=False,input_path='cached_P0_IQ_only',teacher_input_path='P0')
    if any(i.get(k)!=v for k,v in expected.items()) or Path(i.get('checkpoint','')).resolve()!=(Path(c['train_config']['output_root'])/'final.pt').resolve():raise ValueError('Actual original-F0-initialized frozen P0 signal checkpoint required')
    p=i.get('parent_F0_identity',{});d=i.get('dura_identity',{})
    if any(p.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items()) or any(d.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()):raise ValueError('Original F0 and fixed observation/DURA ancestry required')
    return p


def _backend(c,device):return _train().load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)
def _load_role(root,role):return _train().load_received(Path(root)/role,role)
def _close_role(a):IQ.close_role(a)


def _references(c,identity):
    s,known_states,_,sc=macro._reference(c['signal_reference_root'],c['train_config']['control_train_config'])
    m=_read(Path(c['reference_prediction_root'])/'complete.json')
    expected=dict(schema=macro.SCHEMA,status=macro.STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,method_names=list(macro.METHODS),primary_method=macro.PRIMARY_METHOD,
        classification_path='P1',source_only=True,private_truth_reads=0,target_reads=0,controls_reused_exactly=True,identity_predictions_unchanged=True,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,model_parameters_buffers_unchanged=True,source_states_unchanged=True)
    if any(m.get(k)!=v for k,v in expected.items()) or m.get('reference_prediction_identity')!=s or m.get('model_identity')!=s['model_identity'] or m.get('parent_model_identity')!=s['model_identity']:raise ValueError('Macro12 control and fixed signal-r03 source must be identical parents')
    if s['parent_F0_identity']!=identity['parent_F0_identity'] or s['dura_identity']!=identity['dura_identity']:raise ValueError('New P0 and frozen P1 branches must share original F0 and fixed observation ancestry')
    with np.load(Path(c['reference_prediction_root'])/'models/signal_known.npz',allow_pickle=False) as z:known={k:z[k].copy() for k in z.files}
    if not _same(known,known_states['signal_known']):raise ValueError('P1 known12 control model must be the exact original P1 known8 state')
    with np.load(Path(c['signal_reference_root'])/'models/signal_density_ratio.npz',allow_pickle=False) as z:ratio={k:z[k].copy() for k in z.files}
    _core().validate_state(known,'known_only');_core().validate_state(ratio,'density_ratio')
    if any(not np.array_equal(v,ratio[k]) for k,v in known.items()):raise ValueError('Frozen P1 known/ratio controls must share exact known statistics')
    with np.load(Path(c['reference_prediction_root'])/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'} or z['conditional'].shape!=(4,6):raise ValueError('Complete original macro12 thresholds required')
        tau=z['conditional'][3].copy()
    cal=_read(Path(c['reference_prediction_root'])/'calibration/complete.json')
    if any(cal.get(k)!=v for k,v in dict(schema=macro.SCHEMA,status=macro.CALIBRATION_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,known_only=True,unknown_inputs_used=False).items()) or cal['methods']['signal_known12']['conditional_thresholds']!=tau.tolist():raise ValueError('Exact P1 known12 calibration control required')
    return m,s,known,ratio,tau,cal


def _reference_rows(c,role):
    # These helpers read only the requested public role. Never call a parent
    # scorer here: it would open novel before the three new calibrations freeze.
    a=macro._rows(c['reference_prediction_root'],role);s=macro._rows(c['signal_reference_root'],role)
    for k,v in a.items():
        if k not in ('scores','raw_scores') and not np.array_equal(v,s[k]):raise ValueError('Two fixed P1 parents disagree on original identity, metadata or feature rows')
    if not np.array_equal(a['raw_scores'][:,3],s['raw_scores'][:,1]):raise ValueError('Macro12 P1 known raw evidence must exactly match signal-r03')
    return a,s


def _iq_role(c,role,identity,parent_marker):
    a=_load_role(c['train_config']['IQ_root'],role);m=a['manifest']
    if (m!=parent_marker['source_IQ_manifests'][role] or m.get('parent_F0_identity')!=identity['parent_F0_identity'] or m.get('dura_identity')!=identity['dura_identity']
            or m.get('role')!=role or m.get('count')!=COUNTS[role] or m.get('received_input_path')!='P0' or a['received'].shape!=(COUNTS[role],2,256) or a['received'].dtype!=np.float32):
        _close_role(a);raise ValueError('Only identical fixed source G10 received P0 observations may feed the new branch')
    return a


def _align(a,old,role):
    if not np.array_equal(a['ids'],old['ids']):raise ValueError('Received IQ and fixed P1 parent IDs differ')
    for k,value in a['metadata'].items():
        expected=np.searchsorted([1,3,4,6,8],old['rx_i'])*3+old['day_i']-1 if k=='domain' else old[k]
        if not np.array_equal(value,expected):raise ValueError('Received IQ and fixed P1 parent metadata differ: '+k)
    # The received-only loader deliberately never opens restored/teacher or
    # cached P1 baseline arrays. Both completed parents retain the same P1
    # default prediction and baseline, verified in _reference_rows.


def _new_scores(z,states):
    t=time.perf_counter();raw=np.column_stack((_core().score(states[METHODS[2]],z,'known_only'),_core().score(states[METHODS[3]],z,'density_ratio')))
    if raw.shape!=(len(z),2) or not np.isfinite(raw).all():raise ValueError('Two finite same-P0-projection density scores required')
    return raw,time.perf_counter()-t


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable P0 evaluation output required')
    started=time.perf_counter();model,identity=_backend(c,device);_identity(identity,c)
    if sum(p.numel() for p in model.parameters())!=334721 or any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Actual frozen FP32 P0 headless signal and projection required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=_snapshot(model);smoke=_smoke(model,device);_unchanged(model,before)
    if smoke.get('status')!='PASS':raise ValueError('Actual P0 checkpoint synthetic smoke must pass before source roles')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_write(out/'resolved_config.json',c)
    pm,sm,p1known,p1ratio,old_tau,old_cal=_references(c,identity)
    seen=set();manifests={};fit_seconds=0.
    for role in ('L','OE'):
        a=None
        try:
            a=_iq_role(c,role,identity,pm)
            if seen&set(a['ids'].tolist()):raise ValueError('Source fitting roles overlap')
            seen.update(a['ids'].tolist());manifests[role]=a['manifest'];z=_project(model,a['received'],device);_unchanged(model,before);meta=a['metadata'];t=time.perf_counter()
            if role=='L':known,kdiag=_core().fit_known(z,meta['y'],meta['rx_i'],meta['day_i']);known_before=deepcopy(known)
            else:ratio,odiag=_core().fit_background(known,z,meta['rx_i'],meta['day_i'])
            fit_seconds+=time.perf_counter()-t
        finally:
            if a is not None:_close_role(a)
    if not _same(known,known_before) or any(not np.array_equal(v,ratio[k]) for k,v in known.items()):raise ValueError('New P0 known/ratio must share exact L-only statistics')
    states=dict(zip(METHODS,(p1known,p1ratio,known,ratio)));frozen=deepcopy(states)
    def check():
        _unchanged(model,before)
        if not _same(states,frozen):raise ValueError('Frozen P0/P1 known/background states changed')
    models=out/'models';models.mkdir()
    for name,state in states.items():_npz(models/(name+'.npz'),state)
    _write(models/'complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,method_names=list(METHODS),
        P1_models_reused=True,P0_known_statistics_shared=True,P1_known_statistics_shared=True,known=kdiag,background=odiag,source_V_fit=False,novel_fit=False))
    V,sv=_reference_rows(c,'known');a=None
    try:
        a=_iq_role(c,'V',identity,pm);_align(a,V,'V')
        if seen&set(a['ids'].tolist()):raise ValueError('V overlaps source fitting')
        seen.update(a['ids'].tolist());manifests['V']=a['manifest'];V['raw_signal_features']=_project(model,a['received'],device);check()
    finally:
        if a is not None:_close_role(a)
    raw,cpu_v=_new_scores(V['raw_signal_features'],states);all_raw=np.column_stack((V['raw_scores'][:,3],sv['raw_scores'][:,3],raw))
    tau=np.empty((4,6),np.float64);tau[0]=old_tau;methods={METHODS[0]:deepcopy(old_cal['methods']['signal_known12'])}
    for j in (1,2,3):
        state,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],all_raw[:,j],V['rx_i'],V['scenario']);tau[j]=state['conditional_thresholds'];methods[METHODS[j]]=report;check()
    _npz(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),known_only=True,unknown_inputs_used=False,target_reads=0,
        conditional_thresholds=tau.tolist(),methods=methods,P1_known12_calibration_reused=True,source_V_neural_fit=False,source_V_model_fit=False,source_V_threshold_fit=True))
    def combine(a,raw):
        old=a['scores'][:,3].copy();a['raw_scores']=raw;a['scores']=raw-tau[:,a['closed']].T
        if not np.array_equal(a['scores'][:,0],old):raise ValueError('Exact P1 known12 operating point changed')
    combine(V,all_raw);_npz(out/'known_predictions.npz',V);check();frozen_tau=tau.copy();print(CALIBRATION_STATUS,flush=True)
    N,sn=_reference_rows(c,'novel');a=None
    try:
        a=_iq_role(c,'novel',identity,pm);_align(a,N,'novel')
        if seen&set(a['ids'].tolist()):raise ValueError('Novel overlaps fitting/calibration records')
        manifests['novel']=a['manifest'];N['raw_signal_features']=_project(model,a['received'],device);check()
    finally:
        if a is not None:_close_role(a)
    raw,cpu_n=_new_scores(N['raw_signal_features'],states);combine(N,np.column_stack((N['raw_scores'][:,3],sn['raw_scores'][:,3],raw)));check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Thresholds changed after novel access')
    _npz(out/'novel_predictions.npz',N)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,primary_arm=PRIMARY_ARM,config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,classification_path='P1',rejection_input_path='P0',score_threshold=0.,
        model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],IQ_manifests=manifests,reference_prediction_identity=pm,signal_reference_identity=sm,
        reference_run_id=REFERENCE_RUN,reference_code_commit=REFERENCE_CODE,signal_reference_run_id=SIGNAL_REFERENCE_RUN,signal_reference_code_commit=SIGNAL_REFERENCE_CODE,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,
        real_signal_forward_rows=sum(COUNTS.values()),synthetic_signal_forward_rows=2,signal_forward_rows_by_role=dict(COUNTS),real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,G_checkpoint_loaded=False,
        native_updates=0,open_signal_updates=2220,eval_neural_updates=0,teacher_input_path='P0',teacher_real_F_forward_rows_in_eval=0,restored_P1_values_read=0,received_P0_values_read=sum(COUNTS.values()),channel_regenerations=0,second_RMS_applied=False,
        identity_predictions_unchanged=True,identity_baseline_path='P1',default_head_replaced=False,P1_known12_control_reused_exactly=True,P1_models_reused=True,P1_known_statistics_shared=True,P0_known_statistics_shared=True,same_projection_for_P0_methods=True,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,
        signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=567714,
        source_aggregate_bytes_by_method={k:sum(v.nbytes for v in s.values()) for k,s in states.items()},threshold_bytes=int(tau.nbytes),cpu_fit_seconds=fit_seconds,cpu_readout_seconds=cpu_v+cpu_n,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],cpu_readout_score_rows=2*(COUNTS['V']+COUNTS['novel']),
        training_teacher_cost_scope='Frozen original F0 OE-P0 teacher forwards belong to training, not eval; P1 controls and their prior training are inherited.',
        final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    check();_write(out/'complete.json',result);print(STATUS,flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_RAW_OPEN_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv))),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
