"""Source-only joint F0/signal density, with all three fixed DURA conditions."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from . import star_class_density_run as parent
from . import star_local_cov_run as frozen
from . import star_joint_feature_core as core
from . import star_signal_open_eval as signal_eval
from . import star_crossset_metric_train as source_features
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261031-phase3-joint-feature-s2026092701-r01'
SCHEMA='pamf_phase3_joint_feature_prediction_v1'
STATUS='JOINT_FEATURE_PREDICTIONS_COMPLETE'
MODELS_STATUS='JOINT_FEATURE_MODELS_FROZEN'
CALIBRATION_STATUS='JOINT_FEATURE_THRESHOLDS_FROZEN'
METHODS=core.METHODS
PRIMARY_METHOD=core.PRIMARY_METHOD
CONDITIONS=parent.CONDITIONS
COUNTS=parent.COUNTS
Q_COUNT=parent.Q_COUNT
TARGET_FIELDS=parent.TARGET_FIELDS|{'unit_features'}
_read,_write,_npz,_same=frozen._read,frozen._write,frozen._npz,frozen._same
_reference,_rows=frozen._reference,parent._rows
risk=parent.risk


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=frozen.FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source-only joint feature configuration required; no truth/U/K')
    p=parent.validate_config(c['reference_config'])
    if p['code_commit']!=frozen.PARENT_CODE:raise ValueError('Selected completed class-density parent required')
    out=Path(c['output_root']).resolve()
    for value in (p['output_root'],p['reference_prediction_root'],p['train_config']['IQ_root'],p['train_config']['feature_root'],p['train_config']['output_root']):
        source=Path(value).resolve()
        if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('New output must not overlap frozen inputs')
    return c


def _source_pairs(c,b,identity,device):
    tc=c['reference_config']['train_config'];result={};manifests={};seen=set();teacher_manifest=None
    for role in ('L','OE'):
        feature=source_features.load_role(tc['feature_root'],role);fm=feature['manifest'];iq=None
        if fm['parent_F0_identity']!=identity['parent_F0_identity'] or fm['dura_identity']!=identity['dura_identity']:
            raise ValueError('Source u160 must be from the original F0 on the same DURA P1')
        if teacher_manifest is not None and fm!=teacher_manifest:raise ValueError('Both teacher roles must share one source feature identity')
        teacher_manifest=fm
        try:
            iq=signal_eval._iq_role(tc,role,identity);signal_eval._align(iq,dict(ids=feature['ids'],**feature['metadata']),role)
            if iq['manifest']!=fm['source_manifests'][role] or seen&set(iq['ids'].tolist()):raise ValueError('Source teacher/IQ roles must align and be disjoint')
            seen.update(iq['ids'].tolist());z=signal_eval._project(b['model'],iq['restored'],device)
            result[role]=dict(u=feature['u'].copy(),z=z,metadata=deepcopy(feature['metadata']),ids=feature['ids'].copy());manifests[role]=deepcopy(iq['manifest'])
            print('JOINT_FEATURE_SOURCE',role,len(z),flush=True)
        finally:
            if iq is not None:signal_eval._close_role(iq)
    return result,dict(teacher=teacher_manifest,IQ=manifests)


def _current(b,values,device,with_signal=False):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(len(values),2,256) or not np.isfinite(values).all():raise ValueError('Complete finite FP32 full256 public IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(values.tolist(),dtype=torch.float32,device=device);restored=b['dura'][1](x)[0].detach().clone()
        if restored.dtype!=torch.float32 or restored.shape!=x.shape or not bool(torch.isfinite(restored).all()):raise ValueError('Same finite DURA output required')
        fused=b['reader'].features(restored);logits=b['reader'].classify_features(fused)
        z=b['model'](restored) if with_signal else None
    logits=np.asarray(logits.detach().cpu().tolist(),np.float32);fused=np.asarray(fused.detach().cpu().tolist(),np.float32)
    u=core.density.unit(fused).astype(np.float32)
    if logits.shape!=(len(values),6) or not np.isfinite(logits).all():raise ValueError('Finite original F0 default-six logits required')
    result=dict(unit_features=u,default_logits=logits,closed=logits.argmax(1).astype(np.int64))
    if with_signal:
        result['signal_features']=np.asarray(z.detach().cpu().tolist(),np.float32);core.join(u,result['signal_features'])
    return result


def _smoke(b,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32);a=_current(b,values,device,True)
    if len(a['closed'])!=2:raise ValueError('Two actual-checkpoint synthetic rows required')
    return dict(status='PASS',actual_F0_signal_DURA_checkpoints=True,joint_unit320=True,real_data_reads=0,
                G_synthetic_forward_rows=2,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,optimizer_updates=0)


def _target_rows(root,name):
    if name not in CONDITIONS:raise ValueError('Declared DURA target condition required')
    with np.load(Path(root)/'targets'/name/'predictions.npz',allow_pickle=False) as saved:a={k:saved[k].copy() for k in saved.files}
    if set(a)!=TARGET_FIELDS:raise ValueError('Exact public paired-feature query fields required; no truth')
    return a


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('Observe existing output; no automatic restart or replacement')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=c['reference_config'];b=parent._backend(p,device)
    snapshots=[(model,_snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=_smoke(b,device);unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Checkpoint smoke must pass before source data')
    pm,state,old_tau,old_cal=_reference(c);identity=pm['model_identity']
    if b['model_identity']!=identity or b['reader_identity']!=identity['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Actual original F0/r03/DURA must match all inherited predictions')
    original=deepcopy(state);out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    pairs,manifests=_source_pairs(c,b,identity,device);unchanged();L,O=pairs['L'],pairs['OE'];lm,om=L['metadata'],O['metadata']
    joint,diag=core.fit(L['u'],L['z'],lm['y'],lm['rx_i'],lm['day_i'],O['u'],O['z'],om['rx_i'],om['day_i'])
    fit_ids=set(L['ids'].tolist())|set(O['ids'].tolist());del pairs,L,O
    joint_before=deepcopy(joint)
    def check():
        unchanged()
        if not _same(state,original) or not _same(joint,joint_before):raise ValueError('Original or new frozen source density changed')
    check();(out/'models').mkdir();_npz(out/'models/density.npz',state);_npz(out/'models/joint.npz',joint)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],models_frozen=True,
        original_density_refitted=False,joint_density_fitted=True,source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_V_fit=False,novel_fit=False,target_fit=False,source_manifests=manifests,diagnostics=diag))
    tau=np.empty((4,6),np.float64);tau[:2]=old_tau;reports={METHODS[j]:deepcopy(old_cal['methods'][parent.METHODS[k]]) for j,k in enumerate((1,3))}
    def raw(a):
        result=core.raw_scores(state,joint,a['unit_features'],a['signal_features'],a['closed'])
        if not np.allclose(result[:,:2],a['raw_scores'][:,[1,3]],rtol=1e-12,atol=1e-9):raise ValueError('Selected original ratio scores must reconstruct')
        result[:,:2]=a['raw_scores'][:,[1,3]];return result
    V=_rows(p['output_root'],'known')
    if fit_ids&set(V['ids'].tolist()):raise ValueError('Known V cannot overlap source fitting roles')
    values=raw(V);check()
    for j in (2,3):
        fitted,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],values[:,j],V['rx_i'],V['scenario'])
        tau[j]=fitted['conditional_thresholds'];reports[METHODS[j]]=report;check()
    frozen_tau=tau.copy();_npz(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),methods=reports,conditional_thresholds=tau.tolist(),
        known_only=True,unknown_inputs_used=False,target_reads=0,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,shared_thresholds_for_all_conditions=True))
    _npz(out/'known_predictions.npz',frozen._combine(V,values,tau));print(CALIBRATION_STATUS,flush=True)
    N=_rows(p['output_root'],'novel');seen=fit_ids|set(V['ids'].tolist())
    if seen&set(N['ids'].tolist()):raise ValueError('Source novel cannot overlap fit/calibration')
    _npz(out/'novel_predictions.npz',frozen._combine(N,raw(N),tau));seen.update(N['ids'].tolist());check()
    for name in CONDITIONS:
        iq=None
        try:
            old=parent._target_rows(p['output_root'],name);manifest,ids,iq=parent._public(p,name)
            if not np.array_equal(ids,old['ids']) or seen&set(ids.tolist()):raise ValueError('Same source-disjoint parent Q physical records required')
            blocks=[_current(b,iq[start:start+128],device) for start in range(0,len(ids),128)]
            u=np.concatenate([a['unit_features'] for a in blocks]);logits=np.concatenate([a['default_logits'] for a in blocks]);closed=np.concatenate([a['closed'] for a in blocks])
            if not np.array_equal(closed,old['closed']) or not np.allclose(logits,old['default_logits'],rtol=1e-6,atol=1e-6):raise ValueError('Repeated same-DURA F0 logits/identity must reproduce the parent')
            a=dict(old,unit_features=u);folder=out/'targets'/name;folder.mkdir(parents=True);_npz(folder/'predictions.npz',frozen._combine(a,raw(a),tau));check()
            print('JOINT_FEATURE_Q_COMPLETE',name,len(ids),flush=True)
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Source thresholds changed after novel/Q access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        reference_prediction_identity=pm,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],classification_path='P1',
        identity_baseline_path='same_DURA_output_P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts=pm['Q_counts'],Q_manifests=pm['Q_manifests'],
        source_only=True,private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,original_U_s_reads=0,formal_target_accessed=False,
        neural_updates=0,density_fit_updates=1,joint_density_fit_updates=1,source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,source_states_unchanged=True,
        model_parameters_buffers_unchanged=True,controls_reused_exactly=True,identity_predictions_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        source_V_threshold_fit=True,real_signal_forward_rows=COUNTS['L']+COUNTS['OE'],real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,
        synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,Q_signal_new_forward_rows=0,source_F_new_forward_rows=0,
        final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    print(json.dumps(dict(event='JOINT_FEATURE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
