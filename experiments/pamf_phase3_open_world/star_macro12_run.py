"""Calibration-only C12/C42 comparison of two fixed source known densities."""
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
from . import star_signal_open_eval as source
from . import star_macro12_risk as risk

RUN_ID=risk.RUN_ID
SCHEMA='pamf_phase3_star_macro12_prediction_v1'
STATUS='STAR_MACRO12_PREDICTIONS_COMPLETE'
MODELS_STATUS='STAR_MACRO12_MODELS_FROZEN'
CALIBRATION_STATUS='STAR_MACRO12_THRESHOLDS_FROZEN'
REFERENCE_RUN='20261030-phase3-star-signal-open-s2026092701-r03'
REFERENCE_CODE='dc01a402582f93e452dbaa3ef59a9fbb44151818'
METHODS=('frozen_known42','frozen_known12','signal_known42','signal_known12')
PRIMARY_METHOD=METHODS[3]
PRIMARY_INDEX=3
MODEL_NAMES=('frozen_known','signal_known')
METHOD_MODEL_INDEX=(0,0,1,1)
FIELDS={'run_id','code_commit','reference_prediction_root','train_config','output_root'}
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
KNOWN_FIELDS,NOVEL_FIELDS=source.KNOWN_FIELDS,source.NOVEL_FIELDS
_read,_write,_npz,_same=source._read,source._write,source._npz,source._same
_core=source._core
_smoke=source._smoke
ZEROS=('private_truth_reads','target_reads','original_U_s_reads','source_L_reads','source_OE_reads','source_L_IQ_reads','source_OE_IQ_reads',
    'source_L_feature_reads','source_OE_feature_reads','neural_updates','new_neural_updates','native_updates','density_fit_updates','score_normalizer_fit_updates',
    'real_signal_forward_rows','real_F_forward_rows','synthetic_F_forward_rows','G_forward_rows','received_P0_values_read','channel_regenerations')


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or not isinstance(c.get('train_config'),dict) or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('reference_prediction_root','output_root'))):
        raise ValueError('Exact five-field source calibration-only configuration required; no truth/IQ/U/Q/K inputs')
    t=source._train().validate_config(c['train_config'])
    if t['run_id']!=REFERENCE_RUN or t['code_commit']!=REFERENCE_CODE or t['arm']!='E_SIGNAL':raise ValueError('Fixed original r03 signal checkpoint configuration required')
    inputs=[c['reference_prediction_root'],t['output_root'],t['IQ_root'],t['feature_root'],t['native_config']['source_output'],t['native_config']['source_cache']]
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in inputs):raise ValueError('Immutable calibration output must be separate from every frozen input')
    return c


def _backend(c,device):return source._train().load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)


def _reference(root,train_config):
    """Only parent metadata, two known8 aggregates and frozen thresholds."""
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=source.SCHEMA,status=source.STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,arm='E_SIGNAL',primary_arm='E_SIGNAL',
        method_names=list(source.METHODS),primary_method=source.PRIMARY_METHOD,classification_path='P1',known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,
        model_parameters_buffers_unchanged=True,source_V_model_fit=False,source_V_threshold_fit=True)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Completed fixed r03 four-readout source signal predictions required')
    c=source.validate_config(m['config'])
    if (c['train_config']!=train_config or Path(c['output_root']).resolve()!=root.resolve()
            or source._identity(m['model_identity'],c)!=m['parent_F0_identity'] or m['dura_identity']!=m['model_identity']['dura_identity']):
        raise ValueError('Exact original checkpoint and source ancestry must match the reference')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(status=source.DENSITY_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,arm='E_SIGNAL',method_names=list(source.METHODS),
            controls_reused=True,signal_known_statistics_shared=True,source_V_fit=False,novel_fit=False).items()):
        raise ValueError('Both parent source known-only models must be frozen')
    states={}
    for name,oldname in zip(MODEL_NAMES,source.METHODS[:2]):
        with np.load(root/'models'/(oldname+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        _core().validate_state(state,'known_only')
        if state['whitener'].shape!=(160,160):raise ValueError('Only fixed D160 known8 states allowed')
        for v in state.values():v.setflags(write=False)
        states[name]=state
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional','global'}:raise ValueError('Complete parent conditional/global thresholds required')
        tau=z['conditional'].copy();glob=z['global'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or glob.shape!=(4,) or glob.dtype!=np.float64 or not np.isfinite(glob).all()
            or any(cal.get(k)!=v for k,v in dict(schema=source.SCHEMA,status=source.CALIBRATION_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,arm='E_SIGNAL',
                method_names=list(source.METHODS),primary_method=source.PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,
                source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0).items()) or not np.array_equal(tau,cal.get('conditional_thresholds'))):
        raise ValueError('Exact parent known42 calibration and finite thresholds required')
    for j,name in enumerate(source.METHODS[:2]):
        if cal.get('methods',{}).get(name,{}).get('conditional_thresholds')!=tau[j].tolist():raise ValueError('Original known42 threshold/report mismatch')
    return m,states,tau[:2].copy(),cal


def _rows(root,role):
    if role not in ('known','novel'):raise ValueError('Only completed known/novel public predictions allowed')
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    source._validate_rows(a,role,4,True)
    return a


def _parent_margins(a,old_tau):
    if not np.array_equal(a['scores'][:,:2],a['raw_scores'][:,:2]-old_tau[:,a['closed']].T):raise ValueError('Parent raw/known42 margin or class routing mismatch')


def _combine(a,tau):
    result=dict(a);result['raw_scores']=a['raw_scores'][:,[0,0,1,1]].copy();result['scores']=result['raw_scores']-tau[:,a['closed']].T
    if not np.isfinite(result['scores']).all() or not np.array_equal(result['scores'][:,[0,2]],a['scores'][:,:2]):raise ValueError('Exact known42 margins must remain unchanged')
    if np.any((result['scores'][:,[1,3]]>=0)&(result['scores'][:,[0,2]]<0)):raise ValueError('New C12 acceptances must be subsets of known42 acceptances')
    return result


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable calibration-only output root required')
    started=time.perf_counter();model,identity=_backend(c,device)
    source._identity(identity,dict(code_commit=REFERENCE_CODE,train_config=c['train_config']))
    if sum(p.numel() for p in model.parameters())!=334721 or any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Actual frozen parent signal checkpoint required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=source._snapshot(model);smoke=_smoke(model,device);source._unchanged(model,before)
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_checkpoint=True,synthetic_signal_forward_rows=2,real_signal_forward_rows=0,G_forward_rows=0).items()):
        raise ValueError('Actual parent checkpoint synthetic smoke must pass before cache operations')
    pm,states,old_tau,pc=_reference(c['reference_prediction_root'],c['train_config'])
    if identity!=pm['model_identity']:raise ValueError('Loaded actual checkpoint differs from completed parent predictions')
    frozen=deepcopy(states);tau=np.empty((4,6),np.float64);tau[[0,2]]=old_tau
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);(out/'models').mkdir()
    for name,state in states.items():_npz(out/'models'/(name+'.npz'),state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_names=list(MODEL_NAMES),
        method_model_index=list(METHOD_MODEL_INDEX),models_frozen=True,models_refitted=False,source_only=True,source_V_fit=False,novel_fit=False,
        reference_prediction_identity=pm,parent_model_identity=identity,target_reads=0))
    def check():
        source._unchanged(model,before)
        if not _same(states,frozen):raise ValueError('Frozen source known-only statistics changed')
    V=_rows(c['reference_prediction_root'],'known');_parent_margins(V,old_tau);check()
    methods={METHODS[0]:deepcopy(pc['methods'][source.METHODS[0]]),METHODS[2]:deepcopy(pc['methods'][source.METHODS[1]])}
    all_names=list(risk.HARD_NAMES+risk.DIAGNOSTIC_NAMES)
    method_hard={name:all_names if j%2==0 else list(risk.HARD_NAMES) for j,name in enumerate(METHODS)}
    for j in (1,3):
        state,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j//2],V['rx_i'],V['scenario'])
        tau[j]=state['conditional_thresholds'];methods[METHODS[j]]=report;check()
    if np.any(tau[[1,3]]<tau[[0,2]]):raise ValueError('Identical parent raw scores require componentwise nondecreasing C12 thresholds')
    _npz(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        known_count=COUNTS['V'],conditional_thresholds=tau.tolist(),methods=methods,method_hard_groups=method_hard,
        group_count=42,hard_group_counts=[42,12,42,12],diagnostic_group_counts=[0,30,0,30],hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),
        known_only=True,source_only=True,unknown_inputs_used=False,novel_inputs_used=False,source_V_neural_fit=False,source_V_normalizer_fit=False,source_V_threshold_fit=True,target_reads=0,
        reference_calibration_identity=pc,controls_reused_exactly=True))
    _npz(out/'known_predictions.npz',_combine(V,tau));check();frozen_tau=tau.copy();print(CALIBRATION_STATUS,flush=True)
    N=_rows(c['reference_prediction_root'],'novel');_parent_margins(N,old_tau)
    if set(V['ids'].tolist())&set(N['ids'].tolist()):raise ValueError('Known and novel physical IDs overlap')
    _npz(out/'novel_predictions.npz',_combine(N,tau));check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Thresholds changed after novel access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],counts=dict(COUNTS),
        reference_run_id=REFERENCE_RUN,reference_code_commit=REFERENCE_CODE,reference_prediction_root=c['reference_prediction_root'],reference_prediction_identity=pm,
        parent_model_identity=identity,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        source_feature_manifest=pm['source_feature_manifest'],source_feature_handle=pm['feature_manifest'],source_IQ_manifests=pm['IQ_manifests'],
        model_names=list(MODEL_NAMES),method_model_index=list(METHOD_MODEL_INDEX),model_aggregate_bytes={k:sum(v.nbytes for v in s.values()) for k,s in states.items()},threshold_bytes=int(tau.nbytes),
        source_only=True,source_V_labels_used=True,source_V_threshold_fit=True,source_V_model_fit=False,source_novel_fit_reads=0,
        synthetic_signal_forward_rows=2,identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        models_refitted=False,controls_reused_exactly=True,control_refit=False,control_recalibrated=False,model_parameters_buffers_unchanged=True,source_states_unchanged=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,
        parent_open_signal_updates=2220,seconds=time.perf_counter()-started,**{k:0 for k in ZEROS})
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main(argv=None):
    print(json.dumps(dict(event='STAR_MACRO12_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    run(_read(a.config),a.device)


if __name__=='__main__':main()
