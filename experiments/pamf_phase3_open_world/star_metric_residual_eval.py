"""Source-only density evaluation of frozen learned metric residuals on cached features."""
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
from . import star_contribution_density_run as reference
from . import star_class_risk as risk

RUN_ID='20261030-phase3-star-metric-residual-s2026092701-r01'
SCHEMA='pamf_phase3_star_metric_residual_prediction_v1'
STATUS='STAR_METRIC_RESIDUAL_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_METRIC_RESIDUAL_THRESHOLDS_FROZEN'
ARMS=('M_L','M_LOE')
PRIMARY_ARM='M_LOE'
METHODS=('fused160_density_ratio','learned_metric_density_ratio')
PRIMARY_METHOD=METHODS[1]
REFERENCE_RUN=reference.RUN_ID
REFERENCE_CODE='edc28ad00baace5e4b9567fbf3391154bf6c34b4'
FIELDS={'run_id','code_commit','arm','train_config','cache_root','reference_prediction_root','output_root','batch_size'}
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
KNOWN_FIELDS=reference.KNOWN_FIELDS|{'learned_features'}
NOVEL_FIELDS=reference.NOVEL_FIELDS|{'learned_features'}
_read,_write,_npz,_same=reference._read,reference._write,reference._npz,reference._same


def _train():
    from . import star_metric_residual_train
    return star_metric_residual_train


def _core():return reference._core()


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or not isinstance(c.get('train_config'),dict)
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('cache_root','reference_prediction_root','output_root'))):
        raise ValueError('Exact eight-field source metric evaluation configuration required; no IQ/truth/target')
    t=_train().validate_config(c['train_config'])
    if any(t.get(k)!=c[k] for k in ('run_id','code_commit','arm','cache_root')):raise ValueError('Actual trained row and shared feature-cache identities differ')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in (c['cache_root'],c['reference_prediction_root'],t['output_root'])):
        raise ValueError('Evaluation output must be separate from immutable training/cache/reference inputs')
    return c


def _identity(identity,c):
    required=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],updates=2220,new_parameters=102688,
        original_F0_parameters=232993,combined_active_parameters=335681,native_updates=0,original_parent_updates=11100,
        initialization_seed=2026092701,OE_embedding_training=c['arm']=='M_LOE',identity_head_replaced=False,input_path='cached_P1_features_only')
    if any(identity.get(k)!=v for k,v in required.items()) or Path(identity.get('checkpoint','')).resolve()!=(Path(c['train_config']['output_root'])/'final.pt').resolve():
        raise ValueError('Strict frozen 2220-step metric checkpoint and original-F0 ancestry required')
    p=identity.get('parent_F0_identity',{});d=identity.get('dura_identity',{})
    if (any(p.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(d.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items())):
        raise ValueError('Only the fixed source original F0/DURA feature ancestry is permitted')
    return p


def _backend(c,device):return _train().load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)
def _load_role(root,role):return _train().load_role(root,role)
def _snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _unchanged(model,before):
    after=model.state_dict()
    if set(after)!=set(before) or any(not torch.equal(v,after[k].detach().cpu()) for k,v in before.items()):
        raise ValueError('Frozen learned metric parameters or buffers changed during evaluation')


def _tensor(x,device):return torch.tensor(np.asarray(x).tolist(),dtype=torch.float32,device=device)


def _project(model,u,c,device):
    values=[]
    for start in range(0,len(u),128):
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):z=model(_tensor(u[start:start+128],device),_tensor(c[start:start+128],device))
        a=np.asarray(z.detach().cpu().tolist(),np.float32)
        if a.shape!=(len(u[start:start+128]),160) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):
            raise ValueError('Frozen metric must produce finite same-row FP32 unit160 features')
        values.append(a)
    return np.concatenate(values)


def _smoke(model,device):
    t=np.arange(2*640,dtype=np.float32).reshape(2,640)
    u=np.cos(t[:,:160]*.017).astype(np.float32);c=np.sin(t*.023+.1).astype(np.float32)
    z=_project(model,u,c,device)
    if z.shape!=(2,160):raise ValueError('Actual new metric checkpoint synthetic feature smoke failed')
    return dict(status='PASS',actual_checkpoint=True,synthetic_open_forward_rows=2,synthetic_F_forward_rows=0,
        real_F_forward_rows=0,G_forward_rows=0,source_role_reads=0,target_reads=0,optimizer_updates=0)


def _role(root,role,identity):
    a=_load_role(root,role);m=a['manifest'];n=COUNTS[role];meta=a['metadata']
    if (m.get('status')!='STAR_METRIC_RESIDUAL_FEATURES_COMPLETE' or m.get('schema')!='pamf_phase3_star_metric_residual_cache_v1'
            or m.get('run_id')!=RUN_ID or m.get('code_commit')!=identity['code_commit']
            or m.get('parent_F0_identity')!=identity['parent_F0_identity'] or m.get('dura_identity')!=identity['dura_identity']
            or set(meta)!=({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
            or a['ids'].shape!=(n,) or a['u'].shape!=(n,160) or a['c'].shape!=(n,640)
            or a['u'].dtype!=np.float32 or a['c'].dtype!=np.float32 or not np.isfinite(a['u']).all() or not np.isfinite(a['c']).all()):
        raise ValueError('Matching source-only cached u/c pairs required; OE TX labels forbidden')
    return a


def _reference(root,identity,feature_manifest):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,schema=reference.SCHEMA,status=reference.STATUS,
        model_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],source_feature_manifest=feature_manifest,
        method_names=list(reference.METHODS),classification_path='P1',source_only=True,private_truth_reads=0,target_reads=0,
        identity_predictions_unchanged=True,control_reused_exactly=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Fixed complete original-F0 reference with identical source handles required')
    with np.load(root/'models/fused160_density_ratio.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    _core().validate_state(state,'density_ratio')
    if state['whitener'].shape!=(160,160):raise ValueError('Reference must remain the fixed original F0 160-dimensional model')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'][0].copy();global_tau=float(z['global'][0])
    cal=_read(root/'calibration/complete.json')
    if cal.get('run_id')!=REFERENCE_RUN or cal.get('code_commit')!=REFERENCE_CODE or cal.get('status')!=reference.CALIBRATION_STATUS or cal['methods'][METHODS[0]]['conditional_thresholds']!=tau.tolist():
        raise ValueError('Exact frozen F0 control thresholds/calibration required')
    return m,state,tau,global_tau,cal


def _reference_rows(root,role):
    fields=reference.KNOWN_FIELDS if role=='known' else reference.NOVEL_FIELDS
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Exact public frozen contribution-reference predictions required')
        a={k:z[k].copy() for k in z.files}
    reference.parent.validate_rows({k:v for k,v in a.items() if k!='contribution_features'},role)
    a['scores']=a['scores'][:,0].copy();a['raw_scores']=a['raw_scores'][:,0].copy()
    return a


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable metric evaluation output required')
    started=time.perf_counter();model,identity=_backend(c,device);_identity(identity,c)
    if (sum(p.numel() for p in model.parameters())!=102688 or any(m.training for m in model.modules())
            or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters())):
        raise ValueError('Only the frozen FP32 new 102688-parameter metric projection may evaluate cached features')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=_snapshot(model);smoke=_smoke(model,device);_unchanged(model,before)
    if smoke.get('status')!='PASS':raise ValueError('Actual metric checkpoint smoke must pass before source data')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_write(out/'resolved_config.json',c)
    L=_role(c['cache_root'],'L',identity);fm=L['manifest'];source_manifest=fm['source_feature_manifest']
    pm,control_state,control_tau,control_global,pc=_reference(c['reference_prediction_root'],identity,source_manifest)
    z=_project(model,L['u'],L['c'],device);_unchanged(model,before);meta=L['metadata'];t=time.perf_counter()
    state,kdiag=_core().fit_known(z,meta['y'],meta['rx_i'],meta['day_i']);fit_seconds=time.perf_counter()-t
    seen=set(L['ids'].tolist());del L,z
    OE=_role(c['cache_root'],'OE',identity)
    if OE['manifest']!=fm or seen&set(OE['ids'].tolist()):raise ValueError('L/OE input handles must match and roles remain disjoint')
    seen.update(OE['ids'].tolist());z=_project(model,OE['u'],OE['c'],device);_unchanged(model,before);meta=OE['metadata'];t=time.perf_counter()
    state,odiag=_core().fit_background(state,z,meta['rx_i'],meta['day_i']);fit_seconds+=time.perf_counter()-t;del OE,z
    states=dict(zip(METHODS,(control_state,state)));frozen=deepcopy(states)
    def check():
        _unchanged(model,before)
        if not _same(states,frozen):raise ValueError('Frozen control or learned density aggregate changed')
    models=out/'models';models.mkdir()
    for name,value in states.items():_npz(models/(name+'.npz'),value)
    _write(models/'complete.json',dict(status='STAR_METRIC_RESIDUAL_DENSITIES_FROZEN',run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],
        method_names=list(METHODS),control_reused=True,known=kdiag,background=odiag,source_V_fit=False,novel_fit=False))
    known=_reference_rows(c['reference_prediction_root'],'known')
    if seen&set(known['ids'].tolist()):raise ValueError('V cannot overlap source embedding/density fitting')
    seen.update(known['ids'].tolist());known['learned_features']=_project(model,known['unit_features'],known['contribution_features'],device);check()
    t=time.perf_counter();raw=_core().score(state,known['learned_features'],'density_ratio');cpu_known=time.perf_counter()-t
    new,report=risk.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw,known['rx_i'],known['scenario']);check()
    tau=np.stack((control_tau,new['conditional_thresholds']));global_tau=np.asarray([control_global,new['global_threshold']],np.float64)
    methods={METHODS[0]:deepcopy(pc['methods'][METHODS[0]]),METHODS[1]:report['methods']['predicted_class']}
    controls={METHODS[0]:deepcopy(pc['global_controls'][METHODS[0]]),METHODS[1]:report['methods']['global_control']}
    _npz(out/'thresholds.npz',dict(conditional=tau,**{'global':global_tau}));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,
        source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),global_thresholds=global_tau.tolist(),
        methods=methods,global_controls=controls,control_calibration_reused=True))
    control_margin=known['scores'].copy();known['raw_scores']=np.column_stack((known['raw_scores'],raw));known['scores']=known['raw_scores']-tau[:,known['closed']].T
    if not np.array_equal(known['scores'][:,0],control_margin):raise ValueError('Frozen F0 known control margins changed')
    _npz(out/'known_predictions.npz',known);check();frozen_tau=tau.copy();print(CALIBRATION_STATUS,c['arm'],flush=True)
    novel=_reference_rows(c['reference_prediction_root'],'novel')  # First novel open after density/model/calibration freeze.
    if seen&set(novel['ids'].tolist()):raise ValueError('Novel source role overlaps fitting/calibration')
    novel['learned_features']=_project(model,novel['unit_features'],novel['contribution_features'],device);check()
    t=time.perf_counter();raw=_core().score(state,novel['learned_features'],'density_ratio');cpu_novel=time.perf_counter()-t
    control_margin=novel['scores'].copy();novel['raw_scores']=np.column_stack((novel['raw_scores'],raw));novel['scores']=novel['raw_scores']-tau[:,novel['closed']].T
    if not np.array_equal(tau,frozen_tau) or not np.array_equal(novel['scores'][:,0],control_margin):raise ValueError('Frozen control/threshold changed after novel access')
    _npz(out/'novel_predictions.npz',novel);check()
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],primary_arm=PRIMARY_ARM,config=c,
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,classification_path='P1',score_threshold=0.,model_identity=identity,
        parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],feature_manifest=fm,source_feature_manifest=source_manifest,
        reference_run_id=REFERENCE_RUN,reference_code_commit=REFERENCE_CODE,reference_prediction_identity=pm,reference_prediction_root=c['reference_prediction_root'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],source_only=True,
        private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,
        G_forward_rows=0,G_checkpoint_loaded=False,synthetic_open_forward_rows=2,open_projection_rows=sum(COUNTS.values()),open_projection_rows_by_role=dict(COUNTS),
        eval_neural_updates=0,native_updates=0,default_head_replaced=False,identity_predictions_unchanged=True,
        control_reused_exactly=True,control_refit=False,control_recalibrated=False,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        source_V_model_fit=False,source_V_threshold_fit=True,embedding_training_uses_OE=c['arm']=='M_LOE',density_background_uses_OE=True,
        metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=335681,
        source_aggregate_bytes_by_method={k:sum(a.nbytes for a in v.values()) for k,v in states.items()},threshold_bytes=int(tau.nbytes),
        cpu_fit_seconds=fit_seconds,cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],
        final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(out/'complete.json',done);print(STATUS,c['arm'],flush=True);return done


def main():
    print(json.dumps(dict(event='STAR_METRIC_RESIDUAL_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
