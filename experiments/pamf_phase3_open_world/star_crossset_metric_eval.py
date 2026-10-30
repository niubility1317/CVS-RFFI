"""Frozen cross-set metric evaluation with shared known-support/ratio statistics."""
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
from . import star_metric_residual_eval as previous
from . import star_class_risk as risk

RUN_ID='20261030-phase3-star-crossset-metric-s2026092701-r01'
SCHEMA='pamf_phase3_star_crossset_metric_prediction_v1'
STATUS='STAR_CROSSSET_METRIC_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CROSSSET_METRIC_THRESHOLDS_FROZEN'
DENSITY_STATUS='STAR_CROSSSET_METRIC_DENSITIES_FROZEN'
ARMS=('S_SEP','S_SEP_TOPO')
PRIMARY_ARM='S_SEP_TOPO'
METHODS=('fused160_density_ratio','learned_known_support','learned_density_ratio')
PRIMARY_METHOD=METHODS[1]
FIELDS=previous.FIELDS.copy()
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
KNOWN_FIELDS=previous.KNOWN_FIELDS.copy()
NOVEL_FIELDS=previous.NOVEL_FIELDS.copy()
REFERENCE_RUN,REFERENCE_CODE=previous.REFERENCE_RUN,previous.REFERENCE_CODE
reference=previous.reference
_read,_write,_npz,_same=previous._read,previous._write,previous._npz,previous._same
_snapshot,_unchanged,_project,_smoke=previous._snapshot,previous._unchanged,previous._project,previous._smoke
_reference,_reference_rows=previous._reference,previous._reference_rows


def _train():
    from . import star_crossset_metric_train
    return star_crossset_metric_train


def _core():return previous._core()
def _load_role(root,role):return _train().load_role(root,role)
def _backend(c,device):return _train().load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or not isinstance(c.get('train_config'),dict)
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('cache_root','reference_prediction_root','output_root'))):
        raise ValueError('Exact eight-field cross-set source evaluation config required; no truth/target/IQ')
    t=_train().validate_config(c['train_config'])
    if any(t.get(k)!=c[k] for k in ('run_id','code_commit','arm','cache_root')):raise ValueError('Actual row and prepared source cache differ')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in (c['cache_root'],c['reference_prediction_root'],t['output_root'])):
        raise ValueError('Immutable eval output must be separate from every input')
    return c


def _identity(i,c):
    required=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],updates=2220,new_parameters=102688,
        original_F0_parameters=232993,combined_active_parameters=335681,native_updates=0,original_parent_updates=11100,initialization_seed=2026092701,
        OE_embedding_training=True,OE_topology_regularization=c['arm']=='S_SEP_TOPO',crossset_separation=True,
        identity_head_replaced=False,input_path='cached_P1_features_only')
    if any(i.get(k)!=v for k,v in required.items()) or Path(i.get('checkpoint','')).resolve()!=(Path(c['train_config']['output_root'])/'final.pt').resolve():
        raise ValueError('Strict frozen 2220-step cross-set checkpoint required')
    p=i.get('parent_F0_identity',{});d=i.get('dura_identity',{})
    if (any(p.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(d.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items())):
        raise ValueError('Actual original F0/DURA source ancestry required')
    return p


def _role(root,role,identity):
    a=_load_role(root,role);m=a['manifest'];n=COUNTS[role];meta=a['metadata']
    if (m.get('schema')!='pamf_phase3_star_crossset_metric_cache_v1' or m.get('status')!='STAR_CROSSSET_METRIC_FEATURES_COMPLETE'
            or m.get('run_id')!=RUN_ID or m.get('code_commit')!=identity['code_commit']
            or m.get('parent_F0_identity')!=identity['parent_F0_identity'] or m.get('dura_identity')!=identity['dura_identity']
            or set(meta)!=({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
            or a['ids'].shape!=(n,) or a['u'].shape!=(n,160) or a['c'].shape!=(n,640)
            or a['u'].dtype!=np.float32 or a['c'].dtype!=np.float32 or not np.isfinite(a['u']).all() or not np.isfinite(a['c']).all()):
        raise ValueError('Only matching source L labels and public OE features/metadata may fit density')
    return a


def _learned_scores(z,states):
    t=time.perf_counter()
    raw=np.column_stack((_core().score(states[METHODS[1]],z,'known_only'),_core().score(states[METHODS[2]],z,'density_ratio')))
    if raw.shape!=(len(z),2) or raw.dtype!=np.float64 or not np.isfinite(raw).all():raise ValueError('Two finite same-projection learned scores required')
    return raw,time.perf_counter()-t


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable cross-set evaluation root required')
    started=time.perf_counter();model,identity=_backend(c,device);_identity(identity,c)
    if sum(p.numel() for p in model.parameters())!=102688 or any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Only the actual frozen FP32 102688-parameter cross-set projector is evaluated')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=_snapshot(model);smoke=_smoke(model,device);_unchanged(model,before)
    if smoke.get('status')!='PASS':raise ValueError('Actual checkpoint synthetic-feature smoke must precede source values')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_write(out/'resolved_config.json',c)
    L=_role(c['cache_root'],'L',identity);fm=L['manifest'];source=fm['source_feature_manifest']
    pm,f0_state,f0_tau,f0_global,pc=_reference(c['reference_prediction_root'],identity,source)
    z=_project(model,L['u'],L['c'],device);_unchanged(model,before);meta=L['metadata'];t=time.perf_counter()
    known,kdiag=_core().fit_known(z,meta['y'],meta['rx_i'],meta['day_i']);fit_seconds=time.perf_counter()-t
    known_before=deepcopy(known);seen=set(L['ids'].tolist());del L,z
    OE=_role(c['cache_root'],'OE',identity)
    if OE['manifest']!=fm or seen&set(OE['ids'].tolist()):raise ValueError('Source L/OE must share ancestry and be disjoint')
    seen.update(OE['ids'].tolist());z=_project(model,OE['u'],OE['c'],device);_unchanged(model,before);meta=OE['metadata'];t=time.perf_counter()
    ratio,odiag=_core().fit_background(known,z,meta['rx_i'],meta['day_i']);fit_seconds+=time.perf_counter()-t;del OE,z
    if not _same(known,known_before) or any(not np.array_equal(v,ratio[k]) for k,v in known.items()):raise ValueError('Both learned scores must share exactly the same source-L statistics')
    states=dict(zip(METHODS,(f0_state,known,ratio)));frozen=deepcopy(states)
    def check():
        _unchanged(model,before)
        if not _same(states,frozen):raise ValueError('Frozen projector/control/known/background state changed')
    models=out/'models';models.mkdir()
    for name,state in states.items():_npz(models/(name+'.npz'),state)
    _write(models/'complete.json',dict(status=DENSITY_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],method_names=list(METHODS),
        control_reused=True,known=kdiag,background=odiag,learned_known_statistics_shared=True,source_V_fit=False,novel_fit=False))
    V=_reference_rows(c['reference_prediction_root'],'known')
    if seen&set(V['ids'].tolist()):raise ValueError('V cannot overlap fitting records')
    seen.update(V['ids'].tolist());V['learned_features']=_project(model,V['unit_features'],V['contribution_features'],device);check()
    raw,cpu_known=_learned_scores(V['learned_features'],states);check()
    tau=[f0_tau];glob=[f0_global];methods={METHODS[0]:deepcopy(pc['methods'][METHODS[0]])};controls={METHODS[0]:deepcopy(pc['global_controls'][METHODS[0]])}
    for j,name in enumerate(METHODS[1:]):
        s,r=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],raw[:,j],V['rx_i'],V['scenario'])
        tau.append(s['conditional_thresholds']);glob.append(s['global_threshold']);methods[name]=r['methods']['predicted_class'];controls[name]=r['methods']['global_control']
    tau=np.asarray(tau,np.float64);glob=np.asarray(glob,np.float64);check()
    _npz(out/'thresholds.npz',dict(conditional=tau,**{'global':glob}));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],method_names=list(METHODS),
        primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,
        conditional_thresholds=tau.tolist(),global_thresholds=glob.tolist(),methods=methods,global_controls=controls,control_calibration_reused=True))
    f0_margin=V['scores'].copy();V['raw_scores']=np.column_stack((V['raw_scores'],raw));V['scores']=V['raw_scores']-tau[:,V['closed']].T
    if not np.array_equal(V['scores'][:,0],f0_margin):raise ValueError('Exact F0 known control changed')
    _npz(out/'known_predictions.npz',V);check();frozen_tau=tau.copy();print(CALIBRATION_STATUS,c['arm'],flush=True)
    N=_reference_rows(c['reference_prediction_root'],'novel')
    if seen&set(N['ids'].tolist()):raise ValueError('Novel source role overlaps fitting/calibration')
    N['learned_features']=_project(model,N['unit_features'],N['contribution_features'],device);check();raw,cpu_novel=_learned_scores(N['learned_features'],states);check()
    f0_margin=N['scores'].copy();N['raw_scores']=np.column_stack((N['raw_scores'],raw));N['scores']=N['raw_scores']-tau[:,N['closed']].T
    if not np.array_equal(tau,frozen_tau) or not np.array_equal(N['scores'][:,0],f0_margin):raise ValueError('Threshold/F0 control changed after novel access')
    _npz(out/'novel_predictions.npz',N);check()
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],primary_arm=PRIMARY_ARM,config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='P1',score_threshold=0.,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],feature_manifest=fm,source_feature_manifest=source,
        reference_run_id=REFERENCE_RUN,reference_code_commit=REFERENCE_CODE,reference_prediction_identity=pm,reference_prediction_root=c['reference_prediction_root'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,
        source_novel_fit_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,G_checkpoint_loaded=False,synthetic_open_forward_rows=2,
        open_projection_rows=sum(COUNTS.values()),open_projection_rows_by_role=dict(COUNTS),eval_neural_updates=0,native_updates=0,default_head_replaced=False,identity_predictions_unchanged=True,
        control_reused_exactly=True,control_refit=False,control_recalibrated=False,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,
        embedding_training_uses_OE=True,OE_topology_regularization=c['arm']=='S_SEP_TOPO',crossset_separation=True,density_background_uses_OE=True,
        learned_known_statistics_shared=True,same_projection_for_learned_methods=True,learned_known_statistic_role='L',learned_background_statistic_role='OE',
        metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=335681,source_aggregate_bytes_by_method={k:sum(a.nbytes for a in v.values()) for k,v in states.items()},
        threshold_bytes=int(tau.nbytes),cpu_fit_seconds=fit_seconds,cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],cpu_readout_score_rows=2*(COUNTS['V']+COUNTS['novel']),
        final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(out/'complete.json',result);print(STATUS,c['arm'],flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_CROSSSET_METRIC_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv))),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
