"""Frozen structure-embedding cosine readout and its CPU source pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from . import star_class_risk as risk
from . import star_metric_embedding_score as oldscore
from .star_local_support_core import _features
from .star_class_risk_run import _read,_write,_npz
from .star_representation_launch import _env,_job,_write as _state_write

RUN_ID='20261030-phase3-star-metric-cosine-s2026092701-r01'
SCHEMA='pamf_phase3_star_metric_cosine_prediction_v1'
STATUS='STAR_METRIC_COSINE_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_METRIC_COSINE_THRESHOLDS_FROZEN'
DEPLOYMENT_SCHEMA='pamf_phase3_star_metric_cosine_deployment_v1'
DEPLOYMENT_STATUS='STAR_METRIC_COSINE_DEPLOYMENT_FROZEN'
DEPLOYMENT_FIELDS={'schema','status','run_id','code_commit','parameters','source_free','per_record_source_rows_retained'}
PARENT_RUN='20261030-phase3-star-metric-embedding-s2026092701-r01'
PARENT_COMMIT='efb82a4177e9b59533b7e71408ded6b63fc8733a'
PARENT_ARM='compact_oe_structure'
PARENT_SCHEMA=oldscore.evaluation.SCHEMA
PARENT_STATUS=oldscore.evaluation.STATUS
METHODS=('predicted_cosine','maximum_cosine')
PRIMARY_METHOD='maximum_cosine'
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
KNOWN_FIELDS=oldscore.evaluation.KNOWN_FIELDS
NOVEL_FIELDS=oldscore.evaluation.NOVEL_FIELDS
FIELDS={'run_id','code_commit','parent_prediction_root','output_root'}
RUNTIME_FIELDS={'run_id','code_commit','release_root','run_root','python','launch_owner','parent_prediction_root','novel_truth'}
RECEIVERS,SCENES=oldscore.RECEIVERS,oldscore.SCENES
_deployment,_reference_smoke=oldscore._deployment,oldscore._reference_smoke


def prototypes(class_means):
    means=np.asarray(class_means)
    if means.shape!=(6,160) or means.dtype!=np.float64 or not np.isfinite(means).all():
        raise ValueError('Six finite FP64 source-L class means required')
    norm=np.sqrt(np.einsum('nd,nd->n',means,means,optimize=False))
    if not np.isfinite(norm).all() or np.any(norm<=0):raise ValueError('Finite nonzero class-mean norms required for cosine prototypes')
    return means/norm[:,None]


def cosine_scores(metric_units,closed,prototype_units):
    units,norm=_features(metric_units,np.float32);closed=np.asarray(closed);p=np.asarray(prototype_units)
    if (closed.shape!=(len(units),) or closed.dtype!=np.int64 or not np.isin(closed,range(6)).all()
            or p.shape!=(6,160) or p.dtype!=np.float64 or not np.isfinite(p).all()
            or not np.allclose(np.sqrt(np.einsum('nd,nd->n',p,p,optimize=False)),1.,rtol=0,atol=2e-12)):
        raise ValueError('Original predicted identities and six frozen FP64 unit prototypes required')
    value=np.einsum('nd,kd->nk',units/norm[:,None],p,optimize=False);np.clip(value,-1.,1.,out=value)
    return np.column_stack((value[np.arange(len(units)),closed],value.max(axis=1)))


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or any(not isinstance(config[k],str) or not config[k].strip() for k in FIELDS)
            or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None):
        raise ValueError('Only the exact four-field source-only cosine configuration is allowed')
    if Path(config['output_root']).resolve().is_relative_to(Path(config['parent_prediction_root']).resolve()):
        raise ValueError('Cosine output must be outside the frozen parent output')
    return config


def _parent(root):
    marker=_read(root/'complete.json');identity=marker.get('model_identity',{});dura=marker.get('dura_identity',{})
    required=dict(schema=PARENT_SCHEMA,status=PARENT_STATUS,run_id=PARENT_RUN,code_commit=PARENT_COMMIT,arm=PARENT_ARM,
        primary_arm=PARENT_ARM,primary_method=oldscore.PRIMARY,classification_path='P1',default_head_replaced=False,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],g_parameters=20704,
        g_training_steps=1000,g_training_L_exposures=120000,g_training_OE_exposures=128000,g_forward_rows=248000+COUNTS['L']+COUNTS['V']+COUNTS['novel'],
        F_forward_rows=2,G_forward_rows=0,real_IQ_values_read=0,neural_F_updates=0,G_updates=0,source_U_reads=0,target_reads=0,
        source_novel_truth_reads=0,per_record_source_rows_retained=0,fresh_closed_vs_cached_baseline_changed_count=0)
    if (any(marker.get(k)!=v for k,v in required.items()) or marker.get('method_names')!=list(oldscore.METHODS)
            or any(marker.get(k) is not True for k in ('source_only','model_parameters_buffers_unchanged','metric_parameters_buffers_frozen_for_V_novel',
                'aggregate_frozen_before_V','all_thresholds_frozen_before_novel','source_free_deployment_state','parent_public_features_reused'))
            or any(identity.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(dura.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items())):
        raise ValueError('Only the completed fixed structure arm with original F0/DURA and legal source ancestry is allowed')
    handles=marker.get('cache_identities',{})
    if (set(handles)!=set(COUNTS) or any(handles[r].get('role')!=r or handles[r].get('count')!=n
            or handles[r].get('parent_F0_identity')!=identity or handles[r].get('dura_identity')!=dura for r,n in COUNTS.items())):
        raise ValueError('Matching already-validated source role handles required')
    return marker


def _parent_calibration(root,parent,tau):
    cal=_read(root/'calibration/complete.json')
    if (cal.get('schema')!=PARENT_SCHEMA or cal.get('status')!=oldscore.evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=parent[k] for k in ('run_id','code_commit','arm')) or cal.get('method_names')!=list(oldscore.METHODS)
            or cal.get('group_count')!=42 or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal.get('source_V_neural_fit') is not False or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or set(cal.get('methods',{}))!=set(oldscore.METHODS) or not np.array_equal(cal.get('conditional_thresholds'),tau)
            or any(not np.array_equal(cal['methods'][name].get('conditional_thresholds'),tau[j]) for j,name in enumerate(oldscore.METHODS))):
        raise ValueError('Frozen parent known-only twelve-threshold calibration identity required')


def _public(root,known):
    fields=KNOWN_FIELDS if known else NOVEL_FIELDS;n=COUNTS['V' if known else 'novel']
    with np.load(root/('known_predictions.npz' if known else 'novel_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Exact public prediction fields required; private novel labels forbidden')
        data={k:z[k].copy() for k in fields-{'raw_scores','scores'}}
    ids=data['ids'];closed=data['closed'];logits=data['default_logits']
    metadata_ok=all(data[k].shape==(n,) and data[k].dtype==np.int64 and np.isin(data[k],allowed).all()
        for k,allowed in (('rx_i',RECEIVERS),('day_i',(1,2,3))))
    if (not oldscore.common._ids(ids,n) or any(not oldscore._units(data[k],n) for k in ('unit_features','metric_unit_features'))
            or logits.shape!=(n,6) or logits.dtype!=np.float32 or not np.isfinite(logits).all()
            or closed.shape!=(n,) or closed.dtype!=np.int64 or not np.array_equal(closed,logits.argmax(1))
            or not metadata_ok):
        raise ValueError('Original F0 identities, public source metadata and frozen metric units required')
    if known and (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or not np.isin(data['y'],range(6)).all()
            or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
            or data['baseline_closed'].dtype!=np.int64 or not np.array_equal(data['baseline_closed'],closed)):
        raise ValueError('Known-only source labels and unchanged original F0 closed baseline required')
    return data


def run(config):
    validate_config(config);out=Path(config['output_root']);root=Path(config['parent_prediction_root'])
    if out.exists():raise FileExistsError('Observe the existing immutable cosine output; never overwrite')
    started=time.perf_counter();parent=_parent(root);net,metric,parent_tau=_deployment(root,parent);_parent_calibration(root,parent,parent_tau)
    frozen_net={k:v.detach().clone() for k,v in net.state_dict().items()}
    out.mkdir(parents=True);_reference_smoke(net,parent,out/'smoke.json');p=prototypes(metric['class_means']);frozen_p=p.copy();del metric,parent_tau
    def check():
        if not np.array_equal(p,frozen_p) or any(not torch.equal(net.state_dict()[k],v) for k,v in frozen_net.items()):
            raise ValueError('Frozen deployed metric or cosine prototypes changed')
    check();deployment=out/'deployment';deployment.mkdir();shutil.copyfile(root/'deployment/model.pt',deployment/'model.pt');_npz(deployment/'prototypes.npz',dict(prototypes=p))
    known=_public(root,True);started_score=time.perf_counter();raw=cosine_scores(known['metric_unit_features'],known['closed'],p);cpu=time.perf_counter()-started_score;check()
    tau=[];methods={}
    for j,name in enumerate(METHODS):
        values,report=risk.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw[:,j],known['rx_i'],known['scenario'])
        tau.append(values['conditional_thresholds']);methods[name]=report['methods']['predicted_class']
    tau=np.asarray(tau,np.float64)
    if tau.shape!=(2,6) or not np.isfinite(tau).all():raise ValueError('Twelve finite known-only cosine thresholds required')
    check();_npz(deployment/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditional_thresholds=tau.tolist(),methods=methods,group_count=42,
        known_only=True,unknown_inputs_used=False,source_V_neural_fit=False,source_only=True,target_reads=0))
    _npz(out/'known_predictions.npz',dict(known,raw_scores=raw,scores=raw-tau[:,known['closed']].T));frozen_tau=tau.copy()
    _write(deployment/'complete.json',dict(schema=DEPLOYMENT_SCHEMA,status=DEPLOYMENT_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
        parameters=20704,source_free=True,per_record_source_rows_retained=0));check();print(CALIBRATION_STATUS,flush=True)
    novel=_public(root,False)
    if set(known['ids'].tolist())&set(novel['ids'].tolist()):raise ValueError('Known and novel physical IDs must be disjoint')
    started_score=time.perf_counter();raw=cosine_scores(novel['metric_unit_features'],novel['closed'],p);cpu+=time.perf_counter()-started_score;check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Cosine thresholds changed after novel access')
    _npz(out/'novel_predictions.npz',dict(novel,raw_scores=raw,scores=raw-tau[:,novel['closed']].T))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],parent_run_id=PARENT_RUN,parent_code_commit=PARENT_COMMIT,
        parent_arm=PARENT_ARM,parent_prediction_root=str(root),method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        **{k:parent[k] for k in ('model_identity','dura_identity','cache_identities','cache_root','teacher_run_id','teacher_code_commit','teacher_root')},
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],g_parameters=20704,inherited_g_training_steps=1000,new_neural_updates=0,
        F_forward_rows=0,G_forward_rows=0,g_forward_rows=2,synthetic_unit_smoke_rows=2,real_IQ_values_read=0,source_L_forward_rows=0,source_OE_forward_rows=0,
        source_L_metadata_labels_read=0,target_reads=0,source_U_reads=0,source_novel_truth_reads=0,source_only=True,classification_path='P1',default_head_replaced=False,
        prototypes_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_free_deployment_state=True,parent_public_features_reused=True,
        deployed_g_unchanged=True,per_record_source_rows_retained=0,cpu_readout_seconds=cpu,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],
        fresh_closed_is_parent_reuse=True,fresh_closed_vs_cached_baseline_changed_count=0,A_N=None,H_OW=None,final_phase3_complete=False,seconds=time.perf_counter()-started)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


def _producer_config(r):
    return dict(run_id=RUN_ID,code_commit=r['code_commit'],parent_prediction_root=r['parent_prediction_root'],output_root=str(Path(r['run_root'])/'predict'))


def validate_runtime(r):
    if (not isinstance(r,dict) or set(r)!=RUNTIME_FIELDS or r.get('run_id')!=RUN_ID
            or any(not isinstance(r[k],str) or not r[k].strip() for k in RUNTIME_FIELDS)
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None):
        raise ValueError('Only the exact eight-field CPU-only cosine runtime is allowed')
    validate_config(_producer_config(r));return r


def start(runtime):
    r=validate_runtime(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing run; never relaunch or overwrite')
    root.mkdir();(root/'logs').mkdir()
    command=[r['python'],'-B','-u','-m',__package__+'.star_metric_cosine','pipeline','--runtime',str(runtime)]
    with (root/'logs/pipeline.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(run_id=RUN_ID,code_commit=r['code_commit'],pid=child.pid,cwd=r['release_root'],argv=command,time=time.time(),launch_owner=r['launch_owner'])
    _write(root/'launch.json',launch);print(json.dumps(launch),flush=True)


def pipeline(runtime):
    r=validate_runtime(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN_ID,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state)
    try:
        _write(root/'config.json',_producer_config(r))
        commands=[('predict',[r['python'],'-B','-u','-m',__package__+'.star_metric_cosine','predict','--config',str(root/'config.json')]),
            ('score',[r['python'],'-B','-u','-m',__package__+'.star_metric_cosine_score','--prediction-root',str(root/'predict'),'--novel-truth',r['novel_truth'],'--output',str(root/'score')])]
        for name,command in commands:
            if not _job(r,state,path,name,command,None):raise RuntimeError(name+' failed; preserve partial output without restart')
        prediction=_read(root/'predict/complete.json');score=_read(root/'score/complete.json')
        passed=score.get('source_gate_passed');next_step='SOURCE_METRIC_COSINE_READY_FOR_TARGET_FALSIFICATION' if passed else 'NO_PROMOTION'
        if (prediction.get('status')!=STATUS or score.get('status')!='STAR_METRIC_COSINE_SCORES_COMPLETE'
                or any(v.get('run_id')!=RUN_ID or v.get('code_commit')!=r['code_commit'] or v.get('primary_method')!=PRIMARY_METHOD for v in (prediction,score))
                or type(passed) is not bool or score.get('eligible_primary') is not passed or score.get('next_step')!=next_step):
            raise ValueError('Exact completed source prediction/score identities and fixed-primary terminal fields required')
        state.update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=passed,eligible_primary=passed,next_step=next_step,
            primary=PRIMARY_METHOD,target_reads=0,final_phase3_complete=False,CPU_only=True,GPU_processes_added=0,foreign_processes_stopped=0)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_state_write(path,state);print(state['status'],flush=True);return state


def main(argv=None):
    print(json.dumps(dict(event='STAR_METRIC_COSINE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('start','pipeline'):sub.add_parser(mode).add_argument('--runtime',required=True)
    sub.add_parser('predict').add_argument('--config',required=True);args=parser.parse_args(argv)
    if args.mode=='predict':run(_read(args.config));return 0
    if args.mode=='start':start(args.runtime);return 0
    return 0 if pipeline(args.runtime)['status']=='ANALYZED_SOURCE_ONLY' else 1


if __name__=='__main__':raise SystemExit(main())
