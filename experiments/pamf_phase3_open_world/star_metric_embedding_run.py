"""Fixed-budget source-only metric learning with a row-free deployment export."""
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

from . import star_conditional_support_run as base
from . import star_class_risk as risk
from . import star_metric_embedding_core as core

RUN_ID='20261030-phase3-star-metric-embedding-s2026092701-r01'
SCHEMA='pamf_phase3_star_metric_embedding_prediction_v1'
STATUS='STAR_METRIC_EMBEDDING_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_METRIC_EMBEDDING_THRESHOLDS_FROZEN'
DEPLOYMENT_SCHEMA='pamf_phase3_star_metric_embedding_deployment_v1'
DEPLOYMENT_STATUS='STAR_METRIC_EMBEDDING_DEPLOYMENT_FROZEN'
TRAINING_SCHEMA='pamf_phase3_star_metric_embedding_training_v1'
TRAINING_STATUS='STAR_METRIC_EMBEDDING_TRAINING_COMPLETE'
ARMS,METHODS,PRIMARY_ARM,PRIMARY_METHOD=core.ARMS,core.METHODS,core.PRIMARY_ARM,core.PRIMARY_METHOD
TRAIN_STEPS,PARAMETERS,METRIC_FIELDS=core.TRAIN_STEPS,core.PARAMETERS,core.METRIC_FIELDS
MODEL_FIELDS=('fc1.weight','fc1.bias','fc2.weight','fc2.bias')
DEPLOYMENT_FIELDS={'schema','status','run_id','code_commit','arm','parameters','source_free','per_record_source_rows_retained'}
TRACE_FIELDS={'step','total','known_contrast','OE_repulsion','OE_structure','g_grad_norm','L_indices','OE_indices','L_exposures','OE_exposures','L_unique','OE_unique'}
MODEL_SEED,SAMPLING_SEED=2026092701,2026103002
FIELDS=base.FIELDS|{'arm'}
SOURCE_FIELDS,COUNTS=base.SOURCE_FIELDS,base.COUNTS
RECEIVERS,SCENES=base.RECEIVERS,base.SCENES
KNOWN_FIELDS=base.KNOWN_FIELDS|{'metric_unit_features'}
NOVEL_FIELDS=base.NOVEL_FIELDS|{'metric_unit_features'}
_backend,_smoke,_snapshot,_check_models,_identity=base._backend,base._smoke,base._snapshot,base._check_models,base._identity
_same,_write,_npz,_read=base._same,base._write,base._npz,base._read
_parent,_bank,_public=base._parent,base._bank,base._public
_losses,_sample_indices,_fit_metric,_metric_scores=core.losses,core.sample_indices,core.fit_metric,core.metric_scores


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID or config.get('arm') not in ARMS
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or not isinstance(config.get('source_config'),dict) or set(config['source_config'])!=SOURCE_FIELDS
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('parent_prediction_root','cache_root','output_root'))):
        raise ValueError('Only the eight fixed-arm source-only metric configuration fields are allowed')
    source=config['source_config'];base.base.base.runner.validate_config('calibrate',dict(source,retention=.99))
    if (source['run_id']!=RUN_ID or source['row_id']!='star-metric-embedding-'+config['arm']+'-s2026092701'
            or source['code_commit']!=config['code_commit'] or source['output_root']!=config['output_root']
            or source['batch_size']!=128 or source['model_seed']!=MODEL_SEED or source['registered_classes']!=base.base.base.native.CLASSES):
        raise ValueError('Matching arm identity and the complete original-F0 source contract required')
    out=Path(config['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) for p in (config['parent_prediction_root'],config['cache_root'],source['source_output'],source['source_cache'])):
        raise ValueError('Output cannot overwrite the parent or source feature inputs')
    return config


def _sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def _new_model(device):
    torch.manual_seed(MODEL_SEED)
    model=core.MetricNet().float()
    if set(model.state_dict())!=set(MODEL_FIELDS) or sum(p.numel() for p in model.parameters())!=PARAMETERS:
        raise ValueError('Only the fixed four-tensor 20704-parameter metric network is allowed')
    return model.to(device)


def _tensor(array,device,dtype=torch.float32):
    return torch.tensor(np.asarray(array).tolist(),dtype=dtype,device=device)


def _train(model,bank,arm,device,trace_path,config):
    started=time.perf_counter();rng=np.random.default_rng(SAMPLING_SEED);steps=0 if arm=='identity' else TRAIN_STEPS
    seen_L=set();seen_OE=set();initial=final=None;optimizer=None
    if steps:
        model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    with Path(trace_path).open('x',encoding='utf-8') as stream:
        for step in range(1,steps+1):
            li,oi=_sample_indices(bank['L_y'],bank['L_rx_i'],len(bank['OE_unit_features']),rng)
            xL=_tensor(bank['L_unit_features'][li],device);xOE=_tensor(bank['OE_unit_features'][oi],device)
            y=_tensor(bank['L_y'][li],device,torch.int64);rx=_tensor(bank['L_rx_i'][li],device,torch.int64)
            optimizer.zero_grad(set_to_none=True);loss=_losses(model(xL),y,rx,model(xOE),xOE,arm)
            if set(loss)!={'total','known_contrast','OE_repulsion','OE_structure'} or any(not torch.isfinite(v).all() for v in loss.values()):
                raise ValueError('Finite fixed-objective metric losses required')
            values={k:float(v.detach().cpu().item()) for k,v in loss.items()};loss['total'].backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Finite metric-only gradients required')
            grad_norm=float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt().cpu().item())
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Metric parameters became nonfinite')
            seen_L.update(li.tolist());seen_OE.update(oi.tolist());initial=values if initial is None else initial;final=values
            row=dict(step=step,**values,g_grad_norm=grad_norm,L_indices=li.tolist(),OE_indices=oi.tolist(),
                L_exposures=step*120,OE_exposures=step*128,L_unique=len(seen_L),OE_unique=len(seen_OE))
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if step==1 or step%100==0 or step==steps:
                print('STAR_METRIC_EMBEDDING_TRAIN_PROGRESS',arm,step,steps,json.dumps(values),flush=True)
    _sync(device)
    return dict(schema=TRAINING_SCHEMA,status=TRAINING_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=arm,g_parameters=PARAMETERS,
        g_training_steps=steps,g_training_L_exposures=steps*120,g_training_OE_exposures=steps*128,g_training_L_unique=len(seen_L),g_training_OE_unique=len(seen_OE),
        g_training_forward_rows=steps*248,model_seed=MODEL_SEED,sampling_seed=SAMPLING_SEED,loss_initial=initial,loss_final=final,
        optimizer=None if not steps else dict(name='Adam',lr=.001,weight_decay=.0001),initialization_cpu_seeded=True,F_updates=0,G_updates=0,
        source_only=True,target_reads=0,source_OE_TX_or_K_used=False,source_L_TX_RX_labels_used=True,trace_rows=steps,
        train_seconds=time.perf_counter()-started,g_parameters_finite=bool(all(torch.isfinite(p).all() for p in model.parameters())))


def _transform(model,units,device,batch_size=128):
    result=[]
    with torch.no_grad():
        for start in range(0,len(units),batch_size):
            output=model(_tensor(units[start:start+batch_size],device))
            result.append(np.asarray(output.detach().cpu().tolist(),dtype=np.float32))
    result=np.concatenate(result)
    if (result.shape!=units.shape or not np.isfinite(result).all()
            or not np.allclose(np.linalg.norm(result.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):
        raise ValueError('Finite per-record FP32 unit metric features required')
    return result


def _predict(model,data,metric,device,batch_size):
    _sync(device);started=time.perf_counter();units=_transform(model,data['unit_features'],device,batch_size);_sync(device)
    evaluation=time.perf_counter()-started;started=time.perf_counter();raw=_metric_scores(units,data['closed'],metric);readout=time.perf_counter()-started
    if raw.shape!=(len(data['ids']),2) or raw.dtype!=np.float64 or not np.isfinite(raw).all():raise ValueError('Two finite fixed metric scores required')
    return units,raw,evaluation,readout


def _export(data,units,raw,tau):
    result={k:v for k,v in data.items() if k not in ('raw_scores','scores')}
    result.update(metric_unit_features=units,raw_scores=raw,scores=raw-tau[:,data['closed']].T)
    return result


def run(config,device='cuda:0'):
    validate_config(config);out=Path(config['output_root']);arm=config['arm']
    if out.exists():raise FileExistsError('A new immutable metric embedding output root is required')
    started=time.perf_counter();backend,_=_backend(config['source_config'],device);identity=backend['reader_identity'];_identity(identity,config['source_config'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=_snapshot(backend);smoke=_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual original-F0 checkpoint smoke must pass before source feature consumption')
    if any(p.requires_grad or p.grad is not None for p in backend['reader'].parameters()):raise ValueError('Original F0 must remain frozen and gradient-free')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke)
    parent=_parent(config,identity);bank=_bank(config,parent);seen=set(bank['L_ids'].tolist())|set(bank['OE_ids'].tolist())
    model=_new_model(device);training=_train(model,bank,arm,device,out/'training.jsonl',config);_check_models(snapshots)
    model.eval().requires_grad_(False);model_frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
    L_units=_transform(model,bank['L_unit_features'],device,config['batch_size']);metric=_fit_metric(L_units,bank['L_y']);metric_frozen=deepcopy(metric)
    if set(metric)!=set(METRIC_FIELDS):raise ValueError('Only four fixed source aggregates may be exported')
    def check():
        _check_models(snapshots)
        if any(not torch.equal(model.state_dict()[k],v) for k,v in model_frozen.items()) or not _same(metric,metric_frozen):
            raise ValueError('Metric parameters/buffers or aggregate changed during V/novel inference')
    check();deployment=out/'deployment';deployment.mkdir();torch.save(model.state_dict(),deployment/'model.pt');_npz(deployment/'metric.npz',metric)
    _write(out/'training_complete.json',training)
    del L_units,bank
    known=_public(config,'known',seen);seen.update(known['ids'].tolist());units,raw,g_known,cpu_known=_predict(model,known,metric,device,config['batch_size']);check()
    thresholds=[];methods={}
    for j,name in enumerate(METHODS):
        values,report=risk.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],raw[:,j],known['rx_i'],known['scenario'])
        thresholds.append(values['conditional_thresholds']);methods[name]=report['methods']['predicted_class']
    tau=np.asarray(thresholds,np.float64)
    if tau.shape!=(2,6) or not np.isfinite(tau).all():raise ValueError('Twelve finite source-known metric thresholds required')
    check();_npz(deployment/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=arm,
        method_names=list(METHODS),conditional_thresholds=tau.tolist(),methods=methods,group_count=42,source_only=True,target_reads=0,
        known_only=True,unknown_inputs_used=False,source_V_neural_fit=False,source_L_TX_RX_labels_used=True,source_OE_TX_or_K_used=False))
    _npz(out/'known_predictions.npz',_export(known,units,raw,tau));frozen_tau=tau.copy()
    _write(deployment/'complete.json',dict(schema=DEPLOYMENT_SCHEMA,status=DEPLOYMENT_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
        arm=arm,parameters=PARAMETERS,source_free=True,per_record_source_rows_retained=0))
    if {p.name for p in deployment.iterdir()}!={'model.pt','metric.npz','thresholds.npz','complete.json'}:raise ValueError('Exactly four source-free deployment files required')
    check();print(CALIBRATION_STATUS,arm,flush=True)
    novel=_public(config,'novel',seen);units,raw,g_novel,cpu_novel=_predict(model,novel,metric,device,config['batch_size']);check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Metric thresholds changed after novel access')
    _npz(out/'novel_predictions.npz',_export(novel,units,raw,tau))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=arm,method_names=list(METHODS),primary_arm=PRIMARY_ARM,primary_method=PRIMARY_METHOD,
        parent_run_id=base.PARENT_RUN,parent_code_commit=base.PARENT_COMMIT,parent_prediction_root=config['parent_prediction_root'],cache_root=config['cache_root'],
        teacher_run_id=base.base.TEACHER_RUN,teacher_code_commit=base.base.TEACHER_COMMIT,teacher_root=parent['teacher_root'],
        model_identity=identity,dura_identity=parent['dura_identity'],cache_identities=parent['cache_identities'],known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],g_parameters=PARAMETERS,g_training_steps=training['g_training_steps'],
        g_training_L_exposures=training['g_training_L_exposures'],g_training_OE_exposures=training['g_training_OE_exposures'],
        g_forward_rows=training['g_training_forward_rows']+COUNTS['L']+COUNTS['V']+COUNTS['novel'],
        F_forward_rows=2,synthetic_P0_forward_rows=2,G_forward_rows=0,real_IQ_values_read=0,source_L_metadata_labels_read=COUNTS['L'],
        neural_F_updates=0,G_updates=0,source_U_reads=0,target_reads=0,source_novel_truth_reads=0,source_only=True,classification_path='P1',default_head_replaced=False,
        model_parameters_buffers_unchanged=True,metric_parameters_buffers_frozen_for_V_novel=True,aggregate_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_free_deployment_state=True,per_record_source_rows_retained=0,parent_public_features_reused=True,source_L_TX_RX_labels_used=True,source_OE_TX_or_K_used=False,
        cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],g_evaluation_seconds=g_known+g_novel,
        fresh_closed_vs_cached_baseline_changed_count=0,fresh_closed_correct_count=int(np.count_nonzero(known['closed']==known['y'])),
        cached_baseline_correct_count=int(np.count_nonzero(known['baseline_closed']==known['y'])),A_N=None,H_OW=None,final_phase3_complete=False,seconds=time.perf_counter()-started)
    _write(out/'complete.json',result);print(STATUS,arm,flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_METRIC_EMBEDDING_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv);run(_read(args.config),args.device);return 0


if __name__=='__main__':raise SystemExit(main())
