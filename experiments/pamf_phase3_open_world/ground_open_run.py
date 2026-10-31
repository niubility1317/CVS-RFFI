"""Freeze ground students, calibrate on CAL only, then predict four conditions."""
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
from . import ground_open_core as core
from . import ground_open_data as data
from . import fresh_world_run as fresh
from . import star_dura_best_run as utilities

RUN_ID='20261101-phase3-ground-open-s2026092701-r02'
SCHEMA='pamf_phase3_ground_open_prediction_v1'
STATUS='GROUND_OPEN_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='GROUND_OPEN_THRESHOLDS_FROZEN'
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
METHODS=('F0_energy','SUP_energy','SSL_energy')
PRIMARY_METHOD='SSL_energy'
CONDITIONS=('source_clean_DURA','source_star_DURA','cross_clean_DURA','cross_star_DURA')
SOURCE_Q_COUNT=16650
TARGET_Q_COUNT=10238
CAL_COUNT=13500
FIELDS={'run_id','code_commit','reference_config','data_root','train_config','output_root','batch_size'}
ROW_FIELDS={'ids','default_logits','closed','baseline_closed','raw_scores','scores'}
SOURCE_FIELDS=ROW_FIELDS|{'rx_i','day_i'}
_read,_write,_same=utilities._read,utilities._write,utilities._same
_save=utilities._save


def _train():
    from . import ground_open_train
    return ground_open_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('data_root','output_root'))):
        raise ValueError('Exact ground evaluation config required; no private test labels or threshold options')
    t=_train().validate_config(c['train_config']);r=fresh.validate_config(c['reference_config'])
    if (r['code_commit']!=REFERENCE_CODE or t['reference_config']!=r or t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or t['data_root']!=c['data_root']):
        raise ValueError('Ground training/source data and original frozen F0 reference must match')
    out=Path(c['output_root']).resolve();roots=[c['data_root'],t['output_root'],r['output_root'],r['train_config']['output_root']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable ground predictions overlap an input')
    return c


def _metadata(c):
    d=_read(Path(c['data_root'])/'complete.json');t=_read(Path(c['train_config']['output_root'])/'complete.json')
    expected=dict(schema=data.SCHEMA,status=data.STATUS,run_id=RUN_ID,code_commit=c['code_commit'],phase3_data_status='VALIDATED_ONCE')
    if any(d.get(k)!=v for k,v in expected.items()):raise ValueError('Completed once-validated ground source capsule required')
    required=dict(schema=_train().SCHEMA,status=_train().STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1420800,
        source_L_OE_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_U_GT_reads=0,final_phase3_complete=False)
    if any(t.get(k)!=v for k,v in required.items()):raise ValueError('Both complete fixed-budget ground training arms required')
    if set(t.get('identities',{}))!={'SUP','SSL'}:raise ValueError('Both complete native ground identities required')
    return d,t


def _joint(model,values,device):
    parts=[]
    for start in range(0,len(values),128):
        a=np.asarray(values[start:start+128])
        if a.dtype!=np.float32 or a.shape!=(len(a),2,256) or not np.isfinite(a).all():raise ValueError('Finite FP32 full256 same-DURA IQ required')
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            _,logits=model(torch.tensor(a.tolist(),device=device,dtype=torch.float32))
        l=np.asarray(logits.detach().cpu().tolist(),np.float32)
        if l.shape!=(len(a),6) or not np.isfinite(l).all():raise ValueError('Complete six-class native default logits required')
        parts.append(l)
    return np.concatenate(parts)


def _stack(ids,baseline,students,tau,metadata=None):
    logits=np.stack((baseline,*students),axis=1).astype(np.float32,copy=False);closed=logits.argmax(2).astype(np.int64)
    raw=np.column_stack([core.energy(logits[:,j]) for j in range(3)])
    margins=np.column_stack([raw[:,j]-tau[j,closed[:,j]] for j in range(3)])
    row=dict(ids=np.asarray(ids).copy(),default_logits=logits,closed=closed,baseline_closed=closed[:,0].copy(),raw_scores=raw,scores=margins)
    if metadata is not None:row.update(rx_i=metadata['rx_i'].copy(),day_i=metadata['day_i'].copy())
    return row


def _validate_rows(a,condition):
    source=condition in CONDITIONS[:2];n=SOURCE_Q_COUNT if source else TARGET_Q_COUNT
    fields=SOURCE_FIELDS if source else ROW_FIELDS
    if (condition not in CONDITIONS or set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n
            or np.any(np.char.strip(a['ids'])=='') or a['default_logits'].shape!=(n,3,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,3) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(2))
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'][:,0])
            or any(a[k].shape!=(n,3) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Complete role-blind three-method predictions with original same-DURA F0 baseline required')
    if source:
        core.masks(a['rx_i'],a['day_i'])
        if len(a['rx_i'])!=n:raise ValueError('Public source query metadata must align')
    return a


def _rows(root,condition):
    with np.load(Path(root)/'queries'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate_rows(a,condition)


def _public(c,condition):return fresh._public(c['reference_config'],'clean_DURA' if condition=='cross_clean_DURA' else 'star_DURA')
def _old_target(c,condition):return fresh._target_rows(c['reference_config']['output_root'],'clean_DURA' if condition=='cross_clean_DURA' else 'star_DURA')


def _smoke(b,models,device):
    t=torch.arange(256,device=device,dtype=torch.float32);x=torch.stack((torch.sin(.07*t),torch.cos(.11*t))).repeat(2,1,1)
    with torch.no_grad():restored=b['dura'][1](x)[0].detach().clone()
    for model in models.values():
        if _joint(model,np.asarray(restored.cpu().tolist(),np.float32),device).shape!=(2,6):raise ValueError('Actual ground student/G no-query smoke required')
    return dict(status='PASS',actual_ground_students_DURA=True,synthetic_student_forward_rows=4,G_synthetic_forward_rows=2,optimizer_updates=0,real_data_reads=0)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable ground prediction root required')
    dataset,trained=_metadata(c);b=_train()._backend(c['train_config'],device);models={};identities={}
    if b['reader_identity']!=dataset['parent_F0_identity'] or b['dura_identity']!=dataset['loaded_dura_identity']:raise ValueError('Same original F0 and frozen DURA dataset identities required')
    for arm in ('SUP','SSL'):
        models[arm],identities[arm]=_train().load_trained_model(c['train_config'],arm,device)
        if identities[arm]!=trained['identities'][arm] or sum(p.numel() for p in models[arm].parameters())!=232993 or any(p.requires_grad or p.dtype!=torch.float32 for p in models[arm].parameters()) or any(m.training for m in models[arm].modules()):
            raise ValueError('Exact complete frozen native ground checkpoint required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,utilities._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0],*models.values())]
    def check():
        for model,before in snapshots:utilities._check_model(model,before)
    smoke=_smoke(b,models,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    cal=data.load_role(c['data_root'],'CAL')
    try:
        if len(cal['ids'])!=CAL_COUNT:raise ValueError('Exact separate known calibration count required')
        logits=[cal['teacher_logits'],*[_joint(models[arm],cal['iq'],device) for arm in ('SUP','SSL')]]
        if any(l.shape!=(CAL_COUNT,6) or l.dtype!=np.float32 or not np.isfinite(l).all() for l in logits):raise ValueError('Known-only source calibration logits required')
        tau=np.empty((3,6),np.float64);reports={};meta=cal['metadata'];baseline=logits[0].argmax(1).astype(np.int64)
        for j,name in enumerate(METHODS):
            state,report=core.calibrate_known(cal['ids'],meta['y'],logits[j].argmax(1).astype(np.int64),baseline,core.energy(logits[j]),meta['rx_i'],meta['day_i'])
            tau[j]=state['conditional_thresholds'];reports[name]=report
        cal_manifest=deepcopy(cal['manifest']);check()
    finally:data.close(cal)
    (out/'calibration').mkdir()
    _save(out/'thresholds.npz',dict(conditional=tau));_write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        methods=reports,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditional_thresholds=tau.tolist(),calibration_count=CAL_COUNT,calibration_manifest=cal_manifest,
        known_only=True,unknown_inputs_used=False,test_inputs_used=False,target_reads=0,neural_models_frozen_before_CAL=True,group_names=list(core.GROUP_NAMES)))
    frozen_tau=tau.copy();source_ids=None;target_ids=None
    for condition in CONDITIONS:
        if condition in CONDITIONS[:2]:
            query=data.load_query(c['data_root'],condition)
            try:
                if source_ids is None:source_ids=query['ids'].copy()
                elif not np.array_equal(source_ids,query['ids']):raise ValueError('Source clean/star must share the same ordered physical records')
                students=[_joint(models[arm],query['iq'],device) for arm in ('SUP','SSL')]
                rows=_stack(query['ids'],query['baseline_logits'],students,tau,query['metadata'])
            finally:data.close(query)
        else:
            manifest,ids,iq=_public(c,condition)
            try:
                old=_old_target(c,condition)
                if not np.array_equal(ids,old['ids']):raise ValueError('Original canonical future records must match the frozen original-F0 baseline')
                if target_ids is None:target_ids=ids.copy()
                elif not np.array_equal(target_ids,ids):raise ValueError('Cross-RX clean/star future records must match')
                blocks={arm:[] for arm in ('SUP','SSL')}
                for start in range(0,len(ids),128):
                    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                        x=torch.tensor(np.asarray(iq[start:start+128]).tolist(),device=device,dtype=torch.float32);z=b['dura'][1](x)[0].detach().clone()
                    restored=np.asarray(z.cpu().tolist(),np.float32)
                    for arm in blocks:blocks[arm].append(_joint(models[arm],restored,device))
                rows=_stack(ids,old['default_logits'],[np.concatenate(blocks[arm]) for arm in ('SUP','SSL')],tau)
            finally:
                if hasattr(iq,'_mmap'):iq._mmap.close()
        _validate_rows(rows,condition);_save(out/'queries'/condition/'predictions.npz',rows);check()
        if not np.array_equal(tau,frozen_tau):raise ValueError('Ground source thresholds changed during query inference')
        print('GROUND_OPEN_QUERY_COMPLETE',condition,len(rows['ids']),flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        data_capsule_id=dataset['capsule_id'],data_split_id=dataset['split_id'],dataset_identity=dataset,model_identities=identities,
        parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],loaded_dura_identity=dataset['loaded_dura_identity'],
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        calibration_known_count=CAL_COUNT,source_query_count=SOURCE_Q_COUNT,cross_query_count=TARGET_Q_COUNT,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,
        target_U_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,source_Us_values_read_during_prediction=0,prediction_optimizer_updates=0,source_density_fit_updates=0,
        new_threshold_values=18,threshold_bytes=144,real_student_forward_rows=2*CAL_COUNT+4*SOURCE_Q_COUNT+4*TARGET_Q_COUNT,
        real_F_forward_rows=0,G_forward_rows=2*TARGET_Q_COUNT,synthetic_student_forward_rows=4,G_synthetic_forward_rows=2,
        trained_optimizer_updates=4440,training_costs={k:v for k,v in trained.items() if isinstance(v,(bool,int,float))},data_costs={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))},
        source_only=True,ground_clean_training=True,new_target_DA=False,original_models_unchanged=True,student_heads_jointly_trained=True,primary_deployed_neural_parameters=499944,primary_threshold_values=6,
        formal_target_accessed=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result

run=evaluate
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(dict(event='GROUND_OPEN_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True);run(_read(a.config),a.device)
if __name__=='__main__':main()
