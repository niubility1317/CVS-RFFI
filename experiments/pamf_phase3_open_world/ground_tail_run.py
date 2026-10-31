"""Two fixed ground-tail students; legacy SSL comparison and four DURA queries."""
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
from . import ground_open_run as previous
from . import ground_open_core as calibration
from . import ground_open_data as data
from . import star_dura_best_run as utilities

RUN_ID='20261101-phase3-ground-tail-s2026092701-r01'
REFERENCE_CODE='d7e0f875397b2f9801df4605ba0051d221aa4e2a'
SCHEMA='pamf_phase3_ground_tail_prediction_v1'
STATUS='GROUND_TAIL_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='GROUND_TAIL_THRESHOLDS_FROZEN'
METHODS=('OLD_SSL_energy','DMEAN_energy','DTAIL_energy')
PRIMARY_METHOD='DTAIL_energy'
CONDITIONS=previous.CONDITIONS
SOURCE_Q_COUNT,TARGET_Q_COUNT,CAL_COUNT=16650,10238,13500
FIELDS={'run_id','code_commit','reference_config','train_config','output_root','batch_size'}
ROW_FIELDS=previous.ROW_FIELDS|{'baseline_logits'}
SOURCE_FIELDS=ROW_FIELDS|{'rx_i','day_i'}
_read,_write,_save=utilities._read,utilities._write,utilities._save


def _train():
    from . import ground_tail_train
    return ground_tail_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source-tail evaluation config required; no private truth/U/threshold options')
    r=previous.validate_config(c['reference_config']);t=_train().validate_config(c['train_config'])
    if r['code_commit']!=REFERENCE_CODE or t['reference_config']!=r or t['code_commit']!=c['code_commit']:
        raise ValueError('Exact old ground reference and new matched native students required')
    out=Path(c['output_root']).resolve();roots=[r['output_root'],r['data_root'],r['train_config']['output_root'],t['output_root'],r['reference_config']['output_root']]
    roots.extend(q['public_root'] for q in r['reference_config']['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable tail predictions overlap an input')
    return c


def _metadata(c):
    r=c['reference_config'];old=_read(Path(r['output_root'])/'complete.json');dataset,_=previous._metadata(r)
    required=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=previous.RUN_ID,code_commit=REFERENCE_CODE,config=r,
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,
        all_thresholds_frozen_before_Q=True,private_truth_reads=0,target_U_reads=0,source_test_labels_read=0)
    if any(old.get(k)!=v for k,v in required.items()):raise ValueError('Completed immutable old ground reference required')
    t=_read(Path(c['train_config']['output_root'])/'complete.json')
    expected=dict(schema=_train().SCHEMA,status=_train().STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1136640,
        source_L_OE_sampling_matched=True,source_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        source_Us_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,final_phase3_complete=False)
    if any(t.get(k)!=v for k,v in expected.items()) or set(t.get('identities',{}))!={'DMEAN','DTAIL'}:raise ValueError('Both complete matched source-tail native students required')
    for k in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if old[k]!=dataset[k] or t[k]!=dataset[k]:raise ValueError('Old G(clean) data, new students and original F0/DURA ancestry must match')
    return old,dataset,t


_reference=_metadata


def _legacy(c,condition):return previous._rows(c['reference_config']['output_root'],condition)


def _public(c,condition):return previous._public(c['reference_config'],condition)


def _old_calibration(c):
    root=Path(c['reference_config']['output_root']);m=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Original fixed source thresholds required')
        tau=z['conditional'].copy()
    if (m.get('status')!=previous.CALIBRATION_STATUS or m.get('code_commit')!=REFERENCE_CODE or m.get('run_id')!=previous.RUN_ID
            or m.get('method_names')!=list(previous.METHODS) or m.get('calibration_count')!=CAL_COUNT or m.get('known_only') is not True
            or m.get('unknown_inputs_used') is not False or m.get('test_inputs_used') is not False or tau.shape!=(3,6)
            or tau.dtype!=np.float64 or not np.isfinite(tau).all() or m.get('conditional_thresholds')!=tau.tolist()):
        raise ValueError('Old known-only source calibration/SSL threshold ancestor differs')
    return m,tau


def _stack(legacy,students,tau):
    logits=np.stack((legacy['default_logits'][:,2],*students),axis=1).astype(np.float32,copy=False);closed=logits.argmax(2).astype(np.int64)
    raw=np.column_stack((legacy['raw_scores'][:,2],*[calibration.energy(logits[:,j]) for j in (1,2)]))
    scores=np.column_stack((legacy['scores'][:,2],*[raw[:,j]-tau[j,closed[:,j]] for j in (1,2)]))
    a=dict(ids=legacy['ids'].copy(),default_logits=logits,closed=closed,raw_scores=raw,scores=scores,
        baseline_logits=legacy['default_logits'][:,0].copy(),baseline_closed=legacy['baseline_closed'].copy())
    if 'rx_i' in legacy:a.update(rx_i=legacy['rx_i'].copy(),day_i=legacy['day_i'].copy())
    return a


def _validate_rows(a,condition):
    source=condition in CONDITIONS[:2];n=SOURCE_Q_COUNT if source else TARGET_Q_COUNT
    if (condition not in CONDITIONS or set(a)!=(SOURCE_FIELDS if source else ROW_FIELDS) or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U'
            or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='') or a['default_logits'].shape!=(n,3,6)
            or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,3) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(2))
            or a['baseline_logits'].shape!=(n,6) or a['baseline_logits'].dtype!=np.float32 or not np.isfinite(a['baseline_logits']).all()
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['baseline_logits'].argmax(1))
            or any(a[k].shape!=(n,3) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Complete tail/legacy predictions and independent original-F0 baseline required')
    if source:
        calibration.masks(a['rx_i'],a['day_i'])
        if len(a['rx_i'])!=n or a['rx_i'].dtype!=np.int64 or a['day_i'].dtype!=np.int64:raise ValueError('Public source metadata must remain aligned')
    return a


def _rows(root,condition):
    with np.load(Path(root)/'queries'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate_rows(a,condition)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source-tail prediction root required')
    old,dataset,trained=_metadata(c);b=_train()._backend(c['train_config'],device);models={};identities={}
    if b['reader_identity']!=dataset['parent_F0_identity'] or b['dura_identity']!=dataset['loaded_dura_identity']:raise ValueError('Original F0 and same fixed G source data required')
    for arm in ('DMEAN','DTAIL'):
        models[arm],identities[arm]=_train().load_trained_model(c['train_config'],arm,device)
        if identities[arm]!=trained['identities'][arm] or sum(p.numel() for p in models[arm].parameters())!=232993 or any(p.requires_grad or p.dtype!=torch.float32 for p in models[arm].parameters()) or any(m.training for m in models[arm].modules()):raise ValueError('Complete frozen native tail students required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,utilities._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0],*models.values())]
    def check():
        for model,before in snapshots:utilities._check_model(model,before)
    smoke=previous._smoke(b,models,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    old_cal,old_tau=_old_calibration(c);cal=data.load_role(c['reference_config']['data_root'],'CAL')
    try:
        if len(cal['ids'])!=CAL_COUNT or cal['manifest']!=old_cal['calibration_manifest']:raise ValueError('Same separate known-only source calibration role required')
        tau=np.empty((3,6),np.float64);tau[0]=old_tau[2];reports={METHODS[0]:deepcopy(old_cal['methods']['SSL_energy'])}
        baseline=cal['teacher_logits'].argmax(1).astype(np.int64);meta=cal['metadata']
        for j,arm in enumerate(('DMEAN','DTAIL'),1):
            logits=previous._joint(models[arm],cal['iq'],device)
            state,report=calibration.calibrate_known(cal['ids'],meta['y'],logits.argmax(1).astype(np.int64),baseline,calibration.energy(logits),meta['rx_i'],meta['day_i'])
            tau[j]=state['conditional_thresholds'];reports[METHODS[j]]=report
        check()
    finally:data.close(cal)
    (out/'calibration').mkdir();_save(out/'thresholds.npz',dict(conditional=tau))
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        methods=reports,conditional_thresholds=tau.tolist(),calibration_count=CAL_COUNT,calibration_known_count=CAL_COUNT,calibration_manifest=old_cal['calibration_manifest'],
        reference_calibration_role=old_cal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,target_reads=0,neural_models_frozen_before_CAL=True,
        group_names=list(calibration.GROUP_NAMES),source_controls_reused_exactly=True,new_threshold_values=12,reused_threshold_values=6,threshold_values=18))
    frozen=tau.copy();source_ids=None;target_ids=None
    for condition in CONDITIONS:
        legacy=_legacy(c,condition)
        if condition in CONDITIONS[:2]:
            q=data.load_query(c['reference_config']['data_root'],condition)
            try:
                if not np.array_equal(q['ids'],legacy['ids']) or not np.array_equal(q['baseline_logits'],legacy['default_logits'][:,0]) or any(not np.array_equal(q['metadata'][k],legacy[k]) for k in ('rx_i','day_i')):raise ValueError('Identical old source clean/star records and original F0 baseline required')
                if source_ids is None:source_ids=q['ids'].copy()
                elif not np.array_equal(source_ids,q['ids']):raise ValueError('Source clean/star physical ID order must match')
                rows=_stack(legacy,[previous._joint(models[a],q['iq'],device) for a in ('DMEAN','DTAIL')],tau)
            finally:data.close(q)
        else:
            manifest,ids,iq=_public(c,condition)
            try:
                if not np.array_equal(ids,legacy['ids']):raise ValueError('Canonical future query and old original-F0 records differ')
                if target_ids is None:target_ids=ids.copy()
                elif not np.array_equal(target_ids,ids):raise ValueError('Cross clean/star future physical records must match')
                blocks={a:[] for a in ('DMEAN','DTAIL')}
                for start in range(0,len(ids),128):
                    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                        x=torch.tensor(np.asarray(iq[start:start+128]).tolist(),device=device,dtype=torch.float32);z=b['dura'][1](x)[0].detach().clone()
                    restored=np.asarray(z.cpu().tolist(),np.float32)
                    for a in blocks:blocks[a].append(previous._joint(models[a],restored,device))
                rows=_stack(legacy,[np.concatenate(blocks[a]) for a in ('DMEAN','DTAIL')],tau)
            finally:
                if hasattr(iq,'_mmap'):iq._mmap.close()
        _validate_rows(rows,condition);_save(out/'queries'/condition/'predictions.npz',rows);check()
        if not np.array_equal(tau,frozen):raise ValueError('Source-tail thresholds changed during queries')
        print('GROUND_TAIL_QUERY_COMPLETE',condition,len(rows['ids']),flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        data_capsule_id=dataset['capsule_id'],data_split_id=dataset['split_id'],dataset_identity=dataset,model_identities=identities,
        source_ground_reference_identity=trained['source_ground_reference_identity'],parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],loaded_dura_identity=dataset['loaded_dura_identity'],
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_method_columns=True,calibration_known_count=CAL_COUNT,source_query_count=SOURCE_Q_COUNT,cross_query_count=TARGET_Q_COUNT,
        source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,target_U_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,source_Us_reads=0,prediction_optimizer_updates=0,source_density_fit_updates=0,
        new_threshold_values=12,reused_threshold_values=6,threshold_values=18,threshold_bytes=144,real_student_forward_rows=2*CAL_COUNT+4*SOURCE_Q_COUNT+4*TARGET_Q_COUNT,
        real_F_forward_rows=0,G_forward_rows=2*TARGET_Q_COUNT,synthetic_student_forward_rows=smoke['synthetic_student_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        trained_optimizer_updates=4440,training_costs={k:v for k,v in trained.items() if isinstance(v,(bool,int,float))},inherited_data_costs={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))},
        source_only=True,ground_clean_training=True,new_target_DA=False,original_models_unchanged=True,student_heads_jointly_trained=True,primary_deployed_neural_parameters=499944,primary_threshold_values=6,
        formal_target_accessed=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


run=evaluate
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(dict(event='GROUND_TAIL_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True);run(_read(a.config),a.device)
if __name__=='__main__':main()
