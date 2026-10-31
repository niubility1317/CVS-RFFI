"""Fixed DMEAN/CPAIR native heads with energy and centered evidence readouts."""
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
from . import ground_tail_run as previous
from . import ground_centered_core as core
from . import ground_open_core as calibration
from . import ground_open_data as data
from . import star_dura_best_run as utilities

RUN_ID='20261101-phase3-ground-centered-s2026092701-r01'
REFERENCE_CODE='315f8ec1be994c40a97cbede4e27942e14267f12'
SCHEMA='pamf_phase3_ground_centered_prediction_v1'
STATUS='GROUND_CENTERED_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='GROUND_CENTERED_THRESHOLDS_FROZEN'
METHODS=('DMEAN_energy','DMEAN_centered','CPAIR_energy','CPAIR_centered')
PRIMARY_METHOD=METHODS[3]
CONDITIONS=previous.CONDITIONS
SOURCE_Q_COUNT,TARGET_Q_COUNT,CAL_COUNT=16650,10238,13500
COUNTS=dict(CAL=CAL_COUNT,source_Q=SOURCE_Q_COUNT,cross_Q=TARGET_Q_COUNT)
FIELDS={'run_id','code_commit','reference_config','train_config','output_root','batch_size'}
ROW_FIELDS=previous.ROW_FIELDS
SOURCE_FIELDS=ROW_FIELDS|{'rx_i','day_i'}
_read,_write,_save=utilities._read,utilities._write,utilities._save
_joint=previous.previous._joint


def _train():
    from . import ground_centered_train
    return ground_centered_train


def _centered(logits):return core.centered_energy(logits)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field centered evaluation config required; no private truth/U or threshold options')
    r=previous.validate_config(c['reference_config']);t=_train().validate_config(c['train_config'])
    if r['code_commit']!=REFERENCE_CODE or t['reference_config']!=r or t['code_commit']!=c['code_commit']:
        raise ValueError('Completed original ground-tail reference and independently trained CPAIR required')
    ground=r['reference_config'];out=Path(c['output_root']).resolve()
    roots=[r['output_root'],r['train_config']['output_root'],t['output_root'],ground['data_root'],ground['output_root'],ground['train_config']['output_root']]
    roots.extend(q['public_root'] for q in ground['reference_config']['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable centered predictions overlap an input')
    return c


def _metadata(c):
    r=c['reference_config'];old=_read(Path(r['output_root'])/'complete.json')
    required=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=previous.RUN_ID,code_commit=REFERENCE_CODE,config=r,
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,
        all_thresholds_frozen_before_Q=True,source_controls_reused_exactly=True,original_F0_baseline_independent_of_method_columns=True,
        private_truth_reads=0,target_U_reads=0,source_test_labels_read=0,final_phase3_complete=False)
    if any(old.get(k)!=v for k,v in required.items()):raise ValueError('Completed immutable original ground-tail predictions required')
    pm,ground,handles,reference=_train()._reference(c['train_config'])
    dataset=_read(Path(r['reference_config']['data_root'])/'complete.json')
    if any(dataset.get(k)!=v for k,v in ground.items()):raise ValueError('Same once-validated D7 ground data ancestor required')
    trained=_read(Path(c['train_config']['output_root'])/'complete.json')
    expected=dict(schema=_train().SCHEMA,status=_train().STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],arms=['CPAIR'],
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=2220,real_student_forward_rows=568320,real_F_forward_rows=0,G_forward_rows=0,
        source_L_OE_sampling_matched=True,source_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        source_Us_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_only=True,ground_clean_training=True,
        ground_data_identity=ground,source_handles=handles,source_tail_reference_identity=reference,
        source_ground_reference_identity=reference['source_ground_reference_identity'],final_phase3_complete=False)
    if (any(trained.get(k)!=v for k,v in expected.items()) or set(trained.get('identities',{}))!={'CPAIR'} or set(trained.get('rows',{}))!={'CPAIR'}):
        raise ValueError('One complete fixed-budget source-only CPAIR student must freeze before CAL')
    row=_read(Path(c['train_config']['output_root'])/'rows/CPAIR/complete.json');identity=trained['identities']['CPAIR']
    row_expected=dict(schema=_train().SCHEMA,status=_train().ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],arm='CPAIR',identity=identity,updates=2220)
    if row!=trained['rows']['CPAIR'] or any(row.get(k)!=v for k,v in row_expected.items()):raise ValueError('The frozen CPAIR row and global completion must agree')
    for key in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if old[key]!=dataset[key] or trained[key]!=dataset[key] or identity[key]!=dataset[key]:raise ValueError('Original F0 and DURA ancestry differs')
    fixed=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm='CPAIR',updates=2220,reader_parameters=232993,trainable_parameters=232945,
        default_head_parameters=960,frozen_Sinc_parameters=48,Sinc_frozen=True,identity_head_updates=2220,identity_head_jointly_trained=True,
        identity_head_replaced=False,projection_updates=0,source_only=True,ground_clean_training=True,source_Us_reads=0,target_U_reads=0,
        source_tail_reference_identity=reference,source_ground_reference_identity=reference['source_ground_reference_identity'],ground_data_identity=ground,source_handles=handles)
    if any(identity.get(k)!=v for k,v in fixed.items()):raise ValueError('Independent complete original-F0 CPAIR identity and legal source handles required')
    return old,dataset,trained


_reference=_metadata


def _legacy(c,condition):return previous._rows(c['reference_config']['output_root'],condition)


def _public(c,condition):return previous._public(c['reference_config'],condition)


def _old_calibration(c):
    root=Path(c['reference_config']['output_root']);m=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Original ground-tail source thresholds required')
        tau=z['conditional'].copy()
    if (m.get('schema')!=previous.SCHEMA or m.get('status')!=previous.CALIBRATION_STATUS or m.get('code_commit')!=REFERENCE_CODE
            or m.get('run_id')!=previous.RUN_ID or m.get('method_names')!=list(previous.METHODS) or m.get('calibration_count')!=CAL_COUNT
            or m.get('known_only') is not True or m.get('unknown_inputs_used') is not False or m.get('test_inputs_used') is not False
            or tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or m.get('conditional_thresholds')!=tau.tolist()
            or m.get('methods',{}).get('DMEAN_energy',{}).get('conditional_thresholds')!=tau[1].tolist()):
        raise ValueError('Original DMEAN known-only calibration and exact thresholds required')
    return m,tau


def _smoke(b,legacy,student,device):
    t=torch.arange(256,device=device,dtype=torch.float32);x=torch.stack((torch.sin(.07*t),torch.cos(.11*t))).repeat(2,1,1)
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        restored=b['dura'][1](x)[0].detach().clone();original=b['reader'].classify_features(b['reader'].features(restored))
    if restored.shape!=(2,2,256) or original.shape!=(2,6) or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (restored,original)):
        raise ValueError('Actual original F0/DURA no-query smoke failed')
    iq=np.asarray(restored.cpu().tolist(),np.float32)
    for model in (legacy,student):
        if _joint(model,iq,device).shape!=(2,6):raise ValueError('Actual DMEAN/CPAIR default-head no-query smoke failed')
    return dict(status='PASS',actual_original_F0_legacy_student_DURA=True,synthetic_F_forward_rows=2,synthetic_legacy_forward_rows=2,
        synthetic_student_forward_rows=2,synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0,G_synthetic_forward_rows=2,optimizer_updates=0,real_data_reads=0)


def _stack(legacy,student,tau):
    old=legacy['default_logits'][:,1];logits=np.stack((old,old,student,student),axis=1).astype(np.float32,copy=False);closed=logits.argmax(2).astype(np.int64)
    raw=np.column_stack((legacy['raw_scores'][:,1],_centered(old),calibration.energy(student),_centered(student)))
    if not np.array_equal(legacy['scores'][:,1],raw[:,0]-tau[0,closed[:,0]]):raise ValueError('Original DMEAN energy control must exactly match its old threshold')
    scores=np.column_stack((legacy['scores'][:,1],*[raw[:,j]-tau[j,closed[:,j]] for j in (1,2,3)]))
    a=dict(ids=legacy['ids'].copy(),default_logits=logits,closed=closed,raw_scores=raw,scores=scores,
        baseline_logits=legacy['baseline_logits'].copy(),baseline_closed=legacy['baseline_closed'].copy())
    if 'rx_i' in legacy:a.update(rx_i=legacy['rx_i'].copy(),day_i=legacy['day_i'].copy())
    return a


def _validate_rows(a,condition):
    source=condition in CONDITIONS[:2];n=SOURCE_Q_COUNT if source else TARGET_Q_COUNT
    if (condition not in CONDITIONS or set(a)!=(SOURCE_FIELDS if source else ROW_FIELDS) or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U'
            or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='') or a['default_logits'].shape!=(n,4,6)
            or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,4) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(2))
            or a['baseline_logits'].shape!=(n,6) or a['baseline_logits'].dtype!=np.float32 or not np.isfinite(a['baseline_logits']).all()
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['baseline_logits'].argmax(1))
            or any(a[k].shape!=(n,4) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Complete four-readout predictions and separate canonical original-F0 baseline required')
    logits=a['default_logits']
    if not np.array_equal(logits[:,0],logits[:,1]) or not np.array_equal(logits[:,2],logits[:,3]):raise ValueError('Each native model must provide the same logits to both readouts')
    raw=np.column_stack((calibration.energy(logits[:,0]),_centered(logits[:,1]),calibration.energy(logits[:,2]),_centered(logits[:,3])))
    if not np.array_equal(a['raw_scores'],raw):raise ValueError('Every native energy and centered evidence must reconstruct')
    if source:
        calibration.masks(a['rx_i'],a['day_i'])
        if len(a['rx_i'])!=n or a['rx_i'].dtype!=np.int64 or a['day_i'].dtype!=np.int64:raise ValueError('Source public RX/day must remain aligned')
    return a


def _rows(root,condition):
    with np.load(Path(root)/'queries'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate_rows(a,condition)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable ground-centered prediction root required')
    old,dataset,trained=_metadata(c);b=_train()._backend(c['train_config'],device)
    student,identity=_train().load_trained_model(c['train_config'],'CPAIR',device)
    legacy,legacy_identity=previous._train().load_trained_model(c['reference_config']['train_config'],'DMEAN',device)
    if (identity!=trained['identities']['CPAIR'] or legacy_identity!=old['model_identities']['DMEAN']
            or b['reader_identity']!=dataset['parent_F0_identity'] or b['dura_identity']!=dataset['loaded_dura_identity']):
        raise ValueError('Actual CPAIR/DMEAN/original-F0/DURA checkpoint identities differ')
    for model in (legacy,student):
        if sum(p.numel() for p in model.parameters())!=232993:raise ValueError('Complete frozen native default head and backbone required')
    models=(b['reader'],b['dura'][0],legacy,student)
    if any(p.requires_grad or p.dtype!=torch.float32 for model in models for p in model.parameters()) or any(module.training for model in models for module in model.modules()):
        raise ValueError('Original F0/DURA and both native students must stay frozen FP32 eval')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(model,utilities._snapshot_model(model)) for model in models]
    def check():
        for model,before in snapshots:utilities._check_model(model,before)
    smoke=_smoke(b,legacy,student,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual four-model no-query smoke must pass before CAL')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    old_cal,old_tau=_old_calibration(c);data_root=c['reference_config']['reference_config']['data_root'];cal=data.load_role(data_root,'CAL')
    try:
        if len(cal['ids'])!=CAL_COUNT or cal['manifest']!=old_cal['calibration_manifest']:raise ValueError('Same separate known CAL role required')
        calibration_ids=set(cal['ids'].tolist());tau=np.empty((4,6),np.float64);tau[0]=old_tau[1]
        reports={METHODS[0]:deepcopy(old_cal['methods']['DMEAN_energy'])};baseline=cal['teacher_logits'].argmax(1).astype(np.int64);meta=cal['metadata']
        old_logits=_joint(legacy,cal['iq'],device);new_logits=_joint(student,cal['iq'],device)
        for j,logits,raw in ((1,old_logits,_centered(old_logits)),(2,new_logits,calibration.energy(new_logits)),(3,new_logits,_centered(new_logits))):
            state,report=calibration.calibrate_known(cal['ids'],meta['y'],logits.argmax(1).astype(np.int64),baseline,raw,meta['rx_i'],meta['day_i'])
            tau[j]=state['conditional_thresholds'];reports[METHODS[j]]=report
        check()
    finally:data.close(cal)
    (out/'calibration').mkdir()
    _save(out/'thresholds.npz',dict(conditional=tau));_write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,methods=reports,conditional_thresholds=tau.tolist(),calibration_count=CAL_COUNT,calibration_known_count=CAL_COUNT,
        calibration_manifest=old_cal['calibration_manifest'],reference_calibration_role=old_cal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,
        target_reads=0,neural_models_frozen_before_CAL=True,group_names=list(calibration.GROUP_NAMES),source_controls_reused_exactly=True,new_threshold_values=18,reused_threshold_values=6,threshold_values=24,threshold_bytes=192))
    frozen=tau.copy();source_ids=None;target_ids=None
    for condition in CONDITIONS:
        legacy_rows=_legacy(c,condition)
        if condition in CONDITIONS[:2]:
            q=data.load_query(data_root,condition)
            try:
                if (not np.array_equal(q['ids'],legacy_rows['ids']) or not np.array_equal(q['baseline_logits'],legacy_rows['baseline_logits'])
                        or any(not np.array_equal(q['metadata'][k],legacy_rows[k]) for k in ('rx_i','day_i'))):raise ValueError('Same source query and canonical original-F0 baseline required; SSL column is not the baseline')
                if set(q['ids'].tolist())&calibration_ids:raise ValueError('CAL and source query must remain physically disjoint')
                if source_ids is None:source_ids=q['ids'].copy()
                elif not np.array_equal(source_ids,q['ids']):raise ValueError('Source clean/star physical ID order differs')
                rows=_stack(legacy_rows,_joint(student,q['iq'],device),tau)
            finally:data.close(q)
        else:
            manifest,ids,iq=_public(c,condition)
            try:
                if not np.array_equal(ids,legacy_rows['ids']):raise ValueError('Same canonical future query records required')
                if set(ids.tolist())&(calibration_ids|set(source_ids.tolist())):raise ValueError('Source/CAL and cross-receiver query records overlap')
                if target_ids is None:target_ids=ids.copy()
                elif not np.array_equal(target_ids,ids):raise ValueError('Cross clean/star physical ID order differs')
                blocks=[]
                for start in range(0,len(ids),128):
                    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                        x=torch.tensor(np.asarray(iq[start:start+128]).tolist(),device=device,dtype=torch.float32);z=b['dura'][1](x)[0].detach().clone()
                    blocks.append(_joint(student,np.asarray(z.cpu().tolist(),np.float32),device))
                rows=_stack(legacy_rows,np.concatenate(blocks),tau)
            finally:
                if hasattr(iq,'_mmap'):iq._mmap.close()
        _validate_rows(rows,condition);_save(out/'queries'/condition/'predictions.npz',rows);check()
        if not np.array_equal(tau,frozen):raise ValueError('Source-frozen centered thresholds changed during queries')
        print('GROUND_CENTERED_QUERY_COMPLETE',condition,len(rows['ids']),flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        reference_prediction_identity=old,data_capsule_id=dataset['capsule_id'],data_split_id=dataset['split_id'],dataset_identity=dataset,
        model_identities=dict(DMEAN=legacy_identity,CPAIR=identity),source_tail_reference_identity=trained['source_tail_reference_identity'],source_ground_reference_identity=trained['source_ground_reference_identity'],
        parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],loaded_dura_identity=dataset['loaded_dura_identity'],
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_method_columns=True,paired_readouts_share_logits=True,original_models_unchanged=True,
        calibration_known_count=CAL_COUNT,source_query_count=SOURCE_Q_COUNT,cross_query_count=TARGET_Q_COUNT,
        source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,target_U_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,source_Us_reads=0,
        prediction_optimizer_updates=0,source_density_fit_updates=0,new_threshold_values=18,reused_threshold_values=6,threshold_values=24,threshold_bytes=192,
        real_student_forward_rows=CAL_COUNT+2*SOURCE_Q_COUNT+2*TARGET_Q_COUNT,real_legacy_forward_rows=CAL_COUNT,real_F_forward_rows=0,G_forward_rows=2*TARGET_Q_COUNT,
        synthetic_student_forward_rows=smoke['synthetic_student_forward_rows'],synthetic_legacy_forward_rows=smoke['synthetic_legacy_forward_rows'],synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],
        synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0,G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],trained_optimizer_updates=2220,
        training_costs={k:v for k,v in trained.items() if isinstance(v,(bool,int,float))},inherited_data_costs={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))},
        source_only=True,ground_clean_training=True,new_target_DA=False,student_heads_jointly_trained=True,primary_deployed_neural_parameters=499944,primary_threshold_values=6,
        formal_target_accessed=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


run=evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(dict(event='GROUND_CENTERED_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True);run(_read(a.config),a.device)


if __name__=='__main__':main()
