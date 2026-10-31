"""Frozen CPAIR controls and one independently trained native UNIFORM student."""
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
from . import ground_centered_run as previous
from . import ground_open_core as calibration
from . import ground_open_data as data
from . import star_dura_best_run as utilities

RUN_ID='20261101-phase3-ground-uniform-s2026092701-r01'
REFERENCE_CODE='1ae125733a0f25eb9c887351eb63f11f730d8cea'
SCHEMA='pamf_phase3_ground_uniform_prediction_v1'
STATUS='GROUND_UNIFORM_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='GROUND_UNIFORM_THRESHOLDS_FROZEN'
METHODS=('OLD_CPAIR_energy','OLD_CPAIR_centered','UNIFORM_energy','UNIFORM_centered')
PRIMARY_METHOD=METHODS[3]
ARMS=('UNIFORM',)
CONDITIONS=previous.CONDITIONS
SOURCE_Q_COUNT,TARGET_Q_COUNT,CAL_COUNT=16650,10238,13500
COUNTS=dict(CAL=CAL_COUNT,source_Q=SOURCE_Q_COUNT,cross_Q=TARGET_Q_COUNT)
FIELDS={'run_id','code_commit','reference_config','train_config','output_root','batch_size'}
ROW_FIELDS=previous.ROW_FIELDS
SOURCE_FIELDS=ROW_FIELDS|{'rx_i','day_i'}
_read,_write,_save=utilities._read,utilities._write,utilities._save
_joint,_centered=previous._joint,previous._centered


def _train():
    from . import ground_uniform_train
    return ground_uniform_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field source-uniform evaluation config required; no private truth/U or threshold options')
    r=previous.validate_config(c['reference_config']);t=_train().validate_config(c['train_config'])
    if r['code_commit']!=REFERENCE_CODE or t['reference_config']!=r or t['code_commit']!=c['code_commit']:
        raise ValueError('Completed unchanged CPAIR reference and independently trained native UNIFORM required')
    tail=r['reference_config'];ground=tail['reference_config'];out=Path(c['output_root']).resolve()
    roots=[r['output_root'],r['train_config']['output_root'],tail['output_root'],tail['train_config']['output_root'],t['output_root'],ground['data_root'],ground['output_root'],ground['train_config']['output_root']]
    roots.extend(q['public_root'] for q in ground['reference_config']['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable uniform predictions overlap an input')
    return c


def _metadata(c):
    r=c['reference_config'];old=_read(Path(r['output_root'])/'complete.json')
    required=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=previous.RUN_ID,code_commit=REFERENCE_CODE,config=r,
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,
        all_thresholds_frozen_before_Q=True,source_controls_reused_exactly=True,paired_readouts_share_logits=True,
        original_F0_baseline_independent_of_method_columns=True,private_truth_reads=0,target_U_reads=0,source_test_labels_read=0,final_phase3_complete=False)
    if any(old.get(k)!=v for k,v in required.items()):raise ValueError('Completed immutable original centered prediction reference required')
    pm,ground,handles,reference=_train()._reference(c['train_config'])
    dataset=_read(Path(r['reference_config']['reference_config']['data_root'])/'complete.json')
    if any(dataset.get(k)!=v for k,v in ground.items()):raise ValueError('Same once-validated D7 source ground data required')
    trained=_read(Path(c['train_config']['output_root'])/'complete.json')
    expected=dict(schema=_train().SCHEMA,status=_train().STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],arms=list(ARMS),
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=2220,real_student_forward_rows=568320,real_F_forward_rows=0,G_forward_rows=0,
        source_L_OE_sampling_matched=True,source_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        source_Us_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_only=True,ground_clean_training=True,
        OE_objective='equal_domain_uniform_posterior',pair_risk_in_objective=False,
        ground_data_identity=ground,source_handles=handles,source_centered_reference_identity=reference,
        source_tail_reference_identity=old['source_tail_reference_identity'],source_ground_reference_identity=old['source_ground_reference_identity'],final_phase3_complete=False)
    if (any(trained.get(k)!=v for k,v in expected.items()) or set(trained.get('identities',{}))!=set(ARMS) or set(trained.get('rows',{}))!=set(ARMS)):
        raise ValueError('The complete fixed-budget source-only UNIFORM student must freeze before CAL')
    for key in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if old[key]!=dataset[key] or trained[key]!=dataset[key]:raise ValueError('Original F0 and DURA ancestry differs')
    for arm in ARMS:
        row=_read(Path(c['train_config']['output_root'])/'rows'/arm/'complete.json');identity=trained['identities'][arm]
        row_expected=dict(schema=_train().SCHEMA,status=_train().ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],arm=arm,identity=identity,updates=2220)
        if row!=trained['rows'][arm] or any(row.get(k)!=v for k,v in row_expected.items()):raise ValueError('Frozen uniform row and global completion disagree')
        fixed=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,updates=2220,reader_parameters=232993,trainable_parameters=232945,
            default_head_parameters=960,frozen_Sinc_parameters=48,Sinc_frozen=True,identity_head_updates=2220,identity_head_jointly_trained=True,
            identity_head_replaced=False,projection_updates=0,source_only=True,ground_clean_training=True,source_Us_reads=0,target_U_reads=0,
            OE_objective='equal_domain_uniform_posterior',pair_risk_in_objective=False,
            source_centered_reference_identity=reference,source_tail_reference_identity=old['source_tail_reference_identity'],
            source_ground_reference_identity=old['source_ground_reference_identity'],ground_data_identity=ground,source_handles=handles)
        if any(identity.get(k)!=v for k,v in fixed.items()) or any(identity[k]!=dataset[k] for k in ('parent_F0_identity','dura_identity','loaded_dura_identity')):
            raise ValueError('Independent complete original-F0 uniform identity and source handles required')
    return old,dataset,trained


_reference=_metadata


def _legacy(c,condition):return previous._rows(c['reference_config']['output_root'],condition)


def _public(c,condition):return previous._public(c['reference_config'],condition)


def _old_calibration(c):
    root=Path(c['reference_config']['output_root']);m=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Original CPAIR source thresholds required')
        tau=z['conditional'].copy()
    if (m.get('schema')!=previous.SCHEMA or m.get('status')!=previous.CALIBRATION_STATUS or m.get('code_commit')!=REFERENCE_CODE
            or m.get('run_id')!=previous.RUN_ID or m.get('method_names')!=list(previous.METHODS) or m.get('calibration_count')!=CAL_COUNT
            or m.get('known_only') is not True or m.get('unknown_inputs_used') is not False or m.get('test_inputs_used') is not False
            or tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or m.get('conditional_thresholds')!=tau.tolist()
            or any(m.get('methods',{}).get(name,{}).get('conditional_thresholds')!=tau[j].tolist() for j,name in ((2,'CPAIR_energy'),(3,'CPAIR_centered')))):
        raise ValueError('Original CPAIR known-only calibration and exact two threshold vectors required')
    return m,tau


def _smoke(b,models,device):
    t=torch.arange(256,device=device,dtype=torch.float32);x=torch.stack((torch.sin(.07*t),torch.cos(.11*t))).repeat(2,1,1)
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        restored=b['dura'][1](x)[0].detach().clone();original=b['reader'].classify_features(b['reader'].features(restored))
    if restored.shape!=(2,2,256) or original.shape!=(2,6) or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (restored,original)):
        raise ValueError('Actual original F0/DURA no-query smoke failed')
    iq=np.asarray(restored.cpu().tolist(),np.float32)
    for model in models.values():
        if _joint(model,iq,device).shape!=(2,6):raise ValueError('Actual uniform native default-head no-query smoke failed')
    return dict(status='PASS',actual_original_F0_uniform_student_DURA=True,synthetic_F_forward_rows=2,synthetic_legacy_forward_rows=0,
        synthetic_student_forward_rows=2,synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0,G_synthetic_forward_rows=2,optimizer_updates=0,real_data_reads=0)


def _stack(legacy,students,tau):
    logits=np.stack((legacy['default_logits'][:,2],legacy['default_logits'][:,3],students['UNIFORM'],students['UNIFORM']),axis=1).astype(np.float32,copy=False)
    closed=logits.argmax(2).astype(np.int64)
    raw=np.column_stack((legacy['raw_scores'][:,2:4],calibration.energy(students['UNIFORM']),_centered(students['UNIFORM'])))
    if any(not np.array_equal(legacy['scores'][:,j+2],raw[:,j]-tau[j,closed[:,j]]) for j in (0,1)):raise ValueError('Original CPAIR energy/centered controls must match their inherited thresholds exactly')
    scores=np.column_stack((legacy['scores'][:,2:4],*[raw[:,j]-tau[j,closed[:,j]] for j in range(2,4)]))
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
    if any(not np.array_equal(logits[:,j],logits[:,j+1]) for j in (0,2)):raise ValueError('Each native model must share logits across its paired readouts')
    raw=np.column_stack([calibration.energy(logits[:,j]) if j%2==0 else _centered(logits[:,j]) for j in range(4)])
    if not np.array_equal(a['raw_scores'],raw):raise ValueError('Every native energy and centered readout must reconstruct')
    if source:
        calibration.masks(a['rx_i'],a['day_i'])
        if len(a['rx_i'])!=n or a['rx_i'].dtype!=np.int64 or a['day_i'].dtype!=np.int64:raise ValueError('Public source RX/day must remain aligned')
    return a


def _rows(root,condition):
    with np.load(Path(root)/'queries'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate_rows(a,condition)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable ground-uniform prediction root required')
    old,dataset,trained=_metadata(c);b=_train()._backend(c['train_config'],device);models={};identities={}
    if b['reader_identity']!=dataset['parent_F0_identity'] or b['dura_identity']!=dataset['loaded_dura_identity']:raise ValueError('Original F0/DURA checkpoint identities differ')
    for arm in ARMS:
        models[arm],identities[arm]=_train().load_trained_model(c['train_config'],arm,device)
        if identities[arm]!=trained['identities'][arm] or sum(p.numel() for p in models[arm].parameters())!=232993:raise ValueError('Exact complete frozen native uniform student required')
    frozen_models=(b['reader'],b['dura'][0],*models.values())
    if any(p.requires_grad or p.dtype!=torch.float32 for model in frozen_models for p in model.parameters()) or any(module.training for model in frozen_models for module in model.modules()):
        raise ValueError('Original F0/DURA and the UNIFORM student must stay frozen FP32 eval')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(model,utilities._snapshot_model(model)) for model in frozen_models]
    def check():
        for model,before in snapshots:utilities._check_model(model,before)
    smoke=_smoke(b,models,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual original F0/DURA/student smoke must pass before CAL')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    old_cal,old_tau=_old_calibration(c);data_root=c['reference_config']['reference_config']['reference_config']['data_root'];cal=data.load_role(data_root,'CAL')
    try:
        if len(cal['ids'])!=CAL_COUNT or cal['manifest']!=old_cal['calibration_manifest']:raise ValueError('Same separate known CAL role required')
        calibration_ids=set(cal['ids'].tolist());tau=np.empty((4,6),np.float64);tau[:2]=old_tau[2:4]
        reports={METHODS[j]:deepcopy(old_cal['methods'][name]) for j,name in enumerate(('CPAIR_energy','CPAIR_centered'))}
        baseline=cal['teacher_logits'].argmax(1).astype(np.int64);meta=cal['metadata']
        for index,arm in enumerate(ARMS):
            logits=_joint(models[arm],cal['iq'],device);closed=logits.argmax(1).astype(np.int64)
            for offset,raw in enumerate((calibration.energy(logits),_centered(logits))):
                j=2+2*index+offset;state,report=calibration.calibrate_known(cal['ids'],meta['y'],closed,baseline,raw,meta['rx_i'],meta['day_i'])
                tau[j]=state['conditional_thresholds'];reports[METHODS[j]]=report
        check()
    finally:data.close(cal)
    (out/'calibration').mkdir();_save(out/'thresholds.npz',dict(conditional=tau))
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        methods=reports,conditional_thresholds=tau.tolist(),calibration_count=CAL_COUNT,calibration_known_count=CAL_COUNT,calibration_manifest=old_cal['calibration_manifest'],
        reference_calibration_role=old_cal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,target_reads=0,neural_models_frozen_before_CAL=True,
        group_names=list(calibration.GROUP_NAMES),source_controls_reused_exactly=True,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,threshold_bytes=192))
    fixed_tau=tau.copy();source_ids=None;target_ids=None
    for condition in CONDITIONS:
        legacy=_legacy(c,condition)
        if condition in CONDITIONS[:2]:
            q=data.load_query(data_root,condition)
            try:
                if (not np.array_equal(q['ids'],legacy['ids']) or not np.array_equal(q['baseline_logits'],legacy['baseline_logits'])
                        or any(not np.array_equal(q['metadata'][k],legacy[k]) for k in ('rx_i','day_i'))):raise ValueError('Same source query and canonical independent original-F0 baseline required')
                if set(q['ids'].tolist())&calibration_ids:raise ValueError('CAL and source queries overlap')
                if source_ids is None:source_ids=q['ids'].copy()
                elif not np.array_equal(source_ids,q['ids']):raise ValueError('Source clean/star physical ID order differs')
                rows=_stack(legacy,{arm:_joint(model,q['iq'],device) for arm,model in models.items()},tau)
            finally:data.close(q)
        else:
            manifest,ids,iq=_public(c,condition)
            try:
                if not np.array_equal(ids,legacy['ids']):raise ValueError('Same canonical future query records required')
                if set(ids.tolist())&(calibration_ids|set(source_ids.tolist())):raise ValueError('Source/CAL and future query records overlap')
                if target_ids is None:target_ids=ids.copy()
                elif not np.array_equal(target_ids,ids):raise ValueError('Cross clean/star physical ID order differs')
                blocks={arm:[] for arm in ARMS}
                for start in range(0,len(ids),128):
                    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                        x=torch.tensor(np.asarray(iq[start:start+128]).tolist(),device=device,dtype=torch.float32);z=b['dura'][1](x)[0].detach().clone()
                    restored=np.asarray(z.cpu().tolist(),np.float32)
                    for arm,model in models.items():blocks[arm].append(_joint(model,restored,device))
                rows=_stack(legacy,{arm:np.concatenate(v) for arm,v in blocks.items()},tau)
            finally:
                if hasattr(iq,'_mmap'):iq._mmap.close()
        _validate_rows(rows,condition);_save(out/'queries'/condition/'predictions.npz',rows);check()
        if not np.array_equal(tau,fixed_tau):raise ValueError('Source-frozen uniform thresholds changed during query inference')
        print('GROUND_UNIFORM_QUERY_COMPLETE',condition,len(rows['ids']),flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        reference_prediction_identity=old,data_capsule_id=dataset['capsule_id'],data_split_id=dataset['split_id'],dataset_identity=dataset,
        model_identities=dict(CPAIR=old['model_identities']['CPAIR'],**identities),source_centered_reference_identity=trained['source_centered_reference_identity'],
        source_tail_reference_identity=trained['source_tail_reference_identity'],source_ground_reference_identity=trained['source_ground_reference_identity'],
        parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],loaded_dura_identity=dataset['loaded_dura_identity'],
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_method_columns=True,paired_readouts_share_logits=True,original_models_unchanged=True,
        calibration_known_count=CAL_COUNT,source_query_count=SOURCE_Q_COUNT,cross_query_count=TARGET_Q_COUNT,
        source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,target_U_reads=0,U_ID_metadata_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,source_Us_reads=0,
        prediction_optimizer_updates=0,source_density_fit_updates=0,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,threshold_bytes=192,
        real_student_forward_rows=CAL_COUNT+2*SOURCE_Q_COUNT+2*TARGET_Q_COUNT,real_legacy_forward_rows=0,real_F_forward_rows=0,G_forward_rows=2*TARGET_Q_COUNT,
        synthetic_student_forward_rows=2,synthetic_legacy_forward_rows=0,synthetic_F_forward_rows=2,synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0,G_synthetic_forward_rows=2,
        trained_optimizer_updates=2220,training_costs={k:v for k,v in trained.items() if isinstance(v,(bool,int,float))},inherited_data_costs={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))},
        source_only=True,ground_clean_training=True,new_target_DA=False,student_heads_jointly_trained=True,OE_objective='equal_domain_uniform_posterior',pair_risk_in_objective=False,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,
        formal_target_accessed=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


run=evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(dict(event='GROUND_UNIFORM_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True);run(_read(a.config),a.device)


if __name__=='__main__':main()
