"""Source-Us-only geometric rejection with one frozen original F0 reader."""
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
from torch.nn import functional as F
from . import ground_geometry_core as core
from . import ground_open_run as previous
from . import ground_open_core as risk
from . import ground_open_data as data
from . import star_dura_best_run as utilities

RUN_ID='20261101-phase3-ground-geometry-s2026092701-r01'
REFERENCE_CODE='d7e0f875397b2f9801df4605ba0051d221aa4e2a'
SCHEMA='pamf_phase3_ground_geometry_prediction_v1'
STATUS='GROUND_GEOMETRY_PREDICTIONS_COMPLETE'
MODELS_STATUS='GROUND_GEOMETRY_SUBSPACES_FROZEN'
CALIBRATION_STATUS='GROUND_GEOMETRY_THRESHOLDS_FROZEN'
METHODS=('F0_energy','OLD_SSL_energy','F0_vim','F0_domain_vim')
PRIMARY_METHOD=METHODS[3]
CONDITIONS=previous.CONDITIONS
MODES=core.MODES
SOURCE_Q_COUNT,TARGET_Q_COUNT,CAL_COUNT,US_COUNT=16650,10238,13500,56700
FIELDS={'run_id','code_commit','reference_config','output_root','batch_size'}
ROW_FIELDS=previous.ROW_FIELDS|{'baseline_logits','fresh_F0_logits','classifier_coordinates','residual_norms'}
SOURCE_FIELDS=ROW_FIELDS|{'rx_i','day_i'}
_read,_write,_save=utilities._read,utilities._write,utilities._save


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact geometry configuration required; no labels/U/rank/threshold knobs')
    r=previous.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Only the unchanged completed ground r03/D7 reference is supported')
    out=Path(c['output_root']).resolve();roots=[r['output_root'],r['data_root'],r['train_config']['output_root'],r['reference_config']['output_root']]
    roots.extend(q['public_root'] for q in r['reference_config']['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New immutable geometry output overlaps an input')
    return c


def _metadata(c):
    r=c['reference_config'];old=_read(Path(r['output_root'])/'complete.json');dataset,_=previous._metadata(r)
    expected=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=previous.RUN_ID,code_commit=REFERENCE_CODE,config=r,
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,private_truth_reads=0,target_U_reads=0)
    if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Completed immutable original ground reference required')
    for key in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if old[key]!=dataset[key]:raise ValueError('Original F0 and source G(clean)/DURA ancestry differs')
    return old,dataset


_reference=_metadata
def _backend(c,device):return previous._train()._backend(c['reference_config']['train_config'],device)
def _legacy(c,condition):return previous._rows(c['reference_config']['output_root'],condition)
def _public(c,condition):return previous._public(c['reference_config'],condition)


def _old_calibration(c):
    root=Path(c['reference_config']['output_root']);m=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Original known-only source thresholds required')
        tau=z['conditional'].copy()
    if (m.get('status')!=previous.CALIBRATION_STATUS or m.get('code_commit')!=REFERENCE_CODE or m.get('run_id')!=previous.RUN_ID
            or m.get('method_names')!=list(previous.METHODS) or m.get('calibration_count')!=CAL_COUNT or m.get('known_only') is not True
            or m.get('unknown_inputs_used') is not False or m.get('test_inputs_used') is not False or tau.shape!=(3,6) or tau.dtype!=np.float64
            or not np.isfinite(tau).all() or m.get('conditional_thresholds')!=tau.tolist()):raise ValueError('Fixed old source calibration ancestor differs')
    return m,tau


def _coordinates(reader,values,device):
    xs=[];ls=[]
    for start in range(0,len(values),128):
        a=np.asarray(values[start:start+128])
        if a.shape!=(len(a),2,256) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Finite same-DURA full256 FP32 IQ required')
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            raw=reader.features(torch.tensor(a.tolist(),device=device,dtype=torch.float32));logits=reader.classify_features(raw)
            x=core.classifier_coordinates(reader,raw);head=reader.core.id_backbone.cls_head
            oracle=head.scale*F.linear(x,F.normalize(head.weight,dim=1,eps=1e-4))
            if not torch.equal(logits,oracle):raise ValueError('Exact native cosine-head FP32 coordinate oracle failed')
        x=np.asarray(x.detach().cpu().tolist(),np.float32);logits=np.asarray(logits.detach().cpu().tolist(),np.float32)
        if x.shape!=(len(a),160) or logits.shape!=(len(a),6) or not np.isfinite(x).all() or not np.isfinite(logits).all():raise ValueError('Actual normalized native160 and default-six logits required')
        xs.append(x);ls.append(logits)
    return np.concatenate(xs),np.concatenate(ls)


def _head_state(reader):
    head=reader.core.id_backbone.cls_head
    with torch.no_grad():weight=F.normalize(head.weight,dim=1,eps=1e-4)
    return dict(normalized_weight=np.asarray(weight.detach().cpu().tolist(),np.float32),scale=np.asarray(30.,np.float32))


def _smoke(b,device):
    t=torch.arange(256,device=device,dtype=torch.float32);iq=torch.stack((torch.sin(.07*t),torch.cos(.11*t))).repeat(2,1,1)
    with torch.no_grad():restored=b['dura'][1](iq)[0].detach().clone()
    x,l=_coordinates(b['reader'],np.asarray(restored.cpu().tolist(),np.float32),device)
    if x.shape!=(2,160) or l.shape!=(2,6):raise ValueError('Actual original F0/DURA no-query smoke failed')
    return dict(status='PASS',actual_original_F0_DURA=True,exact_native_classifier_coordinates=True,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,optimizer_updates=0,real_data_reads=0)


def _states(root):
    result={}
    for mode in MODES:
        with np.load(Path(root)/'models'/mode/'state.npz',allow_pickle=False) as z:result[mode]={k:z[k].copy() for k in z.files}
        core.validate_state(result[mode])
    return result


def _stack(legacy,x,logits,states,tau):
    v=[core.raw_scores(states[mode],x,logits) for mode in MODES]
    all_logits=np.stack((legacy['default_logits'][:,0],legacy['default_logits'][:,2],logits,logits),axis=1)
    closed=all_logits.argmax(2).astype(np.int64);raw=np.column_stack((legacy['raw_scores'][:,0],legacy['raw_scores'][:,2],v[0][:,1],v[1][:,1]))
    scores=np.column_stack((legacy['scores'][:,0],legacy['scores'][:,2],*[raw[:,j]-tau[j,closed[:,j]] for j in (2,3)]))
    a=dict(ids=legacy['ids'].copy(),default_logits=all_logits,closed=closed,raw_scores=raw,scores=scores,
        baseline_logits=legacy['default_logits'][:,0].copy(),baseline_closed=legacy['baseline_closed'].copy(),fresh_F0_logits=logits,
        classifier_coordinates=x,residual_norms=np.column_stack((-v[0][:,0],-v[1][:,0])))
    if 'rx_i' in legacy:a.update(rx_i=legacy['rx_i'].copy(),day_i=legacy['day_i'].copy())
    return a


def _validate_rows(a,condition):
    n=SOURCE_Q_COUNT if condition in CONDITIONS[:2] else TARGET_Q_COUNT;source=condition in CONDITIONS[:2]
    if (condition not in CONDITIONS or set(a)!=(SOURCE_FIELDS if source else ROW_FIELDS) or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U'
            or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
            or a['default_logits'].shape!=(n,4,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,4) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(2))
            or a['baseline_logits'].shape!=(n,6) or a['baseline_logits'].dtype!=np.float32 or not np.array_equal(a['baseline_logits'],a['default_logits'][:,0])
            or a['baseline_closed'].dtype!=np.int64 or a['baseline_closed'].shape!=(n,) or not np.array_equal(a['baseline_closed'],a['baseline_logits'].argmax(1))
            or a['fresh_F0_logits'].shape!=(n,6) or a['fresh_F0_logits'].dtype!=np.float32 or any(not np.array_equal(a['fresh_F0_logits'],a['default_logits'][:,j]) for j in (2,3))
            or a['classifier_coordinates'].shape!=(n,160) or a['classifier_coordinates'].dtype!=np.float32 or not np.isfinite(a['classifier_coordinates']).all()
            or np.any(np.linalg.norm(a['classifier_coordinates'].astype(np.float64),axis=1)>1.+2e-6)
            or any(a[k].shape!=(n,4) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))
            or a['residual_norms'].shape!=(n,2) or a['residual_norms'].dtype!=np.float64 or not np.isfinite(a['residual_norms']).all() or np.any(a['residual_norms']<0)):
        raise ValueError('Complete four-method geometry predictions and actual F0 classifier coordinates required')
    if source:
        risk.masks(a['rx_i'],a['day_i'])
        if len(a['rx_i'])!=n or a['rx_i'].dtype!=np.int64 or a['day_i'].dtype!=np.int64:raise ValueError('Aligned public source receiver/day metadata required')
    return a


def _rows(root,condition):
    with np.load(Path(root)/'queries'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate_rows(a,condition)


def _replay(logits,baseline):
    return dict(count=len(logits),maximum_absolute_logit_difference=float(np.max(np.abs(logits.astype(np.float64)-baseline.astype(np.float64)))),
        closed_mismatch_count=int(np.count_nonzero(logits.argmax(1)!=baseline.argmax(1))))


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source-geometry output required')
    old,dataset=_metadata(c);b=_backend(c,device)
    if b['reader_identity']!=dataset['parent_F0_identity'] or b['dura_identity']!=dataset['loaded_dura_identity']:raise ValueError('Same original F0 and fixed DURA data ancestry required')
    originals=(b['reader'],b['model'],b['dura'][0])
    if sum(p.numel() for p in b['reader'].parameters())!=232993 or any(p.requires_grad or p.dtype!=torch.float32 for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):raise ValueError('All original models must remain frozen FP32 eval')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,utilities._snapshot_model(m)) for m in originals]
    def check():
        for m,before in snapshots:utilities._check_model(m,before)
    smoke=_smoke(b,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    print('GROUND_GEOMETRY_ACTUAL_F0_DURA_SMOKE_PASS',flush=True)
    _save(out/'classifier_head.npz',_head_state(b['reader']))
    us=data.load_role(c['reference_config']['data_root'],'Us')
    try:
        if (len(us['ids'])!=US_COUNT or set(us['metadata'])!={'ids','rx_i','day_i','domain'} or us['manifest']!=dataset['roles']['Us']):raise ValueError('Exact hidden-TX source Us known-membership pool required')
        x,l=_coordinates(b['reader'],us['iq'],device);states={};diagnostics={}
        print('GROUND_GEOMETRY_SOURCE_US_COMPLETE',len(x),flush=True)
        for mode in MODES:
            states[mode],diagnostics[mode]=core.fit(x,l,us['metadata']['domain'],mode);_save(out/'models'/mode/'state.npz',states[mode])
            _write(out/'models'/mode/'complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],mode=mode,
                source_role='Us',source_count=US_COUNT,source_role_manifest=us['manifest'],parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],
                diagnostic=diagnostics[mode],optimizer_updates=0,source_TX_labels_used=False,CAL_inputs_used=False,OE_inputs_used=False,target_U_reads=0,target_Q_reads=0,source_rows_deployed=False))
            print('GROUND_GEOMETRY_MODE_FROZEN',mode,diagnostics[mode]['nuisance_rank'],diagnostics[mode]['alpha'],flush=True)
        (out/'models/complete.json').write_text(json.dumps(dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],modes=list(MODES),source_count=US_COUNT,
            source_role_manifest=us['manifest'],diagnostics=diagnostics,source_TX_labels_used=False,source_L_reads=0,source_OE_reads=0,target_U_reads=0,target_Q_reads=0,optimizer_updates=0,source_only=True),indent=2)+'\n',encoding='utf-8')
        check()
    finally:data.close(us)
    old_cal,old_tau=_old_calibration(c);cal=data.load_role(c['reference_config']['data_root'],'CAL');replays={}
    try:
        if len(cal['ids'])!=CAL_COUNT or cal['manifest']!=old_cal['calibration_manifest']:raise ValueError('Exact independent source known calibration role required')
        x,l=_coordinates(b['reader'],cal['iq'],device);replays['CAL']=_replay(l,cal['teacher_logits']);baseline=cal['teacher_logits'].argmax(1).astype(np.int64);meta=cal['metadata']
        tau=np.empty((4,6),np.float64);tau[0]=old_tau[0];tau[1]=old_tau[2]
        reports={METHODS[0]:deepcopy(old_cal['methods']['F0_energy']),METHODS[1]:deepcopy(old_cal['methods']['SSL_energy'])}
        for j,mode in enumerate(MODES,2):
            scores=core.raw_scores(states[mode],x,l)[:,1]
            state,report=risk.calibrate_known(cal['ids'],meta['y'],l.argmax(1).astype(np.int64),baseline,scores,meta['rx_i'],meta['day_i']);tau[j]=state['conditional_thresholds'];reports[METHODS[j]]=report
        check()
    finally:data.close(cal)
    (out/'calibration').mkdir();_save(out/'thresholds.npz',dict(conditional=tau))
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        methods=reports,conditional_thresholds=tau.tolist(),calibration_count=CAL_COUNT,calibration_manifest=old_cal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,
        target_reads=0,neural_models_frozen=True,source_subspaces_frozen_before_CAL=True,group_names=list(risk.GROUP_NAMES),source_controls_reused_exactly=True,
        new_threshold_values=12,reused_threshold_values=12,threshold_values=24))
    frozen=tau.copy();source_ids=None;target_ids=None
    for condition in CONDITIONS:
        legacy=_legacy(c,condition)
        if condition in CONDITIONS[:2]:
            q=data.load_query(c['reference_config']['data_root'],condition)
            try:
                if not np.array_equal(q['ids'],legacy['ids']) or not np.array_equal(q['baseline_logits'],legacy['default_logits'][:,0]) or any(not np.array_equal(q['metadata'][k],legacy[k]) for k in ('rx_i','day_i')):raise ValueError('Same source query physical records and canonical F0 baseline required')
                if source_ids is None:source_ids=q['ids'].copy()
                elif not np.array_equal(source_ids,q['ids']):raise ValueError('Source clean/star ordered physical records differ')
                x,l=_coordinates(b['reader'],q['iq'],device)
            finally:data.close(q)
        else:
            manifest,ids,iq=_public(c,condition)
            try:
                if not np.array_equal(ids,legacy['ids']):raise ValueError('Original canonical future records required')
                if target_ids is None:target_ids=ids.copy()
                elif not np.array_equal(target_ids,ids):raise ValueError('Cross clean/star ordered physical records differ')
                xs=[];ls=[]
                for start in range(0,len(ids),128):
                    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                        received=torch.tensor(np.asarray(iq[start:start+128]).tolist(),device=device,dtype=torch.float32);z=b['dura'][1](received)[0].detach().clone()
                    xc,lc=_coordinates(b['reader'],np.asarray(z.cpu().tolist(),np.float32),device);xs.append(xc);ls.append(lc)
                x,l=np.concatenate(xs),np.concatenate(ls)
            finally:
                if hasattr(iq,'_mmap'):iq._mmap.close()
        rows=_stack(legacy,x,l,states,tau);_validate_rows(rows,condition);_save(out/'queries'/condition/'predictions.npz',rows);check()
        replays[condition]=_replay(l,rows['baseline_logits'])
        if not np.array_equal(tau,frozen):raise ValueError('Source thresholds changed during query inference')
        print('GROUND_GEOMETRY_QUERY_COMPLETE',condition,len(rows['ids']),flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        data_capsule_id=dataset['capsule_id'],data_split_id=dataset['split_id'],dataset_identity=dataset,parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],loaded_dura_identity=dataset['loaded_dura_identity'],
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen=True,source_subspaces_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_fresh_replay=True,source_subspace_fit_count=2,nuisance_SVD_fit_count=1,source_subspace_fit_role='Us',source_subspace_fit_rows=US_COUNT,
        source_Us_reads=US_COUNT,source_Us_ID_reads=US_COUNT,source_U_GT_reads=0,source_L_reads=0,source_OE_reads=0,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,
        target_U_reads=0,U_ID_metadata_reads=0,target_U_GT_reads=0,prediction_optimizer_updates=0,optimizer_updates=0,source_density_fit_updates=0,
        calibration_known_count=CAL_COUNT,source_query_count=SOURCE_Q_COUNT,cross_query_count=TARGET_Q_COUNT,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,threshold_bytes=192,
        real_F_forward_rows=US_COUNT+CAL_COUNT+2*SOURCE_Q_COUNT+2*TARGET_Q_COUNT,real_SSL_forward_rows=0,real_student_forward_rows=0,G_forward_rows=2*TARGET_Q_COUNT,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],source_fit_diagnostics=diagnostics,
        inherited_data_costs={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))},replay_diagnostics=replays,
        classification_unchanged_from_cached_F0=all(v['closed_mismatch_count']==0 for v in replays.values()),original_models_unchanged=True,original_default_head_unchanged=True,
        exact_native_classifier_coordinates=True,source_only=True,ground_clean_source_fit=True,new_target_DA=False,primary_deployed_neural_parameters=499944,primary_threshold_values=6,
        source_rows_deployed=False,formal_target_accessed=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False)
    _write(out/'complete.json',result);print(STATUS,flush=True);return result


run=evaluate
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(dict(event='GROUND_GEOMETRY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True);run(_read(a.config),a.device)
if __name__=='__main__':main()
