"""Frozen final-PCD full-cluster registration; future rows never fit state."""
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
from . import fresh_world_run as fresh
from . import fresh_pcd_train as training
from . import pcd_cluster_registry_core as core

RUN_ID='20261031-phase3-pcd-cluster-registry-s2026092701-r01'
SCHEMA='pamf_phase3_pcd_cluster_registry_prediction_v1'
STATUS='PCD_CLUSTER_REGISTRY_PREDICTIONS_COMPLETE'
REGISTRY_STATUS='PCD_CLUSTER_REGISTRY_FROZEN'
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
METHODS=core.METHODS
PRIMARY_METHOD=core.PRIMARY_METHOD
PRIMARY_INDEX=core.PRIMARY_INDEX
COUNTS=dict(L=6300,V=27000,novel=3150)
U_COUNT=576
Q_COUNT=10238
FIELDS={'run_id','code_commit','reference_config','output_root','batch_size'}
ROW_KEYS=('source_V','source_NV','Q_clean_DURA','Q_star_DURA')
U_FIELDS={'ids','features','default_logits','closed','base_scores','assignment'}
EVIDENCE={'raw_prediction','decision_score','support_cluster','support_ratio','source_distance'}
_read,_write,_same=fresh._read,fresh._write,fresh._same


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):raise ValueError('Exact five-field frozen registry config required; no truth/unknown K')
    r=fresh.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Exact completed fresh f4b5 PCD reference required')
    out=Path(c['output_root']).resolve();roots=[r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],r['source_control_config']['output_root'],r['source_control_config']['reference_config']['train_config']['IQ_root']]+[v['public_root'] for v in r['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable registry output overlaps reference/model/public input')
    return c


def _npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _reference(c):
    r=c['reference_config'];root=Path(r['output_root']);m=_read(root/'complete.json');old,_,_,_=fresh._reference(r)
    expected=dict(schema=fresh.SCHEMA,status=fresh.STATUS,run_id=fresh.RUN_ID,code_commit=REFERENCE_CODE,config=r,source_control_prediction_identity=old,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,classification_unchanged=True,private_truth_reads=0,U_truth_reads=0)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Completed frozen fresh-PCD reference required')
    identity=fresh._identity(m['model_identities']['TARGET_PCD'],r,old)
    if m['parent_F0_identity']!=old['parent_F0_identity'] or m['dura_identity']!=old['dura_identity']:raise ValueError('Original F0/G reference differs')
    fit=_read(root/'models/complete.json');state=_npz(root/'models/TARGET_PCD.npz');core.density.validate_state(state,'density_ratio')
    if (fit.get('status')!=fresh.MODELS_STATUS or fit.get('code_commit')!=REFERENCE_CODE or fit.get('identities',{}).get('TARGET_PCD')!=identity
            or any(fit.get(k) is not False for k in ('source_V_fit','novel_fit','target_Q_fit'))):raise ValueError('Original PCD source-only density fit role identity differs')
    thresholds=_npz(root/'thresholds.npz');tau=thresholds['conditional'];cal=_read(root/'calibration/complete.json')
    if (tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.isfinite(tau).all() or cal.get('status')!=fresh.CALIBRATION_STATUS or cal.get('code_commit')!=REFERENCE_CODE
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False):raise ValueError('Exact PCD source-calibrated thresholds required')
    for j in (4,5):
        if cal['methods'][fresh.METHODS[j]]['conditional_thresholds']!=tau[j].tolist():raise ValueError('Original PCD calibration/threshold mismatch')
    return m,state,tau[4:6].copy(),cal


def _backend(c,device):return fresh._backend(c['reference_config'],device)


def _load_model(c,device):return training.load_trained_model(c['reference_config']['train_config'],device)


def _smoke(b,model,device):
    from .star_open_feature_export import _synthetic
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        raw=_synthetic(device);restored=b['dura'][1](raw)[0].detach().clone();logits=b['reader'].classify_features(b['reader'].features(restored));z=b['model'](restored);final=model(restored)
    if logits.shape!=(2,6) or z.shape!=(2,160) or final.shape!=(2,160) or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (restored,logits,z,final)):raise ValueError('Actual F0/r03/G/finalPCD synthetic smoke failed')
    return dict(status='PASS',actual_F0_r03_DURA_PCD_checkpoints=True,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2,real_data_reads=0,optimizer_updates=0)


def _source_L(c,b,model,pm,device):
    role=fresh._iq_role(c['reference_config'],'L',pm['source_control_prediction_identity'])
    try:
        if len(role['ids'])!=COUNTS['L']:raise ValueError('Full original labelled source L required')
        features=fresh.source_eval._project(model,role['restored'],device)
        return role['ids'].copy(),features,role['metadata']['y'].copy(),role['manifest']
    finally:fresh.source_eval._close_role(role)


def _U(c,b,model,pm,state,tau,device):
    r=c['reference_config'];root=Path(r['train_config']['output_root']);declared,ids,_=fresh._training_metadata(r,pm['source_control_prediction_identity'])
    header=_read(root/'complete.json');manifest=header['U_manifest']
    if declared!=pm['model_identities']['TARGET_PCD'] or header.get('identity')!=declared or any(manifest.get(k)!=v for k,v in dict(count=U_COUNT,capsule_id=r['train_config']['U_capsule_id'],split_id=r['train_config']['U_split_id']).items()):raise ValueError('Same final PCD and cached star-U ancestry required')
    with np.load(root/'U_features.npz',allow_pickle=False) as z:
        if set(z.files)!={'ids','views','parent_z','neighbors'} or not np.array_equal(z['ids'],ids):raise ValueError('Original opaque U feature-cache identity required')
        views=z['views']
        if views.shape!=(3,U_COUNT,2,256) or views.dtype!=np.float32:raise ValueError('Original P1 weak/derived view cache shape differs')
        weak=views[0].copy()
    if not np.isfinite(weak).all():raise ValueError('Finite original weak P1 U required')
    blocks=[]
    for start in range(0,U_COUNT,128):
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            iq=torch.tensor(weak[start:start+128].tolist(),device=device,dtype=torch.float32);features=model(iq);logits=b['reader'].classify_features(b['reader'].features(iq))
        blocks.append(dict(features=np.asarray(features.cpu().tolist(),np.float32),default_logits=np.asarray(logits.cpu().tolist(),np.float32)))
    a={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};a['ids']=ids.copy();a['closed']=a['default_logits'].argmax(1).astype(np.int64);a['base_scores']=fresh._raw(state,a['features'],a['closed'])-tau[:,a['closed']].T
    return a,manifest


def _parent_rows(c,key):
    root=c['reference_config']['output_root']
    if key in ('source_V','source_NV'):return fresh._source_rows(root,'source_DURA','known' if key=='source_V' else 'novel')
    if key not in ROW_KEYS:raise ValueError('Fixed source/clean/star prediction row required')
    return fresh._target_rows(root,key[2:])


def _compact(old):
    a=dict(ids=old['ids'].copy(),closed=old['closed'].copy(),baseline_closed=old.get('baseline_closed',old['closed']).copy(),default_logits=old['default_logits'].copy(),
        features=old['signal_features'][:,2,:].copy(),base_margin=old['scores'][:,4:6].copy())
    for k in ('y','rx_i','day_i','scenario'):
        if k in old:a[k]=old[k].copy()
    return a


def _validate_features(a,n):
    core.base._ids(a['ids'],n);core.base._closed(a['closed'],n)
    if (a['features'].dtype!=np.float32 or a['features'].shape!=(n,160) or not np.isfinite(a['features']).all() or not np.allclose(np.linalg.norm(a['features'].astype(np.float64),axis=1),1,atol=2e-6,rtol=0)
            or a['default_logits'].dtype!=np.float32 or a['default_logits'].shape!=(n,6) or not np.isfinite(a['default_logits']).all() or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):raise ValueError('Actual final-PCD unit features and original F0 logits required')


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable cluster-registry output required')
    pm,state,tau,cal=_reference(c);b=_backend(c,device);model,identity=_load_model(c,device)
    if identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual PCD/F0/G ancestor differs')
    if any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Final PCD must be entirely frozen before registration')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;snapshots=[(m,fresh.best._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0],model)];frozen_state=deepcopy(state);frozen_tau=tau.copy()
    def check():
        for m,before in snapshots:fresh.best._check_model(m,before)
        if not _same(state,frozen_state) or not np.array_equal(tau,frozen_tau):raise ValueError('Frozen source Gaussian/threshold state changed')
    smoke=_smoke(b,model,device);check();print('PCD_REGISTRY_SMOKE_PASS',flush=True);out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    Lids,Lz,Ly,Lmanifest=_source_L(c,b,model,pm,device);source,source_diag=core.source_statistics(state,Lz,Ly);check();print('PCD_REGISTRY_SOURCE_COMPLETE',len(Lids),flush=True)
    U,Umanifest=_U(c,b,model,pm,state,tau,device);_validate_features(U,U_COUNT)
    if set(U['ids'].tolist())&set(Lids.tolist()):raise ValueError('Source and discovery physical IDs overlap')
    registry,assignment,diag=core.discover(state,source,U['features'],U['closed'],U['base_scores'],U['ids']);U['assignment']=assignment;check()
    fresh.best._save(out/'models/density.npz',state);fresh.best._save(out/'thresholds.npz',dict(conditional=tau));fresh.best._save(out/'registry.npz',registry);fresh.best._save(out/'discovery/U.npz',U)
    registry_meta=dict(schema=SCHEMA,status=REGISTRY_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],U_manifest=Umanifest,
        source_L_manifest=Lmanifest,source_statistics=source_diag,discovery=diag,registry_frozen=True,query_fit=False,source_GMM_refitted=False,thresholds_refitted=False,U_truth_reads=0,source_bank_deployed=False,U_bank_deployed=False)
    _write(out/'registry.json',registry_meta);print('PCD_REGISTRY_FROZEN',U_COUNT,len(registry['centers']),len(registry['anonymous_ids']),flush=True);frozen_registry=deepcopy(registry)
    used=set(Lids.tolist())|set(U['ids'].tolist());source_rows=set();paired=None
    for key in ROW_KEYS:
        a=_compact(_parent_rows(c,key));n=COUNTS['V' if key=='source_V' else 'novel'] if key.startswith('source') else Q_COUNT;_validate_features(a,n);core.base._float(a['base_margin'],(n,2),'cached PCD margins')
        if set(a['ids'].tolist())&used:raise ValueError('Future/source evaluation cannot overlap fitted source L or U')
        if key.startswith('source'):
            if source_rows&set(a['ids'].tolist()):raise ValueError('Known V and source novel roles overlap')
            source_rows.update(a['ids'].tolist())
        else:
            if set(a['ids'].tolist())&source_rows:raise ValueError('Future Q overlaps source evaluation')
            if paired is None:paired=a['ids'].copy()
            elif not np.array_equal(paired,a['ids']):raise ValueError('Clean/star future physical ID order differs')
        a.update(core.predict(state,registry,a['features'],a['closed'],a['base_margin']));fresh.best._save(out/'predictions'/(key+'.npz'),a);check()
        if not _same(registry,frozen_registry):raise ValueError('Frozen registry changed after evaluation access')
        print('PCD_REGISTRY_PREDICTIONS_SAVED',key,n,flush=True)
    files=dict(density='models/density.npz',thresholds='thresholds.npz',registry='registry.npz',registry_metadata='registry.json',discovery='discovery/U.npz',**{key:'predictions/'+key+'.npz' for key in ROW_KEYS})
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,files=files,reference_prediction_identity=pm,
        model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],U_manifest=Umanifest,source_L_manifest=Lmanifest,registry_cluster_count=len(registry['centers']),anonymous_count=len(registry['anonymous_ids']),
        registry_frozen_before_V_NV_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,original_models_unchanged=True,source_state_unchanged=True,thresholds_unchanged=True,registry_unchanged=True,classification_unchanged=False,
        source_L_reads=len(Lids),target_U_reads=U_COUNT,real_signal_forward_rows=len(Lids)+U_COUNT,real_F_forward_rows=U_COUNT,G_forward_rows=0,optimizer_updates=0,new_density_fits=0,new_threshold_fits=0,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,source_bank_deployed=False,U_bank_deployed=False,source_only=False,target_DA=True,target_REG=True,
        formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE')
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,state,tau,_=_reference(c)
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],reference_prediction_identity=pm,method_names=list(METHODS),primary_method=PRIMARY_METHOD,registry_frozen_before_V_NV_Q=True,
        all_three_conditions_complete=True,all_conditions_through_DURA=True,original_models_unchanged=True,source_state_unchanged=True,thresholds_unchanged=True,registry_unchanged=True,classification_unchanged=False,
        source_L_reads=COUNTS['L'],target_U_reads=U_COUNT,real_signal_forward_rows=COUNTS['L']+U_COUNT,real_F_forward_rows=U_COUNT,G_forward_rows=0,optimizer_updates=0,new_density_fits=0,new_threshold_fits=0,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,final_phase3_complete=False)
    if Path(c['output_root']).resolve()!=root.resolve() or any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete frozen registry prediction boundary required before truth')
    if not _same(_npz(root/'models/density.npz'),state) or not np.array_equal(_npz(root/'thresholds.npz')['conditional'],tau):raise ValueError('Original density/threshold state changed')
    registry=core.validate_registry(_npz(root/'registry.npz'));meta=_read(root/'registry.json')
    if meta.get('schema')!=SCHEMA or meta.get('status')!=REGISTRY_STATUS or meta.get('run_id')!=RUN_ID or meta.get('code_commit')!=c['code_commit'] or meta.get('registry_frozen') is not True or meta.get('query_fit') is not False or meta.get('model_identity')!=pm['model_identities']['TARGET_PCD'] or meta.get('U_manifest')!=done['U_manifest']:raise ValueError('Frozen registry identity differs')
    U=_npz(root/'discovery/U.npz')
    if set(U)!=U_FIELDS:raise ValueError('Only private label-free discovery evidence permitted')
    _validate_features(U,U_COUNT);core.base._float(U['base_scores'],(U_COUNT,2),'U margins')
    if U['assignment'].dtype!=np.int64 or U['assignment'].shape!=(U_COUNT,) or np.any((U['assignment']<-1)|(U['assignment']>=len(registry['centers']))):raise ValueError('Canonical U cluster assignments differ')
    if not np.allclose(U['base_scores'],fresh._raw(state,U['features'],U['closed'])-tau[:,U['closed']].T,atol=1e-10,rtol=1e-12):raise ValueError('U original PCD margins differ')
    rows={}
    for key in ROW_KEYS:
        a=_npz(root/'predictions'/(key+'.npz'));parent=_compact(_parent_rows(c,key));n=COUNTS['V' if key=='source_V' else 'novel'] if key.startswith('source') else Q_COUNT
        if set(a)!=set(parent)|EVIDENCE or any(not np.array_equal(a[k],v) for k,v in parent.items()):raise ValueError('Inherited same-input identities/features/margins changed')
        _validate_features(a,n);expected=core.predict(state,registry,a['features'],a['closed'],a['base_margin'])
        if any(not np.array_equal(a[k],v) for k,v in expected.items()):raise ValueError('Prototype/support decision evidence differs')
        if key.startswith('Q_') and set(a)&{'y','tx','scenario','day'}:raise ValueError('Future predictor cannot expose private target fields')
        rows[key]=a
    if not np.array_equal(rows['Q_clean_DURA']['ids'],rows['Q_star_DURA']['ids']) or any(set(a['ids'].tolist())&set(U['ids'].tolist()) for a in rows.values()):raise ValueError('Future pairing/U role isolation differs')
    return done,state,registry,U,rows


evaluate=run


def main():
    print(json.dumps(dict(event='PCD_CLUSTER_REGISTRY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
