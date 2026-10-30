"""Frozen PAMF-OW L prototypes, known-only V risk and unconditional NV/Q inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import star_class_risk as risk
from . import star_consensus_score as common
from . import star_relative_target as public
from .development_transfer import _snapshot_model, _check_model


RUN_ID='20261030-phase3-star-open-episodic-s2026092701-r01'
ARMS=('S0','S1','M');PRIMARY_ARM='M';PRIMARY_METHOD='relation_max'
SCHEMA='pamf_phase3_star_open_world_prediction_v1'
STATUS='STAR_OPEN_WORLD_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_OPEN_WORLD_THRESHOLDS_FROZEN'
FIELDS={'run_id','code_commit','arm','train_config','checkpoint','cache_root','output_root','batch_size',
        'Q_public_root','Q_capsule_id','Q_split_id','dura_checkpoint','dura_release'}
COUNTS=dict(L=6300,V=27000,novel=3150);Q_COUNT=864
RECEIVERS,SCENES=common.RECEIVERS,common.SCENES
BASE_FIELDS={'ids','default_logits','relation_logits','closed','scores'}
KNOWN_FIELDS=BASE_FIELDS|{'y','rx_i','day_i','scenario','baseline_closed'}
NOVEL_FIELDS=BASE_FIELDS|{'rx_i','day_i'}
TARGET_FIELDS=BASE_FIELDS|{'baseline_closed'}
_read,_write=common._read,common._write


def _npz(path,data):
    with Path(path).open('xb') as stream:np.savez(stream,**data)


def _train_validate(c):
    from .star_open_world_train import validate_config
    return validate_config(c)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128):
        raise ValueError('Exact thirteen-field PAMF-OW evaluation config required; no U/truth/K')
    _train_validate(c['train_config']);t=c['train_config']
    if any(t.get(k)!=c[k] for k in ('run_id','code_commit','arm')):
        raise ValueError('Matching immutable trained row required')
    for key in FIELDS-{'train_config','batch_size'}:
        if not isinstance(c[key],str) or not c[key].strip():raise ValueError('Nonempty model/public paths and identities required')
    if (Path(c['checkpoint'])!=Path(t['output_root'])/'final.pt'
            or Path(t['L_root'])!=Path(c['cache_root'])/'L' or Path(t['OE_root'])!=Path(c['cache_root'])/'OE'
            or Path(c['Q_public_root']).name!='public' or Path(c['Q_public_root']).parent.name!='evaluation'):
        raise ValueError('Exact trained cache, final checkpoint and G10 future-Q public handles required')
    output=Path(c['output_root']).resolve()
    if any(output.is_relative_to(Path(p).resolve()) for p in (c['cache_root'],t['output_root'],c['Q_public_root'])):
        raise ValueError('Evaluation output cannot overwrite cache, checkpoint or public inputs')
    return c


def _identity(identity,c):
    required=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],checkpoint=c['checkpoint'],
        updates=2220,cumulative_updates=13320,identifier_parameters=284834,open_world_parameters=51841,
        training_container_parameters=952563,identity_head_replaced=False,input_path='cached_P1_only')
    if not isinstance(identity,dict) or any(identity.get(k)!=v for k,v in required.items()):
        raise ValueError('Actual complete frozen PAMF-OW checkpoint identity differs')
    # The native constructor ancestry and the public cache F0 identity have
    # different established schemas; do not relabel one as the other.
    original=identity.get('original_parent_identity',{})
    if (original.get('parent_epoch')!=50 or original.get('parent_updates')!=11100 or original.get('model_seed')!=2026092701
            or any(original.get(k)!=c['train_config']['native_config'].get(k) for k in ('source_output','source_contract','base_release'))):
        raise ValueError('Native E50 constructor ancestry must match the trained checkpoint')
    parent=identity.get('parent_F0_identity',{})
    if (parent.get('model')!='original_F0_E50' or parent.get('parameters')!=232993 or parent.get('epoch')!=50
            or parent.get('updates')!=11100 or parent.get('model_seed')!=2026092701
            or any(parent.get(k)!=c['train_config']['native_config'].get(k) for k in ('source_output','source_contract','base_release'))):
        raise ValueError('Exact original F0 E50 source parent required')
    return parent


def _backend(c,device):
    import torch
    from .star_channel import _channel_api
    _channel_api()  # Keep this release's G10 implementation before legacy F/G paths.
    from .star_open_world_train import load_trained_model
    from .star_relative_source import _backend as original_backend
    model,identity=load_trained_model(c['train_config'],c['checkpoint'],device)
    parent=_identity(identity,c);native=c['train_config']['native_config']
    old=original_backend(dict(reader_config={k:native[k] for k in ('base_release','frozen_sources','source_output','model_seed')},
        source_contract=native['source_contract'],dura_checkpoint=c['dura_checkpoint'],dura_release=c['dura_release']),device,with_dura=True)
    if old['reader_identity']!=parent:raise ValueError('Same-input original F0 baseline source identity differs')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return dict(model=model,original=old['reader'],dura=old['dura'],identity=identity,parent=parent,dura_identity=old['dura_identity'])


def _load_role(root,role):
    if role not in COUNTS:raise ValueError('Only source L/V/novel roles are consumed by evaluation')
    from .star_representation_cache import load_role
    return load_role(Path(root)/role,role)


def _close_role(role):
    from .star_representation_cache import close_role
    close_role(role)


def _role(role,name,backend):
    m=role['manifest']
    if (m.get('role')!=name or m.get('count')!=COUNTS[name] or m.get('parent_F0_identity')!=backend['parent']
            or m.get('dura_identity')!=backend['dura_identity']):
        raise ValueError('Source role must match the trained original-F0/DURA parents')


def _encode(model,values,device,prototypes=None):
    import torch
    from . import star_open_world_model as ow
    from .source_open_export import _tensor
    if values.dtype!=np.float32 or values.ndim!=3 or values.shape[1:]!=(2,256) or not np.isfinite(values).all():
        raise ValueError('Finite same-record P1 FP32 IQ batch required')
    try:
        with torch.no_grad():
            result=ow.forward(model,_tensor(values,device))
            logits=np.asarray(result['logits'].detach().cpu().tolist(),np.float32)
            features=np.asarray(result['open64'].detach().cpu().tolist(),np.float32)
            relations=None if prototypes is None else np.asarray(ow.relation(model,result['open64'],
                torch.tensor(prototypes.tolist(),dtype=torch.float32,device=device)).detach().cpu().tolist(),np.float32)
    finally:ow.clear(model)
    if (logits.shape!=(len(values),6) or features.shape!=(len(values),64)
            or not np.isfinite(logits).all() or not np.isfinite(features).all()
            or (relations is not None and (relations.shape!=(len(values),6) or not np.isfinite(relations).all()))):
        raise ValueError('Complete native default head, open64 and shared relation outputs required')
    return logits,features,relations


def _predict(model,values,device,prototypes=None):
    logits=[];features=[];relations=[]
    for start in range(0,len(values),128):
        z,u,r=_encode(model,np.asarray(values[start:start+128]),device,prototypes)
        logits.append(z)
        if prototypes is None:features.append(u)
        else:relations.append(r)
    return np.concatenate(logits),np.concatenate(features) if features else None,np.concatenate(relations) if relations else None


def _prototypes(features,y):
    u=np.asarray(features);y=np.asarray(y)
    if (u.ndim!=2 or u.shape[1]!=64 or u.dtype!=np.float32 or not len(u) or not np.isfinite(u).all()
            or y.shape!=(len(u),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(6))):
        raise ValueError('Only full six-class source-L open features and labels fit prototypes')
    means=np.asarray([u[y==label].astype(np.float64).mean(0) for label in range(6)])
    norm=np.linalg.norm(means,axis=1,keepdims=True)
    if np.any(norm<=1e-12):raise ValueError('Source class mean cannot be a degenerate prototype')
    return (means/norm).astype(np.float32)


def _restored(values,g,device):
    from .star_relation_oe_eval import _restored as restore
    return restore(values,g,device)


def _baseline(original,values,device):
    from .star_joint_ova_eval import _encode as encode_original
    z,_,_=encode_original(original,None,values,device)
    return z.argmax(1).astype(np.int64)


def _open(c,role):return public._open(dict(c,Q_expected_count=Q_COUNT),role)


def _probe(backend,device):
    from .star_channel import observe
    mixture=_read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json');t=np.arange(256)
    iq=np.asarray([observe(np.stack((np.cos(w*t),np.sin(w*t))),physical_id='open-world-smoke-'+str(i),receiver='1-19',
        day='2021_03_08',episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture)[0]
        for i,w in enumerate((.07,.11))],np.float32)
    p1=_restored(iq,backend['dura'],device)
    _encode(backend['model'],p1,device,np.eye(64,dtype=np.float32)[:6]);_baseline(backend['original'],p1,device)
    return dict(status='PASS',actual_candidate_F0_DURA_checkpoints=True,synthetic_G10_rows=2,
                candidate_forward_rows=2,original_F0_forward_rows=2,G_forward_rows=2,real_data_reads=0,optimizer_updates=0)


def _rows(ids,z,r,thresholds):
    closed=z.argmax(1).astype(np.int64);raw=r.max(1).astype(np.float64)
    return dict(ids=np.array(ids,copy=True),default_logits=z,relation_logits=r,closed=closed,scores=raw-thresholds[closed])


def evaluate(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('New immutable PAMF-OW evaluation root required')
    start=time.perf_counter();backend=_backend(c,device)
    snapshots=[(m,_snapshot_model(m)) for m in (backend['model'],backend['original'],backend['dura'][0])]
    def check():
        for m,s in snapshots:_check_model(m,s)
    smoke=_probe(backend,device);check();output.mkdir(parents=True);_write(output/'smoke.json',smoke)
    seen=set();role=_load_role(c['cache_root'],'L')
    try:
        _role(role,'L',backend);seen.update(role['ids'].tolist())
        _,features,_=_predict(backend['model'],role['restored'],device)
        prototypes=_prototypes(features,role['metadata']['y'])
        with (output/'prototypes.npy').open('xb') as stream:np.save(stream,prototypes,allow_pickle=False)
    finally:_close_role(role)
    check();frozen_prototypes=prototypes.copy();role=_load_role(c['cache_root'],'V')
    try:
        _role(role,'V',backend)
        if seen&set(role['ids'].tolist()):raise ValueError('Source L/V physical roles must be disjoint')
        seen.update(role['ids'].tolist());meta=role['metadata']
        z,_,r=_predict(backend['model'],role['restored'],device,prototypes)
        closed=z.argmax(1).astype(np.int64);baseline=np.array(role['baseline_closed'][:,1],copy=True)
        state,calibration=risk.calibrate_known(role['ids'],meta['y'],closed,baseline,r.max(1).astype(np.float64),meta['rx_i'],meta['scenario'])
        thresholds=state['conditional_thresholds'].copy()
        calibration.update(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],
                           method=PRIMARY_METHOD,selected_calibration_method='predicted_class',prototypes_fit_role='L')
        (output/'calibration').mkdir();_write(output/'calibration/complete.json',calibration)
        with (output/'thresholds.npy').open('xb') as stream:np.save(stream,thresholds,allow_pickle=False)
        _npz(output/'known_predictions.npz',dict(_rows(role['ids'],z,r,thresholds),y=meta['y'],rx_i=meta['rx_i'],
             day_i=meta['day_i'],scenario=meta['scenario'],baseline_closed=baseline))
    finally:_close_role(role)
    check();frozen_thresholds=thresholds.copy();print(CALIBRATION_STATUS,c['arm'],'source_RU_is_not_target_entry_gate',flush=True)
    role=_load_role(c['cache_root'],'novel')
    try:
        _role(role,'novel',backend)
        if seen&set(role['ids'].tolist()):raise ValueError('Source novel cannot overlap source fit/calibration')
        seen.update(role['ids'].tolist());z,_,r=_predict(backend['model'],role['restored'],device,prototypes)
        _npz(output/'novel_predictions.npz',dict(_rows(role['ids'],z,r,thresholds),rx_i=role['metadata']['rx_i'],day_i=role['metadata']['day_i']))
    finally:_close_role(role)
    check();_,ids,iq=_open(c,'Q');logits=[];relations=[];baselines=[]
    try:
        if seen&set(ids.tolist()):raise ValueError('Future Q and all source physical records must be disjoint')
        for offset in range(0,Q_COUNT,128):
            p1=_restored(np.asarray(iq[offset:offset+128]),backend['dura'],device)
            z,_,r=_encode(backend['model'],p1,device,prototypes);logits.append(z);relations.append(r)
            baselines.append(_baseline(backend['original'],p1,device))
        _npz(output/'target_predictions.npz',dict(_rows(ids,np.concatenate(logits),np.concatenate(relations),thresholds),
                                                baseline_closed=np.concatenate(baselines)))
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()
    check()
    if not np.array_equal(prototypes,frozen_prototypes) or not np.array_equal(thresholds,frozen_thresholds):
        raise ValueError('Source aggregates changed during frozen evaluation')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,
        model_identity=backend['identity'],parent_F0_identity=backend['parent'],dura_identity=backend['dura_identity'],
        method=PRIMARY_METHOD,primary_arm=PRIMARY_ARM,path='P1',prototype_count=6,prototype_fit_count=COUNTS['L'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],target_count=Q_COUNT,target_RX='24-1',target_receiver_index=26,
        source_truth_reads=0,target_truth_reads=0,target_U_reads=0,target_Q_fit=False,source_V_neural_fit=False,source_V_threshold_fit=True,
        prototypes_frozen_before_V=True,thresholds_frozen_before_novel_and_target=True,frozen_models_verified=True,source_aggregates_unchanged=True,
        source_unknown_success_target_entry_gate=False,eval_neural_updates=0,source_bank_deployed=False,per_record_source_rows_deployed=0,
        all_new_IQ_satellite=True,formal_target_accessed=False,target_DA=False,target_REG=False,A_N=None,H_OW=None,final_phase3_complete=False,
        identifier_parameters=284834,open_world_parameters=51841,training_container_parameters=952563,
        candidate_real_P1_forward_rows=sum(COUNTS.values())+Q_COUNT,candidate_synthetic_forward_rows=2,
        original_F0_real_P1_forward_rows=Q_COUNT,original_F0_synthetic_forward_rows=2,G_real_forward_rows=Q_COUNT,G_synthetic_forward_rows=2,
        seconds=time.perf_counter()-start)
    _write(output/'complete.json',done);print(STATUS,c['arm'],flush=True);return done


def main(argv=None):
    print(json.dumps(dict(event='STAR_OPEN_WORLD_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
