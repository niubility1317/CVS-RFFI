"""P1-only frozen source evaluation of jointly trained OVA and native F0.

Known V is used only for empirical class-risk thresholds against the ORIGINAL
F0 cache baseline. All thresholds freeze before novel IQ and the original-F0
retrieval baseline are opened. No L/OE, target or private truth enters eval.
"""
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
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261029-phase3-star-joint-ova-s2026092701-r02'
SCHEMA='pamf_phase3_star_joint_ova_prediction_v1'
STATUS='STAR_JOINT_OVA_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_JOINT_OVA_THRESHOLDS_FROZEN'
ARMS=('detach_control','joint');PRIMARY_ARM='joint'
METHODS=('energy','ova_routed','ova_max');PRIMARY_METHOD='ova_max'
FIELDS={'run_id','code_commit','arm','train_config','checkpoint','cache_root','output_root','batch_size'}
TRAIN_FIELDS={'run_id','code_commit','arm','native_config','L_root','OE_root','teacher_root','output_root','steps','lr','L_seed','OE_seed'}
CACHE_SCHEMA='pamf_phase3_star_representation_cache_role_v1';CACHE_STATUS='STAR_REPRESENTATION_ROLE_COMPLETE'
CACHE_RUN='20261029-phase3-star-representation-s2026092701-r01'
COUNTS=dict(V=27000,novel=3150)
RECEIVERS,SCENES=common.RECEIVERS,common.SCENES
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed','default_logits','aux_logits'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','closed','default_logits','aux_logits','candidate_fused_unit','baseline_fused_unit'}
_read,_write,_ids=common._read,common._write,common._ids


def _npz(path,data):
    with Path(path).open('xb') as stream:np.savez(stream,**data)


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID or config.get('arm') not in ARMS
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('checkpoint','cache_root','output_root'))):
        raise ValueError('Only the fixed eight-field source-only P1 evaluation configuration is allowed')
    train=config['train_config']
    if (not isinstance(train,dict) or set(train)!=TRAIN_FIELDS or any(train.get(k)!=config[k] for k in ('run_id','code_commit','arm'))
            or train.get('steps')!=2220 or not isinstance(train.get('native_config'),dict)):
        raise ValueError('Matching source-only arm/run/code and fixed2220-step train configuration required')
    output=Path(config['output_root']).resolve()
    if any(output.is_relative_to(Path(p).resolve()) for p in (config['cache_root'],train['output_root'])):
        raise ValueError('Evaluation output cannot overwrite cache or trained state')
    return config


def _backend(config,device):
    import torch
    from .star_joint_ova_train import load_trained_model
    model,aux,identity=load_trained_model(config['train_config'],config['checkpoint'],device)
    parent=identity.get('original_parent_identity',{})
    if (any(identity.get(k)!=config[k] for k in ('run_id','arm','code_commit','checkpoint'))
            or identity.get('updates')!=2220 or identity.get('cumulative_updates')!=13320
            or any(identity.get(k)!=2220 for k in ('encoder_updates','identity_head_updates','auxiliary_updates'))
            or identity.get('identity_head_replaced') is not False or identity.get('input_path')!='cached_P1_only'
            or identity.get('model_parameters')!=232993 or identity.get('aux_parameters')!=966
            or parent.get('model')!='original_F0_E50' or parent.get('parameters')!=232993
            or parent.get('epoch')!=50 or parent.get('updates')!=11100
            or any(parent.get(k)!=config['train_config']['native_config'].get(k) for k in ('source_output','source_contract','base_release'))
            or model.training or aux.training or any(p.requires_grad or p.dtype!=torch.float32 for m in (model,aux) for p in m.parameters())):
        raise ValueError('Actual frozen native/aux checkpoint and matching original-F0 parent required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return model,model.id_backbone.encoder,aux,identity


def _load_original(config,device):
    from .star_consensus_run import _backend as original_backend
    backend,_=original_backend(config['train_config']['native_config'],device)
    return backend['reader'],backend['reader_identity']


def _original_snapshot(reader):return _snapshot_model(reader)
def _check_original(reader,snapshot):_check_model(reader,snapshot)


def _load_role(root,role):
    if role not in COUNTS:raise ValueError('Evaluation reads V and novel only, never L/OE/U/target')
    from .star_representation_cache import load_role
    return load_role(Path(root)/role,role)


def _close_role(bundle):
    from .star_representation_cache import close_role
    close_role(bundle)


def _role(bundle,role,parent,dura):
    if role not in COUNTS:raise ValueError('Only known V and unlabeled source novel roles are allowed')
    n=COUNTS[role];manifest=bundle['manifest'];meta=bundle['metadata']
    fields={'manifest','ids','received','restored','metadata'}|({'baseline_closed'} if role=='V' else set())
    mf={'rx_i','day_i','domain'}|({'y','scenario'} if role=='V' else set())
    if (set(bundle)!=fields or set(meta)!=mf or manifest.get('schema')!=CACHE_SCHEMA or manifest.get('status')!=CACHE_STATUS
            or manifest.get('run_id')!=CACHE_RUN or manifest.get('role')!=role or manifest.get('count')!=n
            or manifest.get('iq_shape')!=[n,2,256] or manifest.get('paths')!=['P0','P1']
            or manifest.get('source_only') is not True or manifest.get('all_new_IQ_satellite') is not True
            or manifest.get('parent_F0_identity')!=parent or (dura is not None and manifest.get('dura_identity')!=dura)
            or not _ids(bundle['ids'],n)):
        raise ValueError('Completed source-star cache role and exact parent identity required; novel truth forbidden')
    for key in ('received','restored'):
        if bundle[key].shape!=(n,2,256) or bundle[key].dtype!=np.float32:raise ValueError('Exact FP32 cached IQ shape required')
    for key in mf-{'scenario'}:
        if meta[key].shape!=(n,) or meta[key].dtype!=np.int64:raise ValueError('Aligned public source metadata required')
    if (not np.isin(meta['rx_i'],RECEIVERS).all() or not np.isin(meta['day_i'],(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(RECEIVERS,meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Only fixed original source RX/day/domain may be evaluated')
    if role=='V':
        b=bundle['baseline_closed']
        if (set(meta['y'].tolist())!=set(range(6)) or meta['scenario'].shape!=(n,) or meta['scenario'].dtype.kind!='U'
                or not np.isin(meta['scenario'],SCENES).all() or b.shape!=(n,2) or b.dtype!=np.int64 or np.any((b<0)|(b>=6))):
            raise ValueError('Known V six-class labels/scenes and original per-path baseline required')
    return bundle


def _encode(reader,aux,values,device,keep_unit=False):
    import torch
    from .source_open_export import _tensor,extract_branches
    if values.ndim!=3 or values.shape[1:]!=(2,256) or not np.isfinite(values).all():raise ValueError('Finite P1 IQ batches required')
    with torch.no_grad():
        branches,logits=extract_branches(reader,_tensor(values,device));raw=branches['fused160']
        if bool(torch.any(torch.linalg.vector_norm(raw,dim=1)<1e-12)):raise ValueError('Nonzero fused features required')
        unit=torch.nn.functional.normalize(raw,dim=1)
        g=aux(unit) if aux is not None else None
        z=np.asarray(logits.detach().cpu().tolist(),np.float32)
        a=None if g is None else np.asarray(g.detach().cpu().tolist(),np.float32)
        u=np.asarray(unit.detach().cpu().tolist(),np.float32) if keep_unit else None
    if z.shape!=(len(values),6) or not np.isfinite(z).all() or (a is not None and (a.shape!=z.shape or not np.isfinite(a).all())):
        raise ValueError('Finite unchanged default-head and six auxiliary logits required')
    return z,a,u


def _predict(reader,aux,values,device,batch_size,keep_unit=False):
    logits=[];auxiliary=[];units=[]
    for start in range(0,len(values),batch_size):
        z,g,u=_encode(reader,aux,values[start:start+batch_size],device,keep_unit)
        logits.append(z)
        if g is not None:auxiliary.append(g)
        if u is not None:units.append(u)
    return np.concatenate(logits),np.concatenate(auxiliary) if auxiliary else None,np.concatenate(units) if units else None


def scalars(default_logits,aux_logits):
    z=np.asarray(default_logits,np.float64);g=np.asarray(aux_logits,np.float64)
    if z.ndim!=2 or z.shape[1]!=6 or g.shape!=z.shape or not np.isfinite(z).all() or not np.isfinite(g).all():
        raise ValueError('Aligned finite six default/auxiliary logits required')
    c=z.argmax(1);top=z.max(1);energy=top+np.log(np.exp(z-top[:,None]).sum(1))
    return np.column_stack((energy,g[np.arange(len(c)),c],g.max(1))).astype(np.float64),c.astype(np.int64)


def _smoke(reader,aux,device):
    from .star_channel import observe
    mixture=_read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json')
    t=np.arange(256);values=[]
    for i,w in enumerate((.07,.11)):
        x,_=observe(np.stack((np.cos(w*t),np.sin(w*t))),physical_id='joint-ova-synthetic-'+str(i),receiver='1-19',day='2021_03_08',
            episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture);values.append(x)
    z,g,u=_encode(reader,aux,np.stack(values),device,True);scalars(z,g)
    return dict(status='PASS',actual_candidate_and_aux_checkpoint=True,synthetic_G10_P0_rows=2,candidate_forward_rows=2,G_forward_rows=0,
        real_source_reads=0,novel_reads=0,target_reads=0,optimizer_updates=0,
        scope='Synthetic P0 star checkpoint smoke only; real source training/evaluation is P1-only; not a P1 pipeline smoke')


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable per-arm evaluation root is required')
    started=time.perf_counter();model,reader,aux,identity=_backend(config,device)
    before=(_snapshot_model(model),_snapshot_model(aux));smoke=_smoke(reader,aux,device)
    def check():_check_model(model,before[0]);_check_model(aux,before[1])
    check()
    if smoke.get('status')!='PASS':raise ValueError('Actual candidate/aux synthetic checkpoint smoke must pass')
    output.mkdir(parents=True);_write(output/'smoke.json',smoke)
    parent=identity['original_parent_identity'];bundle=_load_role(config['cache_root'],'V')
    try:
        _role(bundle,'V',parent,None);dura=bundle['manifest']['dura_identity'];V_ids=set(bundle['ids'].tolist());meta=bundle['metadata']
        z,g,_=_predict(reader,aux,bundle['restored'],device,config['batch_size']);check();raw,closed=scalars(z,g)
        baseline=np.array(bundle['baseline_closed'][:,1],copy=True);conditional=[];global_tau=[];methods={};controls={}
        for j,name in enumerate(METHODS):
            state,report=risk.calibrate_known(bundle['ids'],meta['y'],closed,baseline,raw[:,j],meta['rx_i'],meta['scenario'])
            conditional.append(state['conditional_thresholds']);global_tau.append(state['global_threshold'])
            methods[name]=report['methods']['predicted_class'];controls[name]=report['methods']['global_control']
        thresholds=np.asarray(conditional,np.float64);check()
        _npz(output/'thresholds.npz',{'global':np.asarray(global_tau,np.float64),'conditional':thresholds})
        folder=output/'calibration';folder.mkdir()
        _write(folder/'complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],
            method_names=list(METHODS),conditional_thresholds=thresholds.tolist(),global_thresholds=global_tau,methods=methods,global_controls=controls,
            group_count=42,known_only=True,unknown_inputs_used=False,source_V_neural_fit=False,source_V_threshold_fit=True,
            source_only=True,target_reads=0,baseline='cache original F0 same-input P1 closed; never candidate closed'))
        _npz(output/'known_predictions.npz',dict(ids=bundle['ids'],y=meta['y'],rx_i=meta['rx_i'],day_i=meta['day_i'],scenario=meta['scenario'],
            default_logits=z,aux_logits=g,closed=closed,baseline_closed=baseline,scores=raw-thresholds[:,closed].T))
    finally:_close_role(bundle)
    frozen=thresholds.copy();check();print(CALIBRATION_STATUS,config['arm'],flush=True)
    original,original_identity=_load_original(config,device)
    if original_identity!=parent:raise ValueError('Original F0 novelty retrieval baseline identity must match the training/cache parent')
    original_snapshot=_original_snapshot(original)
    bundle=_load_role(config['cache_root'],'novel')
    try:
        _role(bundle,'novel',parent,dura)
        if V_ids&set(bundle['ids'].tolist()):raise ValueError('Known V and novel physical IDs must be disjoint')
        z,g,unit=_predict(reader,aux,bundle['restored'],device,config['batch_size'],True)
        _,_,base_unit=_predict(original,None,bundle['restored'],device,config['batch_size'],True)
        raw,closed=scalars(z,g);check();_check_original(original,original_snapshot)
        if not np.array_equal(frozen,thresholds):raise ValueError('Frozen thresholds changed during novel prediction')
        _npz(output/'novel_predictions.npz',dict(ids=bundle['ids'],rx_i=bundle['metadata']['rx_i'],day_i=bundle['metadata']['day_i'],
            default_logits=z,aux_logits=g,closed=closed,scores=raw-thresholds[:,closed].T,
            candidate_fused_unit=unit,baseline_fused_unit=base_unit))
    finally:_close_role(bundle)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],model_identity=identity,
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,primary_arm=PRIMARY_ARM,classification_path='P1',score_threshold=0.,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_only=True,target_reads=0,source_novel_truth_reads=0,
        source_L_records_consumed=0,source_OE_records_consumed=0,source_U_records_consumed=0,eval_neural_updates=0,G_forward_rows=0,
        real_candidate_P0_forward_rows=0,real_candidate_P1_forward_rows=sum(COUNTS.values()),synthetic_candidate_P0_forward_rows=2,
        candidate_forward_rows=sum(COUNTS.values())+2,original_F0_novel_forward_rows=COUNTS['novel'],
        model_parameters_buffers_unchanged=True,aux_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True,
        original_F0_loaded_after_all_thresholds=True,baseline='cache original F0 same-input P1 closed; never candidate closed',
        cache_identities=dict(parent_F0_identity=parent,dura_identity=dura),source_V_labels_used_for_thresholds=True,
        source_V_neural_fit=False,novel_truth_reads=0,novel_features_scope='truth-side cross-source-RX retrieval only; not anonymous accuracy or deployed source bank',
        source_bank_deployed=False,source_IQ_saved=False,final_phase3_complete=False,A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(output/'complete.json',result);print(STATUS,config['arm'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_JOINT_OVA_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);r=sub.add_parser('run')
    r.add_argument('--config',required=True);r.add_argument('--device',default='cuda:0');a=p.parse_args(argv);run(_read(a.config),a.device);return 0


if __name__=='__main__':raise SystemExit(main())
