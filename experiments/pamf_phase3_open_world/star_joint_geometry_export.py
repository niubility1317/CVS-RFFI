"""Frozen parts-only R1 mapping of source L and exact reuse of novel features."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import star_protected_open_cache as cache
from . import star_protected_open_eval as evaluation
from . import star_protected_open_model as ow
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261030-phase3-star-joint-geometry-s2026092701-r01'
PARENT_RUN='20261030-phase3-star-protected-open-s2026092701-r01'
PARENT_COMMIT='7dd90671968ddf58aab5a5b302b4d56371c8b9c9'
SCHEMA='pamf_phase3_joint_geometry_export_v1'
STATUS='STAR_JOINT_GEOMETRY_FEATURES_COMPLETE'
FIELDS={'run_id','code_commit','parent_eval_config','output_root','batch_size'}
COUNTS=dict(L=6300,novel=3150,union=9450)
FEATURE_FIELDS={'ids','rx_i','day_i','original_fused160','protected160'}
UNION_FIELDS=FEATURE_FIELDS|{'pool'}
ZEROS=('IQ_reads','source_OE_reads','source_V_reads','original_U_s_reads','target_reads','private_truth_reads',
       'optimizer_updates','model_updates','threshold_updates','G_forward_rows','real_F_forward_rows','synthetic_F_forward_rows')
_read,_write=cache._read,cache._write


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or not isinstance(c.get('code_commit'),str)
            or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact five-field source-only joint geometry config required; no truth/U/Q/K')
    e=evaluation.validate_config(c['parent_eval_config'])
    if e['run_id']!=PARENT_RUN or e['code_commit']!=PARENT_COMMIT or e['arm']!='R1':
        raise ValueError('Only the fixed original protected-open R1 parent is allowed')
    out=Path(c['output_root']).resolve();n=e['train_config']['native_config']
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out)
            for p in (e['output_root'],e['feature_root'],e['cache_root'],e['train_config']['output_root'],n['source_output'],n['source_cache'])):
        raise ValueError('New diagnostic output must be separate from every consumed parent input')
    return c


def _source_handle(feature_manifest):
    # This is the existing native constructor's public contract, not a new
    # receipt. No value is backfilled from the checkpoint under validation.
    return dict(classes=list(cache.native.CLASSES),source_rxs=list(cache.native.RX),source_days=[1,2,3],split_seed=392005,
        counts=dict(L_s=6300,U_s=56700,V=27000),source_contract=feature_manifest['config']['native_config']['source_contract'])


def _prediction_identity(m,e,identity):
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=PARENT_RUN,code_commit=PARENT_COMMIT,arm='R1',config=e,
        model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],source_only=True,
        known_count=27000,novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_truth_reads=0,target_reads=0,
        source_OE_reads=0,original_U_s_reads=0,native_identity_frozen=True,native_updates=0,eval_neural_updates=0,
        frozen_models_verified=True,identity_predictions_unchanged=True,novel_features_same_forward=True,source_roles_disjoint=True,
        aggregates_frozen_before_V=True,thresholds_frozen_before_novel=True,source_aggregates_unchanged=True,
        G_forward_rows=0,received_cache_values_read=0,formal_target_accessed=False)
    if not isinstance(m,dict) or any(m.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete same-forward frozen R1 source predictions required; scores are not a promotion gate')


def _novel_manifest(m,identity):
    expected=dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',role='novel',
        count=COUNTS['novel'],source_only=True,all_new_IQ_satellite=True,restored_input_path='P1',
        parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'])
    if not isinstance(m,dict) or any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Matching original source-novel G10P1 public handle required')


def validate_metadata(m):
    """Validate a completed export's metadata only; no IO, weights or labels."""
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,counts=COUNTS,source_only=True,input_path='cached_P1_features_only',
        model_states_unchanged=True,actual_adapter_checkpoint_loaded=True,native_F0_checkpoint_loaded=False,G_checkpoint_loaded=False,
        synthetic_adapter_rows=2,adapter_L_feature_rows=COUNTS['L'],novel_feature_rows_reused=COUNTS['novel'],
        source_RX_seen=True,source_novel_historically_exposed=True,eligible_primary=False,final_phase3_complete=False,
        **{k:0 for k in ZEROS})
    if not isinstance(m,dict) or any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete source-only frozen joint geometry export required')
    c=validate_config(m['config']);e=c['parent_eval_config'];identity=m['parent_model_identity']
    parent=evaluation._identity(identity,e);f=cache._manifest(m['source_feature_manifest'])
    if (m.get('code_commit')!=c['code_commit'] or m.get('parent_F0_identity')!=parent or m.get('dura_identity')!=identity['dura_identity']
            or f['parent_F0_identity']!=parent or f['dura_identity']!=identity['dura_identity'] or f['code_commit']!=PARENT_COMMIT
            or f['config']['output_root']!=e['feature_root'] or f['original_parent_identity']!=identity['original_parent_identity']
            or m.get('source_data_handle')!=_source_handle(f)):
        raise ValueError('Actual source data, original F0 and DURA ancestry must match the unchanged R1 checkpoint')
    _prediction_identity(m['reused_novel_prediction_identity'],e,identity);_novel_manifest(m['source_novel_manifest'],identity)
    return m


def _load_adapter(c,device):
    e=c['parent_eval_config'];tc=e['train_config'];folder=Path(tc['output_root']);done=_read(folder/'complete.json')
    fm=cache._manifest(_read(Path(e['feature_root'])/'complete.json'));identity=done.get('identity',{})
    parent=evaluation._identity(identity,e)
    required=dict(schema='pamf_phase3_star_protected_open_train_v1',status='STAR_PROTECTED_SOURCE_FROZEN',run_id=PARENT_RUN,
        code_commit=PARENT_COMMIT,config=tc,updates=2220,adapter_updates=2220,original_parent_updates=11100,
        model_native_state_unchanged=True,source_only=True,feature_manifest=fm,**{k:0 for k in cache.ZEROS})
    if (any(done.get(k)!=v for k,v in required.items()) or fm['code_commit']!=PARENT_COMMIT or fm['parent_F0_identity']!=parent
            or fm['dura_identity']!=identity['dura_identity'] or fm['original_parent_identity']!=identity['original_parent_identity']):
        raise ValueError('Complete source-only R1 training and its exact original feature-cache identity required')
    prediction=_read(Path(e['output_root'])/'complete.json');_prediction_identity(prediction,e,identity)
    value=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if (not isinstance(value,dict) or value.get('schema')!=required['schema'] or value.get('config')!=tc or value.get('identity')!=identity
            or value.get('feature_manifest')!=fm or value.get('source_data_handle')!=_source_handle(fm)
            or value.get('source_only') is not True or value.get('model_native_state_unchanged') is not True
            or 'model' in value or any(value.get(k)!=0 for k in cache.ZEROS)):
        raise ValueError('Only verified complete new-parts weights with the real unchanged source handle may be loaded')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026092701);parts=ow.ProtectedParts(True)
    state=value.get('open_world',{});expected=parts.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(expected) or any(not torch.is_tensor(v) or v.dtype!=torch.float32
            or v.shape!=expected[k].shape or not bool(torch.isfinite(v).all()) for k,v in state.items())):
        raise ValueError('Strict complete finite FP32 protected adapter and relation state required')
    parts.load_state_dict(state,strict=True)
    if sum(p.numel() for p in parts.parameters())!=66177:raise ValueError('Complete protected160 parts parameter count differs')
    model=nn.Module();model.add_module('open_world',parts.to(device=device,dtype=torch.float32));model.eval().requires_grad_(False)
    return dict(model=model,identity=identity,feature_manifest=fm,prediction=prediction,source_data_handle=value['source_data_handle'])


def _tensor(x,device):return torch.tensor(np.asarray(x).tolist(),dtype=torch.float32,device=device)


def _smoke(model,device):
    with torch.random.fork_rng(devices=[]):
        generator=torch.Generator().manual_seed(2026103017);f=torch.randn(2,160,generator=generator).to(device)
        contributions=torch.stack((.4*f,.3*f,.2*f,.1*f),1)
    with torch.no_grad():
        out=ow.encode(model,f,contributions);r=ow.relation(model,out['protected160'],F.normalize(f,dim=1))
    delta=out['raw_protected_fused160']-f
    if not bool(torch.isfinite(r).all()) or bool((delta.norm(dim=1)>.1*f.norm(dim=1)+2e-6).any()):raise ValueError('Actual frozen adapter smoke failed')
    return dict(status='PASS',actual_adapter_checkpoint_loaded=True,synthetic_adapter_rows=2,
        native_F0_checkpoint_loaded=False,G_checkpoint_loaded=False,F_forward_rows=0,G_forward_rows=0,real_source_reads=0)


def _load_L(e):return cache.load_features(e['feature_root'],'L')


def _encode_L(model,f,device):
    original=[];protected=[]
    with torch.no_grad():
        for start in range(0,COUNTS['L'],128):
            raw=_tensor(f['raw_original_fused160'][start:start+128],device)
            out=ow.encode(model,raw,_tensor(f['contributions'][start:start+128],device))
            original.append(np.asarray(F.normalize(raw,dim=1,eps=1e-12).cpu().tolist(),np.float32))
            protected.append(np.asarray(out['protected160'].cpu().tolist(),np.float32))
            print('STAR_JOINT_GEOMETRY_L_PROGRESS',min(start+128,COUNTS['L']),COUNTS['L'],flush=True)
    return dict(original_fused160=np.concatenate(original),protected160=np.concatenate(protected))


def _public(a,n):
    if (not evaluation.common._ids(a['ids'],n) or any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],roster).all()
            for k,roster in (('rx_i',evaluation.RECEIVERS),('day_i',(1,2,3))))):
        raise ValueError('Unique source opaque IDs and aligned legal public RX/day metadata required')


def _units(a,n):
    for key in ('original_fused160','protected160'):
        x=a[key]
        if x.shape!=(n,160) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Finite FP32 unit160 representations required')
        norm=np.linalg.norm(x.astype(np.float64),axis=1)
        if not np.all((norm==0)|np.isclose(norm,1.,rtol=0.,atol=2e-6)):raise ValueError('Representations must be unit or exact zero')


def _novel(e,identity):
    role=Path(e['cache_root'])/'novel';manifest=_read(role/'manifest.json');_novel_manifest(manifest,identity)
    ids=np.load(role/'ids.npy',allow_pickle=False)
    with np.load(role/'metadata.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'rx_i','day_i','domain'}:raise ValueError('Only public source-novel RX/day/domain metadata allowed')
        meta={k:saved[k].copy() for k in saved.files}
    reference=dict(ids=ids,rx_i=meta['rx_i'],day_i=meta['day_i']);_public(reference,COUNTS['novel'])
    if (meta['domain'].dtype!=np.int64 or meta['domain'].shape!=(COUNTS['novel'],)
            or not np.array_equal(meta['domain'],np.searchsorted(evaluation.RECEIVERS,meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Source-novel public domain layout differs')
    with np.load(Path(e['output_root'])/'novel_features.npz',allow_pickle=False) as saved:
        if set(saved.files)!=FEATURE_FIELDS:raise ValueError('Exact prior novel geometry fields required; labels and pool forbidden')
        a={k:saved[k].copy() for k in saved.files}
    _public(a,COUNTS['novel']);_units(a,COUNTS['novel'])
    if any(not np.array_equal(a[k],reference[k]) for k in reference):raise ValueError('Reused novel features must preserve exact public ID/domain order')
    return a,manifest


def export(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('New immutable joint geometry output root required')
    start=time.perf_counter();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=_load_adapter(c,device);model=b['model'];snapshot=_snapshot_model(model);smoke=_smoke(model,device);_check_model(model,snapshot)
    if smoke.get('status')!='PASS':raise ValueError('Actual adapter no-query smoke must pass before source feature values')
    output.mkdir(parents=True,exist_ok=False);_write(output/'resolved_config.json',c);_write(output/'smoke.json',smoke)
    f=_load_L(c['parent_eval_config'])
    if f['manifest']!=b['feature_manifest']:raise ValueError('Loaded L features and actual adapter cache identity differ')
    public=dict(ids=f['ids'],rx_i=f['metadata']['rx_i'],day_i=f['metadata']['day_i']);_public(public,COUNTS['L'])
    L=dict(public,**_encode_L(model,f,device));_units(L,COUNTS['L']);_check_model(model,snapshot)
    novel,nm=_novel(c['parent_eval_config'],b['identity'])
    if set(L['ids'].tolist())&set(novel['ids'].tolist()):raise ValueError('Source-L and source-novel physical IDs must be disjoint')
    union={k:np.concatenate((L[k],novel[k])) for k in FEATURE_FIELDS}
    union['pool']=np.concatenate((np.zeros(COUNTS['L'],np.int64),np.ones(COUNTS['novel'],np.int64)))
    _public(union,COUNTS['union']);_units(union,COUNTS['union']);_check_model(model,snapshot)
    with (output/'union.npz').open('xb') as stream:np.savez(stream,**union)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,counts=dict(COUNTS),
        parent_model_identity=b['identity'],parent_F0_identity=b['identity']['parent_F0_identity'],dura_identity=b['identity']['dura_identity'],
        source_feature_manifest=b['feature_manifest'],source_data_handle=b['source_data_handle'],source_novel_manifest=nm,
        reused_novel_prediction_identity=b['prediction'],model_states_unchanged=True,actual_adapter_checkpoint_loaded=True,
        native_F0_checkpoint_loaded=False,G_checkpoint_loaded=False,synthetic_adapter_rows=2,adapter_L_feature_rows=COUNTS['L'],
        novel_feature_rows_reused=COUNTS['novel'],source_only=True,input_path='cached_P1_features_only',source_RX_seen=True,
        source_novel_historically_exposed=True,diagnostic_only=True,eligible_primary=False,final_phase3_complete=False,
        seconds=time.perf_counter()-start,**{k:0 for k in ZEROS})
    validate_metadata(done);_write(output/'complete.json',done);print(STATUS,flush=True);return done


def main(argv=None):
    print(json.dumps(dict(event='STAR_JOINT_GEOMETRY_EXPORT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    export(_read(a.config),a.device)


if __name__=='__main__':main()
