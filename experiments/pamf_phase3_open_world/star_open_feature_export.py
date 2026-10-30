"""Frozen source-novel feature export; no fitting, target access or truth."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import runner
from . import star_open_world_train as parent
from . import star_open_world_model as ow
from . import star_relative_source as original
from .star_representation_cache import load_role,close_role
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261030-phase3-star-open-feature-diagnostic-s2026092701-r01'
PARENT_RUN='20261030-phase3-star-open-episodic-s2026092701-r02'
PARENT_COMMIT='1029bc4f4ab86022c75ae0f3332b5dfafc1fe909'
SCHEMA='pamf_phase3_open_feature_export_v1'
STATUS='STAR_OPEN_FEATURES_COMPLETE'
COUNT=3150
ARMS=('S0','S1','M')
FIELDS={'run_id','code_commit','train_configs','cache_root','output_root','batch_size'}
REPRESENTATIONS={'F0.fused160':dict(file='F0.npz',key='fused160',dimension=160)}|{
    a+'.'+key:dict(file=a+'.npz',key=key,dimension=dim) for a in ARMS for key,dim in (('fused160',160),('open64',64))}


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('train_configs'),dict) or set(c['train_configs'])!=set(ARMS)
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('cache_root','output_root'))):
        raise ValueError('Only the six-field frozen source feature diagnostic schema is allowed')
    inputs=[Path(c['cache_root']).resolve()]
    for a,t in c['train_configs'].items():
        parent.validate_config(t)
        if (t['run_id']!=PARENT_RUN or t['code_commit']!=PARENT_COMMIT or t['arm']!=a
                or Path(t['L_root']).resolve()!=Path(c['cache_root'],'L').resolve()
                or Path(t['OE_root']).resolve()!=Path(c['cache_root'],'OE').resolve()):
            raise ValueError('Only the exact frozen r02 three-arm source parents/cache are allowed')
        inputs.extend(Path(p).resolve() for p in (t['output_root'],t['native_config']['source_output'],t['native_config']['source_cache']))
    output=Path(c['output_root']).resolve()
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):raise ValueError('New output must be separate from all inputs')
    return c


def _tensor(x,device):return torch.tensor(np.asarray(x).tolist(),device=device,dtype=torch.float32)


def _features(model,x,is_original=False):
    try:
        with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
            if is_original:values={'fused160':model.features(x)}
            else:
                result=ow.forward(model,x);values=dict(fused160=result['raw_fused160'],open64=result['open64'])
            out={key:np.asarray(F.normalize(z.float(),dim=1,eps=1e-12).cpu().tolist(),np.float32) for key,z in values.items()}
    finally:
        if not is_original:ow.clear(model)
    if any(v.shape!=(len(x),160 if key=='fused160' else 64) or not np.isfinite(v).all() for key,v in out.items()):
        raise ValueError('Finite same-pass unit-or-zero source features required')
    return out


def _synthetic(device):
    from .star_channel import observe
    mixture=runner._read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json')
    t=np.arange(256);rows=[]
    for i,w in enumerate((.07,.11)):
        x=np.stack((np.cos(w*t),np.sin(w*t)))
        p0,_=observe(x,physical_id='source-open-feature-synthetic-'+str(i),receiver='1-19',day='2021_03_08',
            episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture)
        rows.append(p0)
    return _tensor(np.asarray(rows,np.float32),device)


def _backend(c,device):
    from .star_channel import _channel_api
    _channel_api()
    models={};identities={};dura=None
    for a,t in c['train_configs'].items():
        models[a],identities[a]=parent.load_trained_model(t,Path(t['output_root'])/'final.pt',device)
        done=runner._read(Path(t['output_root'])/'complete.json')
        if (done.get('status')!=parent.STATUS or done.get('identity')!=identities[a]
                or done.get('code_commit')!=PARENT_COMMIT or done.get('source_only') is not True
                or done.get('updates')!=2220 or done.get('model_buffers_unchanged') is not True):
            raise ValueError('Completed source-only r02 frozen parent required')
        if dura is not None and done['dura_identity']!=dura:raise ValueError('All three source parents require the same DURA ancestry')
        dura=done['dura_identity']
    n=c['train_configs']['S0']['native_config']
    old=original._backend(dict(reader_config={k:n[k] for k in ('base_release','frozen_sources','source_output','model_seed')},source_contract=n['source_contract']),device,with_dura=False)
    models={'F0':old['reader']}|models;identities={'F0':old['reader_identity']}|identities
    if any(identities[a]['parent_F0_identity']!=identities['F0'] for a in ARMS):raise ValueError('Original F0 cache/model ancestry differs')
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in models.values()):raise ValueError('All four inference models must be frozen')
    return models,identities,dura


def export(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('New immutable source diagnostic root required')
    start=time.perf_counter();models,identities,dura=_backend(c,device)
    snapshots={a:_snapshot_model(m) for a,m in models.items()}
    x=_synthetic(device)
    for a,m in models.items():_features(m,x,a=='F0');_check_model(m,snapshots[a])
    output.mkdir(parents=True,exist_ok=False)
    runner._write(output/'smoke.json',dict(status='PASS',actual_checkpoints=list(models),synthetic_G10_P0_rows_per_model=2,real_IQ_reads=0,target_reads=0,G_forward_rows=0))
    print('STAR_OPEN_FEATURE_SMOKE_PASS',flush=True)
    role=None
    try:
        role=load_role(Path(c['cache_root'])/'novel','novel');manifest=role['manifest'];ids=role['ids'];meta=role['metadata']
        if (manifest.get('role')!='novel' or manifest.get('count')!=COUNT or manifest.get('status')!='STAR_REPRESENTATION_ROLE_COMPLETE'
                or manifest.get('parent_F0_identity')!=identities['F0'] or manifest.get('dura_identity')!=dura
                or manifest.get('all_new_IQ_satellite') is not True or manifest.get('restored_input_path')!='P1'
                or set(meta)!={'rx_i','day_i','domain'} or len(ids)!=COUNT):
            raise ValueError('Only matching public source-novel G10P1 cache is allowed')
        for a,m in models.items():
            features=dict(fused160=[]) if a=='F0' else dict(fused160=[],open64=[])
            for offset in range(0,COUNT,128):
                values=_features(m,_tensor(role['restored'][offset:offset+128],device),a=='F0')
                for key,value in values.items():features[key].append(value)
                print('STAR_OPEN_FEATURE_PROGRESS',a,min(offset+128,COUNT),COUNT,flush=True)
            data=dict(ids=ids,rx_i=meta['rx_i'],day_i=meta['day_i'])|{key:np.concatenate(v) for key,v in features.items()}
            with (output/(a+'.npz')).open('xb') as f:np.savez(f,**data)
            _check_model(m,snapshots[a])
        for a,m in models.items():_check_model(m,snapshots[a])
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,source_count=COUNT,source_role='novel',
            source_manifest=manifest,parent_identities=identities,representations=REPRESENTATIONS,model_states_unchanged=True,
            model_updates=0,optimizer_updates=0,threshold_updates=0,target_reads=0,private_truth_reads=0,
            source_L_reads=0,source_V_reads=0,source_OE_reads=0,original_U_s_reads=0,G_forward_rows=0,
            input_path='cached_P1_only',received_cache_values_read=0,real_F_forward_rows=4*COUNT,synthetic_F_forward_rows=8,
            new_blind_confirmation=False,A_N=None,H_OW=None,final_phase3_complete=False,seconds=time.perf_counter()-start)
        runner._write(output/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        if role is not None:close_role(role)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    export(runner._read(a.config),a.device)


if __name__=='__main__':main()
