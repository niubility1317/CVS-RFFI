"""Frozen physical-open source geometry, with exact original-F0 feature reuse."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import runner
from . import star_physical_open_train as parent
from . import star_physical_open_model as model_api
from . import star_open_feature_export as old
from .star_representation_cache import load_role,close_role
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261030-phase3-star-physical-geometry-s2026092701-r01'
PARENT_RUN='20261030-phase3-star-physical-open-s2026092701-r01'
PARENT_COMMIT='31fa172946aeaecd2d3f1e784aee7366b7b5bee1'
SCHEMA='pamf_phase3_physical_geometry_export_v1'
STATUS='STAR_PHYSICAL_GEOMETRY_FEATURES_COMPLETE'
COUNT=3150
ARMS=('P_L','P_LOE')
FIELDS={'run_id','code_commit','train_configs','cache_root','baseline_feature_root','output_root','batch_size'}
REPRESENTATIONS={'F0.fused160':dict(file='F0.npz',key='fused160',dimension=160)}|{
    a+'.'+key:dict(file=a+'.npz',key=key,dimension=dim) for a in ARMS for key,dim in (('fused160',160),('open64',64))}


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('train_configs'),dict) or set(c['train_configs'])!=set(ARMS)
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('cache_root','baseline_feature_root','output_root'))):
        raise ValueError('Only fixed source-novel export and original-F0 baseline reuse fields allowed')
    inputs=[Path(c[k]).resolve() for k in ('cache_root','baseline_feature_root')]
    for a,t in c['train_configs'].items():
        parent.validate_config(t)
        if (t['run_id']!=PARENT_RUN or t['code_commit']!=PARENT_COMMIT or t['arm']!=a
                or any(Path(t[key]).resolve()!=Path(c['cache_root'],role).resolve() for key,role in (('L_root','L'),('OE_root','OE')))):
            raise ValueError('Only both fixed physical-open source parents and matching cache allowed')
        inputs.extend(Path(p).resolve() for p in (t['output_root'],t['native_config']['source_output'],t['native_config']['source_cache']))
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):raise ValueError('New output must be separate from every source input')
    return c


def _backend(c,device):
    from .star_channel import _channel_api
    _channel_api()
    models={};identities={};dura=None;f0=None
    for a,t in c['train_configs'].items():
        model,identity=parent.load_trained_model(t,Path(t['output_root'])/'final.pt',device)
        marker=runner._read(Path(t['output_root'])/'complete.json')
        if (marker.get('status')!=parent.STATUS or marker.get('identity')!=identity
                or marker.get('code_commit')!=PARENT_COMMIT or marker.get('source_only') is not True
                or marker.get('updates')!=2220 or marker.get('model_buffers_unchanged') is not True
                or any(marker.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','OE_truth_reads'))):
            raise ValueError('Completed frozen source-only physical-open parent required')
        if ((dura is not None and marker['dura_identity']!=dura)
                or (f0 is not None and identity['parent_F0_identity']!=f0)):
            raise ValueError('Both parents must share exact original F0 and DURA ancestry')
        if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Both inference models must be frozen')
        models[a]=model;identities[a]=identity;dura=marker['dura_identity'];f0=identity['parent_F0_identity']
    return models,{'F0':f0}|identities,dura


def _features(model,x):
    count=len(x)
    try:
        with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
            out=model_api.forward(model,x)
            values={'fused160':out['raw_fused160'],'open64':out['open64']}
            arrays={k:np.asarray(F.normalize(z.float(),dim=1,eps=1e-12).cpu().tolist(),np.float32) for k,z in values.items()}
    finally:model_api.clear(model)
    if any(value.shape!=(count,160 if k=='fused160' else 64) or not np.isfinite(value).all()
           for k,value in arrays.items()):
        raise ValueError('Finite same-pass unit-or-zero source features required')
    return arrays


def _baseline(root,role,f0,dura):
    marker=runner._read(Path(root)/'complete.json')
    required=dict(schema='pamf_phase3_open_feature_export_v1',status='STAR_OPEN_FEATURES_COMPLETE',
        run_id='20261030-phase3-star-open-feature-diagnostic-s2026092701-r01',
        source_count=COUNT,source_role='novel',model_states_unchanged=True,model_updates=0,optimizer_updates=0,
        threshold_updates=0,target_reads=0,private_truth_reads=0,input_path='cached_P1_only',received_cache_values_read=0)
    if (any(marker.get(k)!=v for k,v in required.items()) or marker.get('parent_identities',{}).get('F0')!=f0
            or marker.get('source_manifest')!=role['manifest'] or role['manifest']['dura_identity']!=dura):
        raise ValueError('Exact completed original-F0 same-P1 feature baseline required')
    with np.load(Path(root)/'F0.npz',allow_pickle=False) as stored:
        if set(stored.files)!={'ids','rx_i','day_i','fused160'}:raise ValueError('Baseline cannot contain labels or extra state')
        data={k:stored[k].copy() for k in stored.files}
    if (not np.array_equal(data['ids'],role['ids'])
            or any(not np.array_equal(data[k],role['metadata'][k]) for k in ('rx_i','day_i'))
            or data['fused160'].shape!=(COUNT,160) or data['fused160'].dtype!=np.float32
            or not np.isfinite(data['fused160']).all()):raise ValueError('Aligned original-F0 unit features required')
    norms=np.linalg.norm(data['fused160'].astype(np.float64),axis=1)
    if not np.all((norms==0)|np.isclose(norms,1.,rtol=0.,atol=2e-6)):raise ValueError('Baseline must be unit or exact-zero')
    return data,marker


def export(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('New immutable source geometry output required')
    started=time.perf_counter();models,identities,dura=_backend(c,device)
    snapshots={a:_snapshot_model(m) for a,m in models.items()};synthetic=old._synthetic(device)
    for a,m in models.items():_features(m,synthetic);_check_model(m,snapshots[a])
    output.mkdir(parents=True,exist_ok=False)
    runner._write(output/'smoke.json',dict(status='PASS',actual_frozen_checkpoints=list(ARMS),synthetic_F_rows=4,
        real_IQ_reads=0,target_reads=0,G_forward_rows=0,F0_checkpoint_loaded=False))
    print('STAR_PHYSICAL_GEOMETRY_SMOKE_PASS',flush=True);role=None
    try:
        role=load_role(Path(c['cache_root'])/'novel','novel');m=role['manifest'];meta=role['metadata']
        if (m.get('parent_F0_identity')!=identities['F0'] or m.get('dura_identity')!=dura or m.get('count')!=COUNT
                or m.get('role')!='novel' or m.get('source_only') is not True or m.get('all_new_IQ_satellite') is not True
                or m.get('restored_input_path')!='P1' or set(meta)!={'rx_i','day_i','domain'}):
            raise ValueError('Matching public source-novel G10P1 role required')
        baseline,baseline_marker=_baseline(c['baseline_feature_root'],role,identities['F0'],dura)
        with (output/'F0.npz').open('xb') as stream:np.savez(stream,**baseline)
        for a,model in models.items():
            rows={k:[] for k in ('fused160','open64')}
            for start in range(0,COUNT,c['batch_size']):
                arrays=_features(model,old._tensor(role['restored'][start:start+c['batch_size']],device))
                for k,x in arrays.items():rows[k].append(x)
                print('STAR_PHYSICAL_GEOMETRY_PROGRESS',a,min(start+c['batch_size'],COUNT),COUNT,flush=True)
            data=dict(ids=role['ids'],rx_i=meta['rx_i'],day_i=meta['day_i'])|{k:np.concatenate(v) for k,v in rows.items()}
            with (output/(a+'.npz')).open('xb') as stream:np.savez(stream,**data)
            _check_model(model,snapshots[a])
        for a,model in models.items():_check_model(model,snapshots[a])
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,source_count=COUNT,
            source_role='novel',source_manifest=m,parent_identities=identities,representations=REPRESENTATIONS,
            baseline_original_export_identity=baseline_marker,baseline_F0_features_reused=True,
            actual_frozen_checkpoints=list(ARMS),F0_feature_rows_reused=COUNT,F0_checkpoint_loaded=False,
            model_states_unchanged=True,model_updates=0,optimizer_updates=0,threshold_updates=0,target_reads=0,
            private_truth_reads=0,source_L_reads=0,source_V_reads=0,source_OE_reads=0,original_U_s_reads=0,
            G_forward_rows=0,input_path='cached_P1_only',received_cache_values_read=0,real_F_forward_rows=2*COUNT,
            synthetic_F_forward_rows=4,new_blind_confirmation=False,A_N=None,H_OW=None,final_phase3_complete=False,
            seconds=time.perf_counter()-started)
        runner._write(output/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        if role is not None:close_role(role)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    export(runner._read(a.config),a.device)


if __name__=='__main__':main()
