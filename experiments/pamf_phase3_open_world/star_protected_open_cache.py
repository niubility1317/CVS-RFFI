"""One-pass frozen original-F0 features from existing source G10 P1 roles."""
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
from . import runner
from . import source_open_embedding_train as native
from . import star_protected_open_model as ow

RUN_ID='20261030-phase3-star-protected-open-s2026092701-r01'
SCHEMA='pamf_phase3_protected_source_features_v1'
STATUS='STAR_PROTECTED_SOURCE_FEATURES_COMPLETE'
COUNTS=dict(L=6300,OE=6134)
FIELDS={'run_id','code_commit','native_config','L_root','OE_root','output_root','batch_size'}
NATIVE_FIELDS=runner.COMMON|{'source_cache'}
META=dict(L={'y','rx_i','day_i','domain'},OE={'rx_i','day_i','domain'})
FEATURE_SHAPES=dict(raw_original_fused160=(160,),contributions=(4,160),default_logits=(6,))
ZEROS=('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','private_truth_reads',
       'OE_truth_reads','G_forward_rows','raw_IQ_reads','received_cache_values_read','native_updates','identity_head_updates')
_read,_write=runner._read,runner._write


def _native_config(n):
    return dict(n,arm='supervised_only',steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)


def _native_validate(n,code):
    if (not isinstance(n,dict) or set(n)!=NATIVE_FIELDS or n.get('run_id')!=RUN_ID or n.get('code_commit')!=code
            or not isinstance(n.get('row_id'),str) or n['row_id'] not in [a+'-s2026092701' for a in ('cache','N0','R0','R1')]):
        raise ValueError('Exact original-F0 constructor configuration for this run required')
    native.validate_config(_native_config(n))


def _base_validate(c,fields):
    if (not isinstance(c,dict) or set(c)!=fields or c.get('run_id')!=RUN_ID or not isinstance(c.get('code_commit'),str)
            or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None):
        raise ValueError('Exact source-only protected-open configuration required; no private/target roles')
    _native_validate(c['native_config'],c['code_commit'])


def validate_config(c):
    _base_validate(c,FIELDS)
    if type(c['batch_size']) is not int or c['batch_size']!=128:raise ValueError('Fixed batch size 128 required')
    for k in ('L_root','OE_root','output_root'):
        if not isinstance(c[k],str) or not c[k].strip():raise ValueError('Nonempty separate public source/output paths required')
    out=Path(c['output_root']).resolve();n=c['native_config']
    inputs=[Path(p).resolve() for p in (c['L_root'],c['OE_root'],n['source_output'],n['source_cache'],n['output_root'])]
    if inputs[0]==inputs[1] or any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError('Feature output must be separate from source roles, original model and native constructor output')
    return c


def _construct(n,device,residual_enabled):
    from .star_channel import _channel_api
    _channel_api()  # Canonical G10 before legacy model dependency paths.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return ow.install(native.make_trainable_model(_native_config(n),device),residual_enabled,seed=2026092701)


def _tensor(x,device,dtype=torch.float32):return torch.tensor(np.asarray(x).tolist(),dtype=dtype,device=device)


def _synthetic(device):
    from .star_open_feature_export import _synthetic as synthetic
    return synthetic(device)


def _snapshot(model,native_only=False):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not native_only or not k.startswith('open_world.')}


def _unchanged(model,before):
    after=model.state_dict()
    if any(k not in after or not torch.equal(v,after[k].detach().cpu()) for k,v in before.items()):
        raise ValueError('Original frozen model parameters or buffers changed')


def _smoke(model,device):
    before=_snapshot(model);x=_synthetic(device)
    try:
        out=ow.forward(model,x)
        with torch.no_grad():reference=model.id_backbone.encoder.classify_features(out['raw_original_fused160'])
        f=out['raw_original_fused160'];delta=out['raw_protected_fused160']-f
        if (not torch.equal(reference,out['logits']) or not torch.equal(f,out['raw_protected_fused160'])
                or bool((delta.norm(dim=1)>.1*f.norm(dim=1)+2e-6).any())):
            raise ValueError('Actual original-head identity and zero-initialized bounded adapter smoke failed')
        _unchanged(model,before)
        return dict(status='PASS',actual_original_checkpoint=True,synthetic_F_forward_rows=2,
            synthetic_G10_received_rows=2,G_forward_rows=0,real_source_rows=0,original_logits_unchanged=True,
            model_state_unchanged=True,synthetic_input='G10_received_no_G; interface smoke only')
    finally:ow.clear(model)


def _parent(parent,original):
    if (not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,
            epoch=50,updates=11100,model_seed=2026092701).items()) or not isinstance(original,dict)
            or original.get('parent_epoch')!=50 or original.get('parent_updates')!=11100
            or any(parent.get(k)!=original.get(k) for k in ('model_seed','source_output','source_contract','base_release'))):
        raise ValueError('Actual original F0 E50 source ancestry must match cached features')


def _metadata(ids,meta,role):
    n=COUNTS[role]
    if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or not isinstance(meta,dict) or set(meta)!=META[role]
            or any(a.shape!=(n,) or a.dtype!=np.int64 for a in meta.values())):
        raise ValueError('Exact aligned public source metadata required; OE identity labels forbidden')
    rx,day,domain=(meta[k] for k in ('rx_i','day_i','domain'))
    if (not np.isin(rx,native.RX).all() or not np.isin(day,(1,2,3)).all()
            or not np.array_equal(domain,np.searchsorted(native.RX,rx)*3+day-1)
            or set(domain.tolist())!=set(range(15))):raise ValueError('All fifteen source RX/day cells required')
    if role=='L':
        y=meta['y']
        if (not np.isin(y,range(6)).all() or n%90 or not np.all(np.bincount(y*15+domain,minlength=90)==n//90)):
            raise ValueError('The original balanced six-class source-L physical grid is required')


def _source_role(role,name,model):
    _metadata(role['ids'],role['metadata'],name);m=role['manifest'];n=COUNTS[name]
    if (m.get('status')!='STAR_REPRESENTATION_ROLE_COMPLETE' or m.get('schema')!='pamf_phase3_star_representation_cache_role_v1'
            or m.get('role')!=name or m.get('count')!=n or m.get('source_only') is not True
            or m.get('all_new_IQ_satellite') is not True or m.get('restored_input_path')!='P1'
            or role['restored'].shape!=(n,2,256) or role['restored'].dtype!=np.float32):
        raise ValueError('Only complete existing G10 source-L/OE restored P1 roles may be read')
    _parent(m.get('parent_F0_identity'),model._phase3_parent_identity)
    d=m.get('dura_identity',{})
    if any(d.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()):
        raise ValueError('Existing restored roles must use frozen DURA v1 ancestry')


def _load_role(root,role):
    from .star_representation_cache import load_role
    return load_role(root,role)


def _close_role(role):
    from .star_representation_cache import close_role
    close_role(role)


def _encode(model,x,device):
    if x.dtype!=np.float32 or x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():
        raise ValueError('Finite existing FP32 P1 observations required')
    try:
        with torch.no_grad():out=ow.forward(model,_tensor(x,device))
        return {k:np.asarray(out[key].detach().cpu().tolist(),np.float32) for k,key in
            (('raw_original_fused160','raw_original_fused160'),('contributions','contributions'),('default_logits','logits'))}
    finally:ow.clear(model)


def build(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('New immutable feature-cache root required')
    started=time.perf_counter();model=_construct(c['native_config'],device,False).eval().requires_grad_(False)
    smoke=_smoke(model,device);before=_snapshot(model)
    if smoke.get('status')!='PASS':raise ValueError('Actual-checkpoint no-query smoke must pass before source cache reads')
    output.mkdir(parents=True,exist_ok=False);_write(output/'resolved_config.json',c);_write(output/'smoke.json',smoke)
    source_manifests={};seen=set()
    for role in COUNTS:
        bundle=None
        try:
            bundle=_load_role(c[role+'_root'],role);_source_role(bundle,role,model);m=bundle['manifest']
            if seen.intersection(bundle['ids'].tolist()):raise ValueError('Source-L/OE physical IDs must be disjoint')
            seen.update(bundle['ids'].tolist())
            if source_manifests and any(m[k]!=source_manifests['L'][k] for k in ('parent_F0_identity','dura_identity')):
                raise ValueError('Both roles must have identical original F0 and DURA ancestry')
            values={k:[] for k in FEATURE_SHAPES}
            for start in range(0,COUNTS[role],128):
                result=_encode(model,np.asarray(bundle['restored'][start:start+128]),device)
                for key,a in result.items():values[key].append(a)
                print('STAR_PROTECTED_CACHE_PROGRESS',role,min(start+128,COUNTS[role]),COUNTS[role],flush=True)
            arrays=dict(ids=bundle['ids'],**bundle['metadata'],**{k:np.concatenate(v) for k,v in values.items()})
            with (output/(role+'.npz')).open('xb') as stream:np.savez(stream,**arrays)
            source_manifests[role]=m;_unchanged(model,before)
        finally:
            if bundle is not None:_close_role(bundle)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,counts=dict(COUNTS),
        parent_F0_identity=source_manifests['L']['parent_F0_identity'],dura_identity=source_manifests['L']['dura_identity'],
        original_parent_identity=model._phase3_parent_identity,source_manifests=source_manifests,
        source_only=True,input_path='cached_P1_only',model_native_state_unchanged=True,model_states_unchanged=True,
        optimizer_updates=0,model_updates=0,F_forward_rows=sum(COUNTS.values()),synthetic_F_forward_rows=2,
        source_L_rows=COUNTS['L'],source_OE_rows=COUNTS['OE'],feature_fields=list(FEATURE_SHAPES),
        seconds=time.perf_counter()-started,**{k:0 for k in ZEROS})
    _write(output/'complete.json',done);print(STATUS,flush=True);return done


def _manifest(m):
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,counts=COUNTS,source_only=True,input_path='cached_P1_only',
        model_native_state_unchanged=True,model_states_unchanged=True,optimizer_updates=0,model_updates=0,
        F_forward_rows=sum(COUNTS.values()),synthetic_F_forward_rows=2,**{k:0 for k in ZEROS})
    if not isinstance(m,dict) or any(m.get(k)!=v for k,v in required.items()):raise ValueError('Complete frozen source feature cache required')
    validate_config(m['config'])
    if m.get('code_commit')!=m['config']['code_commit']:raise ValueError('Feature cache code identity differs')
    _parent(m.get('parent_F0_identity'),m.get('original_parent_identity'))
    original=m['original_parent_identity'];n=m['config']['native_config'];d=m.get('dura_identity',{})
    if (any(original.get(k)!=n[k] for k in ('source_output','source_contract','source_cache','base_release','model_seed'))
            or not isinstance(d,dict) or any(d.get(k)!=v for k,v in
                dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items())):
        raise ValueError('Feature cache must retain its actual original F0 constructor and frozen DURA ancestry')
    sources=m.get('source_manifests',{})
    if not isinstance(sources,dict) or set(sources)!=set(COUNTS):raise ValueError('Both public source role handles required')
    for role in COUNTS:
        s=sources[role]
        if (not isinstance(s,dict) or s.get('role')!=role or s.get('count')!=COUNTS[role] or s.get('restored_input_path')!='P1'
                or s.get('schema')!='pamf_phase3_star_representation_cache_role_v1' or s.get('status')!='STAR_REPRESENTATION_ROLE_COMPLETE'
                or s.get('source_only') is not True or s.get('all_new_IQ_satellite') is not True
                or s.get('parent_F0_identity')!=m['parent_F0_identity'] or s.get('dura_identity')!=m['dura_identity']):
            raise ValueError('Matching source role ancestry/route/counts required')
    return m


def load_features(root,role):
    if role not in COUNTS:raise ValueError('Only source-L or OE features are available')
    root=Path(root);manifest=_manifest(_read(root/'complete.json'));n=COUNTS[role]
    with np.load(root/(role+'.npz'),allow_pickle=False) as saved:
        if set(saved.files)!={'ids'}|META[role]|set(FEATURE_SHAPES):raise ValueError('Exact role-specific fields required; private labels forbidden')
        data={k:saved[k].copy() for k in saved.files}
    meta={k:data[k] for k in META[role]};_metadata(data['ids'],meta,role)
    for k,shape in FEATURE_SHAPES.items():
        if data[k].shape!=(n,*shape) or data[k].dtype!=np.float32 or not np.isfinite(data[k]).all():
            raise ValueError('Finite same-pass FP32 F0 features, four contributions and six logits required')
    for value in data.values():value.setflags(write=False)
    return dict(ids=data['ids'],metadata=meta,manifest=manifest,**{k:data[k] for k in FEATURE_SHAPES})


def main(argv=None):
    print(json.dumps(dict(event='STAR_PROTECTED_CACHE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    build(_read(a.config),a.device)


if __name__=='__main__':main()
