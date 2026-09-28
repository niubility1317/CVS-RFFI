"""Frozen four-phase identity and FFT cache from each received IQ observation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from cvs_native_artifacts import verify_source
from export_d92_mv_kme_features import (
    FrozenIdentity,read,write,sha256,peak_process_rss,validate_capsule,validate_origin,
)

ROOT=Path(__file__).resolve().parents[1]
CACHE_SCHEMA='d92_bnna_received_views_v1'
CACHE_NAME='received_bnna_features.npz'


def local_core():
    name='_d92_bnna_local'
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(ROOT/'code/cvsrffi')]
        sys.modules[name]=package
    return importlib.import_module(name+'.stage2_d92_bnna')


def synthetic_pipeline_smoke(infer,core):
    """Exercise the actual scalar bridge and full fit without any capsule data."""
    iq=np.random.Generator(np.random.PCG64(0)).normal(size=(8,2,256)).astype(np.float32)
    views=core.make_received_views(iq)
    if views.shape!=(8,4,2,256):raise ValueError('BNNA smoke view shape mismatch')
    identity=np.stack([infer(row) for row in views]).astype(np.float32)
    fft=np.asarray(core.received_fft96(iq),dtype=np.float32)
    if (identity.shape!=(8,4,160) or fft.shape!=(8,96)
            or not np.isfinite(identity).all() or not np.isfinite(fft).all()):
        raise ValueError('Invalid BNNA synthetic features')
    state=core.fit_bnna(support_identity_views=identity[:6],support_fft=fft[:6],
        support_labels=np.repeat(np.arange(2),3),support_ids=[f'synthetic-smoke-{i}' for i in range(6)],
        classes=['synthetic-old','synthetic-new'],old_classes=['synthetic-old'])
    scores=state.score(identity[6:],fft[6:])
    singleton=np.concatenate([state.score(identity[i:i+1],fft[i:i+1]) for i in (6,7)])
    if scores.shape!=(2,2) or not np.isfinite(scores).all() or not np.array_equal(scores,singleton):
        raise ValueError('BNNA synthetic score/singleton mismatch')
    infer.verify_frozen();audit=state.audit_dict()
    return dict(view_shape=list(views.shape),identity_shape=list(identity.shape),fft_shape=list(fft.shape),
        identity_dtype=str(identity.dtype),physical_support_count=6,synthetic_probe_count=2,k=3,
        candidate_count=audit['candidate_count'],fold_count=audit['fold_count'],
        optimizer_steps=audit['optimizer_steps'],finite_features=True,finite_scores=True,singleton_exact=True)


def load_native(*,native_code,source,contract,seed,expected_checkpoint_sha256,device):
    """Use the audited exact loader and scalar-list bridge, with a BNNA smoke."""
    native_code=Path(native_code).resolve();source=Path(source)
    actual,cfg=verify_source(source,read(contract),seed)
    checkpoint=source/'final_ssdg.pth'
    if sha256(checkpoint)!=expected_checkpoint_sha256:raise ValueError('Checkpoint SHA mismatch')
    sys.path.insert(0,str(native_code))
    import torch
    from cvsrffi import checkpoint_loading,identity_only_forward
    for module in (checkpoint_loading,identity_only_forward):
        if not Path(module.__file__).resolve().is_relative_to(native_code):
            raise ValueError('Exact native code import was shadowed')
    torch.set_num_threads(2)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);args=payload['args']
    if (payload['epoch']!=200 or args['seed']!=seed or args['baseline_ckpt'] or args['teacher_ckpt']
            or not args['from_scratch'] or len(actual['classes'])!=actual['num_classes']):
        raise ValueError('Wrong final checkpoint or inheritance')
    device=torch.device(device)
    model,audit=checkpoint_loading.build_exact_ssdg_model_from_checkpoint(payload,input_len=256,device=device)
    infer=FrozenIdentity(model,identity_only_forward.identity_only_feature_forward,device)
    started=time.perf_counter();smoke=synthetic_pipeline_smoke(infer,local_core())
    print(json.dumps(dict(event='NATIVE_SYNTHETIC_PIPELINE_SMOKE',status='PASS',query_read=False,device=str(device),
        **smoke,tensor_numpy_boundary='python_scalar_lists',
        torch_version=torch.__version__,numpy_version=np.__version__,
        model_parameter_count=sum(value.numel() for value in model.parameters()),loader=audit,
        elapsed_seconds=time.perf_counter()-started,all_parameters_frozen=True,buffers_unchanged=True)),flush=True)
    return infer,dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',checkpoint=str(checkpoint),checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,source_role_comparison='EXACT_MATCH',
        model_seed=seed,classes=actual['classes'],model_file_bytes=checkpoint.stat().st_size,loader=audit,
        source_arguments=cfg,native_code=str(native_code),query_read_for_smoke=False)


def transform_received(iq,infer,core):
    iq=np.asarray(iq)
    if iq.ndim!=3 or iq.shape[1:]!=(2,256) or not len(iq) or not np.isfinite(iq).all():
        raise ValueError('Expected finite received IQ (N,2,256)')
    identity=np.empty((len(iq),4,160),dtype=np.float32)
    fft=np.empty((len(iq),96),dtype=np.float32);started=time.perf_counter()
    for index in range(len(iq)):
        sample=iq[index:index+1];views=core.make_received_views(sample)
        if views.shape!=(1,4,2,256):raise ValueError('Invalid BNNA view shape')
        current_identity=infer(views.reshape(4,2,256));current_fft=core.received_fft96(sample)
        if (current_identity.shape!=(4,160) or current_fft.shape!=(1,96)
                or not np.isfinite(current_identity).all() or not np.isfinite(current_fft).all()):
            raise ValueError('Invalid BNNA received features')
        identity[index]=current_identity;fft[index]=current_fft[0]
        if (index+1)%128==0 or index+1==len(iq):
            print(json.dumps(dict(event='FEATURE_PROGRESS',completed=index+1,total=len(iq),views=(index+1)*4,
                extraction_elapsed_seconds=time.perf_counter()-started)),flush=True)
    return identity,fft


def export(*,row_root,source,contract,native_code,seed,capsule,output,config,
           expected_capsule_id,expected_checkpoint_sha256,device='cpu'):
    row_root,capsule,out=map(Path,(row_root,capsule,output))
    if out.exists():raise FileExistsError(out)
    core=local_core()
    if set(config)!={'algorithm'} or config['algorithm']!=core.FROZEN_CONFIG:raise ValueError('Unregistered BNNA configuration')
    validate_capsule(capsule,expected_capsule_id)
    previous,origin=validate_origin(row_root,capsule,expected_checkpoint_sha256)
    if previous['seed']!=seed:raise ValueError('Frozen model seed mismatch')
    infer,provenance=load_native(native_code=native_code,source=source,contract=contract,seed=seed,
        expected_checkpoint_sha256=expected_checkpoint_sha256,device=device)
    if provenance['classes']!=origin['classes']:raise ValueError('Frozen native classes mismatch')
    with np.load(capsule/'received.npz',allow_pickle=False) as data:
        if set(data.files)!={'iq','ids'}:raise ValueError('Unexpected received data members')
        iq,ids=data['iq'],data['ids'].astype(str)
    if ids.shape!=(len(iq),) or len(set(ids))!=len(ids):raise ValueError('Invalid physical IDs')
    if iq.ndim!=3 or iq.shape[1:]!=(2,256) or not len(iq) or not np.isfinite(iq).all():
        raise ValueError('Expected finite received IQ (N,2,256)')
    out.mkdir(parents=True,exist_ok=False)
    startup=dict(argv=sys.argv,python=sys.executable,pid=os.getpid(),config=config,
        capsule_id=expected_capsule_id,checkpoint_sha256=expected_checkpoint_sha256,device=device,
        query_fit_access=False,source_data_access=False,truth_read=False,provenance=provenance)
    write(out/'startup.json',startup);print(json.dumps(dict(event='STARTUP',**startup)),flush=True)
    started=time.perf_counter();identity,fft=transform_received(iq,infer,core);infer.verify_frozen()
    feature_path=out/CACHE_NAME
    np.savez(feature_path,identity_views=identity,fft=fft,ids=ids,checkpoint_sha256=np.asarray(expected_checkpoint_sha256),
        capsule_id=np.asarray(expected_capsule_id),algorithm_json=np.asarray(json.dumps(config['algorithm'],sort_keys=True)))
    write(out/'checkpoint_provenance.json',provenance)
    marker=dict(status='BNNA_FEATURES_COMPLETE',schema=CACHE_SCHEMA,capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256,algorithm=config['algorithm'],count=len(ids),
        identity_views_shape=list(identity.shape),fft_shape=list(fft.shape),dtype='float32',
        classes=provenance['classes'],model_seed=seed,query_used_for_fitting=False,source_data_access=False,truth_read=False,
        encoder_updated=False,native_eval=True,native_buffers_unchanged=True,view_count_per_observation=4,
        native_forward_scope='One observation, four fixed full-length phase views per call; no cross-query statistics',
        feature_seconds=time.perf_counter()-started,identity_views_bytes=identity.nbytes,fft_bytes=fft.nbytes,
        feature_array_bytes=identity.nbytes+fft.nbytes,feature_file_bytes=feature_path.stat().st_size,
        model_file_bytes=provenance['model_file_bytes'],new_ground_statistics_bytes=0,peak_process_rss_bytes=peak_process_rss(),
        peak_process_rss_reason='Linux ru_maxrss process high-water memory; null when unavailable on this platform')
    write(out/'features_complete.json',marker);print(json.dumps(marker,allow_nan=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('row-root','source','contract','native-code','capsule','output','config','expected-capsule-id','expected-checkpoint-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--device',default='cpu')
    args=vars(parser.parse_args());args['config']=read(args['config']);export(**args)


if __name__=='__main__':main()
