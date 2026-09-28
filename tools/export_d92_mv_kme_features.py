"""Export frozen, per-observation MV-KME features from received IQ only."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from cvs_native_artifacts import verify_source

ROOT=Path(__file__).resolve().parents[1]
CACHE_SCHEMA='d92_mvkme_received_blocks_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def local_core():
    # Keep the exact native cvsrffi package separate from the new local method.
    name='_d92_mvkme_local'
    if name not in sys.modules:
        package=types.ModuleType(name)
        package.__path__=[str(ROOT/'code/cvsrffi')]
        sys.modules[name]=package
    return importlib.import_module(name+'.stage2_d92_mv_kme')


def validate_capsule(capsule,expected_capsule_id):
    manifest=read(Path(capsule)/'manifest.json')
    if (manifest.get('protocol_schema')!='p2_min_v1' or manifest.get('phase2_data_status')!='VALIDATED_ONCE'
            or manifest.get('capsule_id')!=expected_capsule_id):
        raise ValueError('Unvalidated or mismatched capsule')
    return manifest


def validate_origin(row_root,capsule,checkpoint_sha256,classes=None):
    row_root=Path(row_root)
    previous=read(row_root/'d92_startup.json')
    provenance=read(row_root/'received_features/checkpoint_provenance.json')
    if (previous.get('checkpoint_sha256')!=checkpoint_sha256 or previous.get('capsule')!=str(capsule)
            or previous.get('features')!=str(row_root/'received_features/received_features.npz')
            or previous.get('truth_read') is not False or previous.get('query_fit_access') is not False
            or provenance.get('target_access_before_freeze') is not False
            or provenance.get('checkpoint_inheritance')!=[]
            or provenance.get('verdict')!='MATCHED_SOURCE_ONLY_SCRATCH'
            or provenance.get('source_role_comparison')!='EXACT_MATCH'
            or (classes is not None and provenance.get('classes')!=classes)):
        raise ValueError('Frozen baseline/checkpoint origin mismatch')
    return previous,provenance


class FrozenIdentity:
    """No training or mutable buffers; each call contains one observation's views."""
    def __init__(self,model,forward,device):
        import torch
        self.torch=torch;self.model=model;self.forward=forward;self.device=device
        model.eval();model.requires_grad_(False)
        for buffer in model.buffers():
            if buffer.is_floating_point() or buffer.is_complex():buffer.requires_grad_(False)
        self.buffers={name:value.detach().clone() for name,value in model.named_buffers()}
        self.versions={name:value._version for name,value in model.named_parameters()}
        self.verify_frozen()

    def verify_frozen(self):
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Native encoder is not frozen/eval')
        if any(value._version!=self.versions[name] for name,value in self.model.named_parameters()):
            raise ValueError('Native parameter changed during received forward')
        if any(not self.torch.equal(value,self.buffers[name]) for name,value in self.model.named_buffers()):
            raise ValueError('Native buffer changed during received forward')

    def __call__(self,views):
        torch=self.torch
        with torch.inference_mode():
            result=self.forward(self.model,torch.as_tensor(np.array(views,copy=True),dtype=torch.float32,device=self.device),'z_id')
            if result is None:raise ValueError('Native identity-only forward unavailable')
            identity=result[0]
            if identity.shape!=(len(views),160) or not torch.isfinite(identity).all():
                raise ValueError('Invalid native identity output')
            return identity.detach().cpu().numpy().astype(np.float64)


def load_native(*,native_code,source,contract,seed,expected_checkpoint_sha256,device):
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
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args=payload['args']
    if (payload['epoch']!=200 or args['seed']!=seed or args['baseline_ckpt'] or args['teacher_ckpt']
            or not args['from_scratch'] or len(actual['classes'])!=actual['num_classes']):
        raise ValueError('Wrong final checkpoint or inheritance')
    device=torch.device(device)
    model,audit=checkpoint_loading.build_exact_ssdg_model_from_checkpoint(payload,input_len=256,device=device)
    infer=FrozenIdentity(model,identity_only_forward.identity_only_feature_forward,device)
    smoke_started=time.perf_counter()
    smoke_views=local_core().make_received_views(np.zeros((1,2,256),dtype=np.float32)).reshape(16,2,256)
    smoke_identity=infer(smoke_views);infer.verify_frozen()
    print(json.dumps(dict(event='NATIVE_ZERO_IQ_SMOKE',status='PASS',query_read=False,
        view_shape=list(smoke_views.shape),identity_shape=list(smoke_identity.shape),device=str(device),
        model_parameter_count=sum(value.numel() for value in model.parameters()),loader=audit,
        elapsed_seconds=time.perf_counter()-smoke_started,all_parameters_frozen=True,buffers_unchanged=True)),flush=True)
    return infer,dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',checkpoint=str(checkpoint),checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,source_role_comparison='EXACT_MATCH',
        model_seed=seed,classes=actual['classes'],model_file_bytes=checkpoint.stat().st_size,loader=audit,
        source_arguments=cfg,native_code=str(native_code),query_read_for_smoke=False)


def transform_received(iq,infer,core):
    iq=np.asarray(iq)
    if iq.ndim!=3 or iq.shape[1:]!=(2,256) or not len(iq) or not np.isfinite(iq).all():
        raise ValueError('Expected finite received IQ (N,2,256)')
    blocks=np.empty((len(iq),3,256),dtype=np.float32)
    started=time.perf_counter()
    for index in range(len(iq)):
        sample=iq[index:index+1]
        views=core.make_received_views(sample)
        if views.shape!=(1,4,4,2,256):raise ValueError('Invalid frozen view shape')
        identity=infer(views.reshape(16,2,256)).reshape(1,4,4,160)
        value=core.build_fourier_blocks(received_iq=sample,identity_views=identity)
        if value.shape!=(1,3,256) or not np.isfinite(value).all():raise ValueError('Invalid Fourier blocks')
        blocks[index]=value[0]
        if (index+1)%128==0 or index+1==len(iq):
            print(json.dumps(dict(event='FEATURE_PROGRESS',completed=index+1,total=len(iq),views=(index+1)*16,
                extraction_elapsed_seconds=time.perf_counter()-started)),flush=True)
    return blocks


def export(*,row_root,source,contract,native_code,seed,capsule,output,config,
           expected_capsule_id,expected_checkpoint_sha256,device='cpu'):
    row_root,capsule,out=map(Path,(row_root,capsule,output))
    if out.exists():raise FileExistsError(out)
    core=local_core()
    if set(config)!={'algorithm'} or config['algorithm']!=core.FROZEN_CONFIG:raise ValueError('Unregistered MV-KME configuration')
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
    started=time.perf_counter();blocks=transform_received(iq,infer,core);infer.verify_frozen()
    feature_path=out/'received_mv_features.npz'
    np.savez(feature_path,blocks=blocks,ids=ids,checkpoint_sha256=np.asarray(expected_checkpoint_sha256),
        capsule_id=np.asarray(expected_capsule_id),algorithm_json=np.asarray(json.dumps(config['algorithm'],sort_keys=True)))
    write(out/'checkpoint_provenance.json',provenance)
    marker=dict(status='MULTIVIEW_FEATURES_COMPLETE',schema=CACHE_SCHEMA,capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256,algorithm=config['algorithm'],count=len(ids),shape=list(blocks.shape),
        dtype=str(blocks.dtype),classes=provenance['classes'],model_seed=seed,query_used_for_fitting=False,
        source_data_access=False,truth_read=False,encoder_updated=False,native_eval=True,native_buffers_unchanged=True,
        view_count_per_observation=16,native_forward_scope='One observation, 16 fixed views per call; no cross-query statistics',
        feature_seconds=time.perf_counter()-started,feature_array_bytes=blocks.nbytes,feature_file_bytes=feature_path.stat().st_size,
        model_file_bytes=provenance['model_file_bytes'],new_ground_statistics_bytes=0)
    write(out/'features_complete.json',marker)
    print(json.dumps(marker,allow_nan=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('row-root','source','contract','native-code','capsule','output','config','expected-capsule-id','expected-checkpoint-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--device',default='cpu')
    args=vars(parser.parse_args());args['config']=read(args['config']);export(**args)


if __name__=='__main__':main()
