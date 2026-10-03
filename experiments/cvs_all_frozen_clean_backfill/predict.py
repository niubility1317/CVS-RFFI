"""Frozen per-packet clean predictions; no satellite file or truth input."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import numpy as np
import torch
from experiments.cvs_clean_eval.contracts import checkpoint_contract,build_model
from experiments.cvs_all_frozen_clean_backfill.contracts import read,validate_config


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)


def predict(c):
    import torch
    original=torch.backends.cudnn.allow_tf32
    try:
        return _predict(c)
    finally:
        torch.backends.cudnn.allow_tf32=original


def _predict(c):
    started=time.perf_counter()
    validate_config(c)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    source=Path(c['source_output']);device=torch.device(c.get('device','cuda:0'));torch.set_num_threads(2)
    done,initial=read(source/'completion.json'),read(source/'initialization.json')
    contract,expected=read(source/'source_contract.json'),read(c['source_contract'])
    resolved=read(source/'resolved_config.json')
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    checkpoint_contract(c,done,initial,contract,expected,resolved,payload)
    if resolved['commit']!=c['source_release_commit']:raise ValueError('Actual frozen source commit differs')
    if 'numerical_policy' in resolved:
        from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY,actual_flags
        if resolved['numerical_policy']!=FULL_FP32_POLICY:raise ValueError('Unregistered frozen numerical policy')
        torch.backends.cudnn.allow_tf32=False
        if actual_flags()!=resolved['numerical_policy']:raise ValueError('Frozen prediction numerical policy mismatch')
    model=build_model(c['variant']);model.load_state_dict(payload['model'],strict=True);model.to(device);model.eval();model.requires_grad_(False)
    with torch.no_grad():scores=model(torch.zeros(2,2,256,device=device))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Frozen checkpoint smoke failed')
    write(out/'provenance.json',dict(status='VERIFIED',checkpoint=str(source/'last.pt'),source_contract_roles='EXACT_MATCH',
        initialization='SCRATCH',ancestors=[],selection='E200',architecture_selection=c['authorization'],query_fit=False))
    capsule=Path(c['p1_capsule']);manifest=read(capsule/'manifest.json')
    if manifest['status']!='VALIDATED_ONCE' or manifest['classes']!=contract['classes'] or manifest['channel']!='residual/post_sync/noeq':
        raise ValueError('Clean capsule class/channel contract mismatch')
    with np.load(capsule/'index.npz',allow_pickle=False) as index:ids=index['ids'].copy()
    if not len(ids) or len(ids)!=len(set(ids.tolist())):raise ValueError('Empty/duplicate physical query IDs')
    array=np.load(capsule/'clean.npy',mmap_mode='r',allow_pickle=False)
    if array.shape!=(len(ids),2,256):raise ValueError('Clean input shape mismatch')
    prediction_resolved=dict(c,pid=os.getpid(),cwd=str(ROOT),python=sys.executable,
        torch_version=torch.__version__,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        source_resolved_ref=str(source/'resolved_config.json'),truth_read=False,query_fit=False,views=['clean'],batch_size=256,
        commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL')
    if 'numerical_policy' in resolved:prediction_resolved.update(numerical_policy=resolved['numerical_policy'],backend_flags=actual_flags())
    write(out/'resolved_config.json',prediction_resolved)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    tic=time.perf_counter();pred=[]
    with torch.no_grad():
        for start in range(0,len(ids),256):
            # N607 Torch2.1/NumPy2 C ABI bridge is unavailable; use an explicit copy.
            x=torch.tensor(np.asarray(array[start:start+256]).tolist(),device=device,dtype=torch.float32)
            logits=model(x)
            if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid frozen clean scores')
            pred.extend(logits.argmax(1).cpu().tolist())
    if device.type=='cuda':torch.cuda.synchronize(device)
    np.savez(out/'clean_predictions.npz',ids=ids,clean=np.asarray(pred,dtype=np.int64))
    write(out/'clean_complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=['clean'],truth_read=False,query_fit=False,
        prediction_seconds=time.perf_counter()-tic,total_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
    print('CLEAN_PREDICTIONS_COMPLETE '+json.dumps(dict(count=len(ids),variant=c['variant'],model_seed=c['model_seed'],truth_read=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();predict(read(a.config))
