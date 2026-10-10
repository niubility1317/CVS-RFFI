"""Frozen clean IQ inference; this process never receives a truth path."""
import argparse
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import numpy as np
import torch
from experiments.standard_ce_baselines.common import read,write,VIEWS
from experiments.standard_ce_baselines.model import build,VARIANTS


def check(c,done,initial,contract,expected,resolved,payload,freeze):
    if c['variant'] not in VARIANTS or any(c.get(k) for k in ('truth','p1_truth','target_truth')):raise ValueError('Invalid inference config')
    if done.get('status')!='SOURCE_TRAINED' or done.get('epoch')!=200 or done.get('steps')!=10000 or done.get('target_access') is not False:raise ValueError('Source incomplete')
    if initial.get('scratch_only') is not True or initial.get('checkpoint') is not None or initial.get('ancestors')!=[] or initial.get('checkpoint_sources')!=[] or initial.get('target_access') is not False or initial.get('model_seed')!=c['model_seed']:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if any(contract.get(k)!=v for k,v in expected.items()) or contract.get('out_len')!=256 or contract.get('equalized')!=1 or contract.get('normalize') is not True:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    from experiments.standard_ce_baselines.source import validate_config
    validate_config(resolved)
    if resolved['variant']!=c['variant'] or resolved['model_seed']!=c['model_seed'] or resolved['source_counts']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Resolved identity mismatch')
    if any(payload.get(k)!=v for k,v in dict(method='standard_ce_baselines',variant=c['variant'],epoch=200,selection='fixed_last_epoch',config=resolved,initialization=initial,source_contract=contract).items()):raise ValueError('Checkpoint payload mismatch')
    if freeze!=dict(status='FROZEN',epoch=200,variant=c['variant'],model_seed=c['model_seed'],checkpoint=str(Path(c['source_output'])/'last.pt'),target_access=False,target_score_used=False):raise ValueError('Freeze mismatch')


def predict(c):
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    source=Path(c['source_output']);device=torch.device('cuda:0');torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    done,initial,contract,resolved,freeze=[read(source/f) for f in ('completion.json','initialization.json','source_contract.json','resolved_config.json','freeze.json')]
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    check(c,done,initial,contract,read(c['source_contract']),resolved,payload,freeze)
    model=build(c['variant']);model.load_state_dict(payload['model'],strict=True);del payload
    model.to(device).eval()
    with torch.no_grad():scores=model(torch.zeros(2,2,256,device=device))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Frozen checkpoint no-query smoke failed')
    capsule=Path(c['capsule']);manifest=read(capsule/'manifest.json')
    if manifest.get('status')!='VALIDATED_ONCE' or manifest.get('count')!=168000 or manifest.get('classes')!=contract['classes'] or manifest.get('source_target_disjoint') is not True or manifest.get('channel')!='residual/post_sync/noeq':raise ValueError('Clean capsule mismatch')
    with np.load(capsule/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
    array=np.load(capsule/'clean.npy',mmap_mode='r',allow_pickle=False)
    if len(ids)!=168000 or len(set(ids.tolist()))!=len(ids) or array.shape!=(len(ids),2,256):raise ValueError('Query coverage mismatch')
    write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=str(ROOT),python=sys.executable,
        torch_version=torch.__version__,hardware=torch.cuda.get_device_name(device),batch_size=128,truth_read=False,query_fit=False,
        total_parameters=sum(p.numel() for p in model.parameters()),commit=(ROOT/'release_commit.txt').read_text().strip()))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.perf_counter();pred=[]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            data=np.ascontiguousarray(array[start:start+128],dtype=np.float32)
            x=torch.frombuffer(bytearray(data.tobytes()),dtype=torch.float32).reshape(data.shape).to(device)
            logits=model(x)
            if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid prediction')
            pred.extend(logits.argmax(1).cpu().tolist())
    torch.cuda.synchronize()
    np.savez(out/'predictions.npz',ids=ids,clean=np.asarray(pred,dtype=np.int64))
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),truth_read=False,query_fit=False,
        inference_seconds=time.perf_counter()-tic,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    print('CLEAN_PREDICTIONS_COMPLETE count='+str(len(ids)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();predict(read(a.config))
