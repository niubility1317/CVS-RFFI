"""Exact source-checkpoint serialization check and synthetic CPU timing."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import runner as training,f_guided
from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values

RUN='20261030-diagnostic-DURA-host-array-s2026092701-r01'


def old_values(value):return np.asarray(value.detach().cpu().tolist(),dtype=np.float32)


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU output required')
    p=training.paths(args.root)
    clean,roles=training.source_training(p);f_guided.source_ids(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2]
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    model,payload=load_checkpoint(p['parent'],'cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=('stage2',20,4940):raise ValueError('Default frozen source-final G required')
    model.eval().requires_grad_(False)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before=[{key:v.clone() for key,v in item.state_dict().items()} for item in (model,reader)]
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    received=np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    restored=model(tensor(received[take],'cpu'));features=reader.features(restored);scores=reader.classify_features(features)
    if restored.shape!=(2,2,256) or scores.shape!=(2,6):raise ValueError('Actual source2Ls G/F shape mismatch')
    compared=0
    for value in (restored,features,scores):
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite source checkpoint output')
        actual=numpy_values(value);expected=old_values(value)
        if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)) or not actual.flags.owndata or not actual.flags.writeable:
            raise ValueError('Owned FP32 serialization differs on actual source G/F output')
        compared+=actual.size
    for item,state in zip((model,reader),before):
        if any(not torch.equal(v,state[key]) for key,v in item.state_dict().items()) or any(p.grad is not None for p in item.parameters()):
            raise ValueError('Source IO smoke changed frozen model state')
    print(json.dumps(dict(event='REAL_SOURCE_HOST_ARRAY_SMOKE_PASS',real_Ls=2,compared_scalars=compared,TX_labels_read=0,target_reads=0)),flush=True)
    generator=torch.Generator().manual_seed(2026103006);results=[]
    started=time.perf_counter()
    for shape in ((128,2,256),(128,6),(512,2,256)):
        value=torch.randn(shape,generator=generator);samples={'old':[],'DLPack_owned':[]}
        for repeat in range(9):
            order=(('old',old_values),('DLPack_owned',numpy_values))
            if repeat%2:order=tuple(reversed(order))
            for name,fn in order:
                begin=time.perf_counter()
                for _ in range(20):output=fn(value)
                samples[name].append((time.perf_counter()-begin)/20*1000)
            if not np.array_equal(numpy_values(value),old_values(value)):raise ValueError('Synthetic serialization differs')
        results.append(dict(shape=list(shape),rounds=9,calls_per_round=20,
            old_median_ms=statistics.median(samples['old']),new_median_ms=statistics.median(samples['DLPack_owned']),
            per_round_ms=samples))
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',cases=results,torch=torch.__version__,numpy=np.__version__,
        source_smoke_Ls=2,actual_source_compared_scalars=compared,exact_owned_writable_FP32=True,G_unchanged=True,F_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Synthetic CPU host-copy conversion timing after actual source G/F bit-equality smoke;not GPU transfer/training/full prediction speed;no model/precision/prediction changes',
        existing_headFeature_release_unchanged=True,goal_complete=False))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
