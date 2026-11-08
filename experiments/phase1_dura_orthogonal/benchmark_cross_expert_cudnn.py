"""Source-only exactness and speed test of deterministic cuDNN autotuning."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.benchmark_cross_expert_resident import train, tree_equal
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.train_source import extract_context, load_source_training


def run_condition(fields, autotune, steps):
    torch.backends.cudnn.benchmark=autotune
    return train(fields,False,steps)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    ids,*_=source_inputs(args.root)
    _,_,_,fit=load_source_training(args.root,ids,fit_size=32768)
    data,_=load_cache(args.cache,fit,32,'cuda:0',deterministic=True)
    selected=tuple(torch.cat([v[i:i+512] for i in (0,32768,65536)]) for v in data)
    del data
    direct=load_direct(args.root,ids).cuda();dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        context=extract_context(dura,selected[0][:8].cuda())
        for actual,expected in zip(context,selected[1:4]):torch.testing.assert_close(actual,expected[:8].cuda(),rtol=1e-4,atol=1e-5)
        proposal=[]
        for i in range(0,len(selected[0]),32):proposal.append(direct(selected[0][i:i+32].cuda(),validate=False).cpu())
    fields=(*selected,torch.cat(proposal));del direct,dura,proposal
    print('CROSS_CUDNN_REAL_SOURCE_SMOKE_PASS',len(fields[0]),flush=True)
    for mode in (False,True):run_condition(fields,mode,1)
    pairs=[]
    for first in (False,True):
        result={mode:run_condition(fields,mode,4) for mode in (first,not first)}
        baseline,fast=result[False],result[True]
        equal=tree_equal(baseline[0],fast[0])
        item=dict(autotune_first=first,baseline_seconds=baseline[1],autotune_seconds=fast[1],steps=4,
                  losses_model_optimizer_and_rng_exact=equal)
        pairs.append(item)
        print('CROSS_CUDNN_PAIR',first,'exact',equal,'baseline',baseline[1],'autotune',fast[1],flush=True)
    result=dict(status='CROSS_CUDNN_BENCHMARK_COMPLETE',commit=args.commit,records=len(fields[0]),
        source_ids=512,views=3,batch=128,pairs=pairs,gpu=torch.cuda.get_device_name(),
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,weights_saved=False,
        caveat='Short real-source paired CPU-staged training steps; deterministic FP32 TF32 off. Warmed algorithm selection; input preparation excluded; not full training or final output parity. Shared GPU conditions can affect timing.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
