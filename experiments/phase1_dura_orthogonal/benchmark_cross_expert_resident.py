"""Measure source training steps with CPU versus GPU-resident fixed inputs."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.signal_confidence_gate import source_gate_loss
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.train_source import extract_context, load_source_training


def tree_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:tree_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [tree_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(tree_cpu(v) for v in value)
    return value


def tree_equal(left,right):
    if type(left)!=type(right):return False
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(tree_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(tree_equal(a,b) for a,b in zip(left,right))
    return left==right


def train(fields,resident,steps):
    torch.manual_seed(2026110830)
    model=CrossExpertRefiner().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    generator=torch.Generator().manual_seed(2026110831)
    torch.cuda.synchronize();cache_start=time.monotonic()
    data=tuple(x.cuda() for x in fields) if resident else fields
    torch.cuda.synchronize();cache_seconds=time.monotonic()-cache_start
    losses=[]
    torch.cuda.synchronize();started=time.monotonic()
    order=torch.randperm(len(fields[0]),generator=generator)
    for step in range(steps):
        idx=order[step*128:(step+1)*128]
        if resident:idx=idx.cuda()
        x,anchor,direction,variance,clean,proposal=(v[idx] if resident else v[idx].cuda() for v in data)
        output=model(x,anchor,direction,proposal,variance)
        loss=source_gate_loss(output,anchor,clean)
        if not torch.isfinite(loss):raise FloatingPointError('Finite source loss required')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.detach())
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    snapshot=dict(model=tree_cpu(model.state_dict()),optimizer=tree_cpu(optimizer.state_dict()),
        losses=tree_cpu(torch.stack(losses)),order_rng=generator.get_state(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    return snapshot,elapsed,cache_seconds


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pilot,*_=source_inputs(args.root)
    _,_,_,fit=load_source_training(args.root,pilot,fit_size=32768)
    data,_=load_cache(args.cache,fit,32,'cuda:0',deterministic=True)
    selected=tuple(torch.cat([v[i:i+512] for i in (0,32768,65536)]) for v in data)
    del data
    direct=load_direct(args.root,pilot).cuda()
    dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        context=extract_context(dura,selected[0][:8].cuda())
        for actual,expected in zip(context,selected[1:4]):
            torch.testing.assert_close(actual,expected[:8].cuda(),rtol=1e-4,atol=1e-5)
        proposal=[]
        for i in range(0,len(selected[0]),32):proposal.append(direct(selected[0][i:i+32].cuda(),validate=False).cpu())
    fields=(*selected,torch.cat(proposal))
    del direct,dura,proposal
    print('CROSS_CACHE_REAL_SOURCE_SMOKE_PASS',len(fields[0]),flush=True)
    for resident in (False,True):train(fields,resident,1)
    pairs=[]
    for order in ((False,True),(True,False)):
        results={resident:train(fields,resident,4) for resident in order}
        baseline,fast=results[False],results[True]
        match=tree_equal(baseline[0],fast[0])
        pairs.append(dict(resident_first=order[0],baseline_seconds=baseline[1],resident_seconds=fast[1],
            cache_transfer_seconds=fast[2],steps=4,losses_model_optimizer_and_rng_exact=match))
        print('CROSS_CACHE_PAIR',order[0],'exact',match,'baseline',baseline[1],'resident',fast[1],flush=True)
        if not match:break
    size=sum(v.numel()*v.element_size() for v in fields)
    result=dict(status='CROSS_CACHE_BENCHMARK_COMPLETE' if len(pairs)==2 and all(x['losses_model_optimizer_and_rng_exact'] for x in pairs) else 'CROSS_CACHE_BENCHMARK_MISMATCH',
        commit=args.commit,source_ids=512,views=3,records=len(fields[0]),batch=128,pairs=pairs,
        selected_cache_bytes=size,estimated_full_52896_cache_bytes=int(size/len(fields[0])*52896),
        gpu=torch.cuda.get_device_name(),TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,weights_saved=False,
        caveat='Short paired source-only training steps; cache load/direct preparation excluded from timers, transfer reported separately; not a full-training or target-equivalence claim.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
