"""Source-only bytewise-checked GPU timing of cross-expert leaf modules."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from torch import nn
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.shared_source_benchmark import same_bits
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.train_source import extract_context, load_source_training


def profiled_forward(model, inputs):
    rows=[];starts={};handles=[]
    def before(name):
        def hook(module, args):
            torch.cuda.synchronize();starts[name]=time.perf_counter()
        return hook
    def after(name):
        def hook(module, args, output):
            torch.cuda.synchronize();rows.append(dict(name=name,kind=type(module).__name__,seconds=time.perf_counter()-starts.pop(name)))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv1d,nn.GroupNorm,nn.GELU)):
            handles.append(module.register_forward_pre_hook(before(name)))
            handles.append(module.register_forward_hook(after(name)))
    try:
        torch.cuda.synchronize();started=time.perf_counter();output=model(*inputs);torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
    finally:
        for handle in handles:handle.remove()
    if starts:raise RuntimeError('Each measured layer must finish')
    return output,rows,elapsed


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    ids,*_=source_inputs(args.root);_,_,_,fit=load_source_training(args.root,ids,fit_size=32768)
    data,_=load_cache(args.cache,fit,32,'cuda:0',deterministic=True)
    fields=tuple(v[:128].cuda() for v in data);del data
    received,anchor,innovation,variance,clean=fields
    direct=load_direct(args.root,ids).cuda();dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        context=extract_context(dura,received[:8])
        for a,b in zip(context,(anchor,innovation,variance)):torch.testing.assert_close(a,b[:8],rtol=1e-4,atol=1e-5)
        proposal=torch.cat([direct(received[i:i+32],validate=False) for i in range(0,128,32)])
    del direct,dura,clean
    torch.manual_seed(2026110830);model=CrossExpertRefiner().cuda().train()
    with torch.no_grad():
        for head in (model.time_out,model.freq_out):
            nn.init.normal_(head.weight,mean=0.,std=.001);nn.init.normal_(head.bias,mean=0.,std=.001)
    inputs=(received,anchor,innovation,proposal,variance)
    print('CROSS_LAYER_REAL_SOURCE_SMOKE_PASS',8,flush=True)
    warm=model(*inputs);torch.cuda.synchronize();del warm
    reference=model(*inputs).detach();torch.cuda.synchronize()
    trials=[]
    for i in range(2):
        output,rows,elapsed=profiled_forward(model,inputs)
        if not torch.isfinite(output).all().item() or not same_bits(output.detach(),reference):
            raise ValueError('Instrumented original forward must match bytewise')
        totals={kind:sum(r['seconds'] for r in rows if r['kind']==kind) for kind in ('Conv1d','GroupNorm','GELU')}
        trials.append(dict(total_seconds=elapsed,layer_totals_seconds=totals,layers=rows,output_bitwise_equal=True))
        del output
        print('CROSS_LAYER_TIMING',i+1,flush=True)
    result=dict(status='CROSS_LAYER_TIMING_COMPLETE',commit=args.commit,batch=128,measured_forwards=2,
        mode='train_autograd_enabled',nonzero_head_std=.001,trials=trials,gpu=torch.cuda.get_device_name(),
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,weights_saved=False,
        caveat='Every measured module synchronizes CUDA; hooks change timing and shared GPU affects results. Bytewise output equality covers only the fixed source batch and diagnostic model.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
