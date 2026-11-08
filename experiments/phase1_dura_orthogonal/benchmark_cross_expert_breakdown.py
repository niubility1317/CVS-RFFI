"""Source-only GPU timing breakdown of the fixed cross-expert training step."""
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


def timed(operation):
    torch.cuda.synchronize()
    started=time.perf_counter()
    value=operation()
    torch.cuda.synchronize()
    return value,time.perf_counter()-started


def step(fields,index,model,optimizer):
    values,t_data=timed(lambda:tuple(v[index].cuda() for v in fields))
    x,anchor,direction,variance,clean,proposal=values
    output,t_forward=timed(lambda:model(x,anchor,direction,proposal,variance))
    loss,t_loss=timed(lambda:source_gate_loss(output,anchor,clean))
    if not torch.isfinite(loss):raise FloatingPointError('Finite source training loss required')
    _,t_zero=timed(lambda:optimizer.zero_grad(set_to_none=True))
    _,t_backward=timed(loss.backward)
    _,t_clip=timed(lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    _,t_update=timed(optimizer.step)
    return dict(data_seconds=t_data,forward_seconds=t_forward,loss_seconds=t_loss,zero_seconds=t_zero,
                backward_seconds=t_backward,clip_seconds=t_clip,update_seconds=t_update,loss=float(loss.detach().cpu()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    ids,*_=source_inputs(args.root)
    _,_,_,fit=load_source_training(args.root,ids,fit_size=32768)
    data,_=load_cache(args.cache,fit,32,'cuda:0',deterministic=True)
    selected=tuple(torch.cat([v[i:i+512] for i in (0,32768,65536)]) for v in data)
    del data
    direct=load_direct(args.root,ids).cuda();dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        context=extract_context(dura,selected[0][:8].cuda())
        for a,b in zip(context,selected[1:4]):torch.testing.assert_close(a,b[:8].cuda(),rtol=1e-4,atol=1e-5)
        proposal=[]
        for i in range(0,len(selected[0]),32):proposal.append(direct(selected[0][i:i+32].cuda(),validate=False).cpu())
    fields=(*selected,torch.cat(proposal));del direct,dura,proposal
    print('CROSS_BREAKDOWN_REAL_SOURCE_SMOKE_PASS',len(fields[0]),flush=True)
    torch.manual_seed(2026110830);model=CrossExpertRefiner().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    generator=torch.Generator().manual_seed(2026110831);order=torch.randperm(len(fields[0]),generator=generator)
    step(fields,order[:128],model,optimizer)
    rows=[]
    for i in range(4):
        rows.append(step(fields,order[(i+1)*128:(i+2)*128],model,optimizer))
        print('CROSS_BREAKDOWN_STEP',i+1,flush=True)
    keys=('data_seconds','forward_seconds','loss_seconds','zero_seconds','backward_seconds','clip_seconds','update_seconds')
    totals={key:sum(r[key] for r in rows) for key in keys}
    result=dict(status='CROSS_BREAKDOWN_COMPLETE',commit=args.commit,source_ids=512,views=3,batch=128,
        warmup_steps=1,measured_steps=4,rows=rows,totals=totals,gpu=torch.cuda.get_device_name(),
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,weights_saved=False,
        caveat='Synchronized per-phase timings add overhead and serialize execution; warm source-only short run on a shared GPU, not an uninstrumented throughput or final accuracy claim.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
