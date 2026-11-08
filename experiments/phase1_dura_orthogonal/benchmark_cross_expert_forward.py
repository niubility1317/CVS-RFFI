"""Source-only, bitwise-checked GPU forward-phase timing for cross-expert model."""
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
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.train_source import extract_context, load_source_training


def timed(operation):
    torch.cuda.synchronize();start=time.perf_counter();value=operation();torch.cuda.synchronize()
    return value,time.perf_counter()-start


def forward_timed(model, received, anchor, innovation, proposal, variance):
    def inputs():
        power=anchor.square().mean((1,2),keepdim=True).clamp_min(1e-12)
        scale=power.sqrt();delta=proposal-anchor
        noise=torch.log1p(variance[:,None,None].clamp_min(0)/power).expand(-1,1,256)
        signals=(received/scale,anchor/scale,innovation/scale,delta/scale)
        return scale,delta,signals,torch.cat((*signals,noise),dim=1)
    (scale,delta,signals,time_input),t_input=timed(inputs)
    def fft():
        spectrum=[torch.fft.fft(torch.complex(z[:,0],z[:,1]),norm='ortho') for z in signals]
        return torch.cat([torch.stack((z.real,z.imag),dim=1) for z in spectrum],dim=1)
    freq_input,t_fft=timed(fft)
    time_delta,t_time=timed(lambda:model.time_out(model.time_blocks(model.time_in(time_input))))
    frequency_delta,t_freq=timed(lambda:model.freq_out(model.freq_blocks(model.freq_in(freq_input))))
    def finish():
        inverse=torch.fft.ifft(torch.complex(frequency_delta[:,0],frequency_delta[:,1]),norm='ortho')
        spectral_delta=torch.stack((inverse.real,inverse.imag),dim=1)
        return anchor+.5*delta+scale*(time_delta+spectral_delta)
    out,t_finish=timed(finish)
    return out,dict(input_seconds=t_input,fft_seconds=t_fft,time_branch_seconds=t_time,
                    frequency_branch_seconds=t_freq,ifft_finish_seconds=t_finish)


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
    for p in model.parameters():p.requires_grad_(True)
    args_i=(received,anchor,innovation,proposal,variance)
    print('CROSS_FORWARD_REAL_SOURCE_SMOKE_PASS',8,flush=True)
    warm=model(*args_i);torch.cuda.synchronize();del warm
    reference=model(*args_i).detach();torch.cuda.synchronize()
    rows=[]
    for i in range(2):
        out,timing=forward_timed(model,*args_i)
        equal=torch.equal(out.detach(),reference)
        if not equal:raise ValueError('Timed forward must match original bitwise')
        rows.append(dict(**timing,output_bitwise_equal=equal));del out
        print('CROSS_FORWARD_TIMING',i+1,flush=True)
    result=dict(status='CROSS_FORWARD_TIMING_COMPLETE',commit=args.commit,batch=128,measured_forwards=2,
        mode='train_autograd_enabled',rows=rows,gpu=torch.cuda.get_device_name(),TX_labels_read=0,
        F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,weights_saved=False,
        caveat='Each phase synchronizes CUDA; shared GPU affects timing. Reimplemented forward is checked bitwise on this batch, not a claim for all inputs or full training.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
