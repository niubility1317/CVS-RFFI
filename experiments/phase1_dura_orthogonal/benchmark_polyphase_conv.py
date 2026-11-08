"""Source-smoke GPU timing and numerical parity for polyphase Conv1d."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.polyphase_conv import polyphase_conv1d
from experiments.phase1_dura_orthogonal.shared_source_benchmark import same_bits

DILATIONS=(2,4,8,16,32)


def run(mode, dilation, base_x, base_w, base_b, grad):
    x=base_x.detach().clone().requires_grad_(True)
    weight=base_w.detach().clone().requires_grad_(True)
    bias=base_b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize();start=time.perf_counter()
    if mode=='native':out=F.conv1d(x,weight,bias,padding=2*dilation,dilation=dilation)
    elif mode=='polyphase':out=polyphase_conv1d(x,weight,bias,dilation)
    else:raise ValueError('Fixed benchmark mode required')
    torch.cuda.synchronize();forward=time.perf_counter()-start
    start=time.perf_counter();derivatives=torch.autograd.grad(out,(x,weight,bias),grad);torch.cuda.synchronize()
    backward=time.perf_counter()-start
    return tuple(z.detach() for z in (out,*derivatives)),forward,backward


def compare(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Matching finite FP32 tensors required')
    diff=(a-b).abs();norm=a.norm().clamp_min(1e-20)
    return dict(bitwise_equal=same_bits(a,b),max_abs=float(diff.max()),
                relative_l2=float((a-b).norm()/norm),allclose_1e4=bool(torch.allclose(a,b,rtol=1e-4,atol=1e-4)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    ids,_,_,_,received,_=source_inputs(args.root);dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        x=core.tensor(received[ids[:8]],'cuda:0');a=dura(x)
        if not torch.isfinite(a).all():raise FloatingPointError('Finite frozen source DURA output required')
    del dura,x,a
    print('POLYPHASE_REAL_SOURCE_SMOKE_PASS',8,flush=True)
    torch.manual_seed(2026110830)
    x=torch.randn(128,192,256,device='cuda:0');w=torch.randn(192,192,5,device='cuda:0')*.02
    b=torch.randn(192,device='cuda:0')*.02;grad=torch.randn(128,192,256,device='cuda:0')
    results=[]
    for dilation in DILATIONS:
        for mode in ('native','polyphase'):run(mode,dilation,x,w,b,grad)
        pairs=[]
        for first in ('native','polyphase'):
            order=(first,'polyphase' if first=='native' else 'native')
            out={mode:run(mode,dilation,x,w,b,grad) for mode in order}
            a,fa,ba=out['native'];c,fc,bc=out['polyphase']
            metrics={name:compare(left,right) for name,left,right in zip(('output','input_grad','weight_grad','bias_grad'),a,c)}
            pairs.append(dict(first=first,native_forward_seconds=fa,polyphase_forward_seconds=fc,
                              native_backward_seconds=ba,polyphase_backward_seconds=bc,metrics=metrics))
        results.append(dict(dilation=dilation,pairs=pairs))
        print('POLYPHASE_DILATION',dilation,flush=True)
    result=dict(status='POLYPHASE_GPU_COMPLETE',commit=args.commit,batch=128,channels=192,length=256,kernel=5,
        dilations=list(DILATIONS),results=results,gpu=torch.cuda.get_device_name(),TX_labels_read=0,F_weights_read=0,
        F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,weights_saved=False,
        caveat='Synthetic fixed tensors, individual convolution only, FP32 deterministic TF32 off and shared GPU. Mathematical equivalence does not imply floating-point, optimizer trajectory, full-model or target parity.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
