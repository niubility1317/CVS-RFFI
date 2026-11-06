"""Held-out source diagnostic for nonlinear DURA residual predictability."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.source_frequency_regression import load_context

FIT_SIZE = 32768
BATCH = 256
RIDGE = .001


def basis(received, anchor):
    x = torch.complex(received[:, 0], received[:, 1])
    a = torch.complex(anchor[:, 0], anchor[:, 1])
    scale = a.abs().square().mean(-1).clamp_min(1e-12).sqrt()
    x, a = x/scale[:, None], a/scale[:, None]
    pa, px = a.abs().square(), x.abs().square()
    columns = (a, a.conj(), x, x.conj(), a*pa, a.conj()*pa,
               x*pa, x.conj()*pa, a*px, a.conj()*px, x*px, x.conj()*px)
    return torch.stack(columns, -1), scale


def solve(gram, rhs):
    diagonal = gram.diagonal().real.mean().clamp_min(1e-12)
    eye = torch.eye(len(gram),dtype=gram.dtype,device=gram.device)
    return torch.linalg.solve(gram+RIDGE*diagonal*eye,rhs)


def fit_group(features, residual, weights, indices, dim):
    z = features[indices, :, :dim] * weights[indices, None, None]
    target = residual[indices] * weights[indices, None]
    z, target = z.reshape(-1,dim), target.reshape(-1,1)
    return z.conj().T @ z, z.conj().T @ target


def diagnostic(data):
    received, anchor, variance, clean = data
    ids = torch.arange(len(received),device=received.device) % FIT_SIZE
    train = ids.remainder(8) != 0
    ratio = variance/received.square().mean((1,2)).clamp_min(1e-12)
    cutoffs = torch.quantile(ratio[train],torch.tensor([.25,.5,.75],device=received.device))
    bins = torch.bucketize(ratio,cutoffs)
    matrices = {dim: torch.zeros((5,dim,dim),dtype=torch.complex64,device=received.device) for dim in (4,12)}
    vectors = {dim: torch.zeros((5,dim,1),dtype=torch.complex64,device=received.device) for dim in (4,12)}
    counts = [0]*5
    ix = train.nonzero().flatten()
    for start in range(0,len(ix),BATCH):
        take = ix[start:start+BATCH]
        features, scale = basis(received[take],anchor[take])
        y = torch.complex(clean[take,0],clean[take,1])
        a = torch.complex(anchor[take,0],anchor[take,1])
        residual = (y-a)/scale[:,None]
        weights = scale/y.abs().square().mean(-1).clamp_min(1e-12).sqrt()
        for group in range(5):
            sel = torch.ones(len(take),dtype=torch.bool,device=take.device) if group == 0 else bins[take] == group-1
            if not sel.any():
                continue
            counts[group] += int(sel.sum())
            for dim in (4,12):
                gram, rhs = fit_group(features,residual,weights,sel,dim)
                matrices[dim][group] += gram
                vectors[dim][group] += rhs
    parameters = {dim: torch.stack([solve(matrices[dim][i],vectors[dim][i]).squeeze(-1) for i in range(5)])
                  for dim in (4,12)}
    ix = (~train).nonzero().flatten()
    rows = []
    for start in range(0,len(ix),BATCH):
        take=ix[start:start+BATCH]
        features, scale = basis(received[take],anchor[take])
        y = torch.complex(clean[take,0],clean[take,1])
        a = torch.complex(anchor[take,0],anchor[take,1])
        outputs = [a]
        for dim in (4,12):
            outputs.extend(a+scale[:,None]*(features[:,:,:dim]*parameters[dim][group]).sum(-1)
                           for group in (0,))
            outputs.append(a+scale[:,None]*(features[:,:,:dim]*parameters[dim][1:][bins[take],None,:]).sum(-1))
        energy = y.abs().square().sum(-1).clamp_min(1e-12)
        errors = torch.stack([(v-y).abs().square().sum(-1)/energy for v in outputs],1)
        if not torch.isfinite(errors).all():
            raise FloatingPointError('Finite held-out source errors required')
        rows.append(torch.cat((take[:,None].float(),bins[take,None].float(),errors),1).cpu())
    rows = torch.cat(rows)
    groups={}
    for view in range(3):
        for group in range(5):
            mask = rows[:,0].long()//FIT_SIZE == view
            if group:
                mask &= rows[:,1].long() == group-1
            values=rows[mask,2:]
            groups[f'view{view+21}/'+('all' if group == 0 else f'noise_q{group}')] = dict(
                count=len(values),mean_nmse=values.mean(0).tolist(),
                cubic_binned_better_than_dura=int((values[:,4] < values[:,0]).sum()),
                cubic_binned_better_than_linear_binned=int((values[:,4] < values[:,2]).sum()))
    return dict(train_ids=28672,heldout_ids=4096,train_counts=counts,noise_cutoffs=cutoffs.cpu().tolist(),groups=groups)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    data=load_context(args.cache)
    print('NONLINEAR_SOURCE_CONTEXT_SMOKE_PASS',flush=True)
    start=time.monotonic()
    result=diagnostic(data)
    torch.cuda.synchronize()
    result.update(status='COMPLETE',commit=args.commit,elapsed_seconds=time.monotonic()-start,
        metric_order=['dura','linear_global','linear_binned','cubic_global','cubic_binned'],
        ridge=RIDGE,split='Physical source ID index modulo 8; all views stay together',
        device=torch.cuda.get_device_name(),TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
        caveat='Source IQ diagnostic only; weights not saved; DURA may have seen source IDs; not TX accuracy.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('NONLINEAR_SOURCE_DIAGNOSTIC_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
