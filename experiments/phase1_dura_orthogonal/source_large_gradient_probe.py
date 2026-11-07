"""Source-only, no-update initial gradient diagnostics for the large DURA branch."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_orthogonal.large_context_dura import LargeContextDura
from experiments.phase1_dura_orthogonal.source_context_cache import _contract, _validate_data
from experiments.phase1_dura_orthogonal.train_source import extract_context, source_loss

IDS_PER_SIDE = 1024
BATCH = 32
MODEL_SEED = 2026110612
ORDER_SEED = 2026110821


def summary(gradients):
    """Describe mean-gradient size and split-half alignment, without independence claims."""
    if gradients.ndim != 3 or gradients.shape[0] != 2 or gradients.shape[1] < 2:
        raise ValueError('Two nonempty sets of batch gradients required')
    means = gradients.mean(1)
    norms = means.norm(dim=1)
    if not torch.isfinite(gradients).all() or (norms == 0).any():
        raise ValueError('Finite nonzero average gradients required')
    batch_norms = gradients.norm(dim=2)
    return dict(side_mean_norms=norms.tolist(),
                side_mean_batch_norms=batch_norms.mean(1).tolist(),
                mean_to_batch_ratio=(norms/batch_norms.mean(1).clamp_min(1e-20)).tolist(),
                split_cosine=float(torch.dot(means[0], means[1])/(norms[0]*norms[1])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and CUDA device required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    payload=torch.load(args.cache,map_location='cpu',weights_only=True)
    fit=np.asarray(payload['contract']['fit_ids'],dtype=np.int64)
    if len(fit)!=32768 or len(np.unique(fit))!=len(fit):
        raise ValueError('Fixed unique source physical IDs required')
    if payload['contract']!=_contract(fit,32,'cuda:0',True):
        raise ValueError('Matching F-free source cache required')
    data=payload['data'];_validate_data(data,payload['counts'],3*len(fit))
    # Fixed source-ID partitions: all three views of a physical record remain together.
    rng=np.random.default_rng(ORDER_SEED)
    chosen=rng.permutation(len(fit))[:2*IDS_PER_SIDE]
    sides=(chosen[:IDS_PER_SIDE],chosen[IDS_PER_SIDE:])
    if np.intersect1d(fit[sides[0]],fit[sides[1]]).size:
        raise ValueError('Disjoint source physical IDs required')
    device='cuda:0'
    dura=frozen_dura(args.root,device)
    with torch.no_grad():
        observed=extract_context(dura,data[0][:32].to(device))
        if any(not torch.allclose(a,b[:32].to(device),rtol=1e-5,atol=1e-6) for a,b in zip(observed,data[1:4])):
            raise ValueError('Real DURA source context replay failed')
    del dura
    print('SOURCE_LARGE_GRADIENT_SMOKE_PASS',32,flush=True)
    torch.manual_seed(MODEL_SEED)
    model=LargeContextDura().to(device).train()
    gradients=[];losses=[]
    for side in sides:
        indexes=np.concatenate([side+j*len(fit) for j in range(3)])
        indexes=rng.permutation(indexes)
        vectors=[]; side_losses=[]
        for start in range(0,len(indexes),BATCH):
            ix=torch.as_tensor(indexes[start:start+BATCH],dtype=torch.long)
            received,anchor,innovation,variance,clean=(v[ix].to(device) for v in data)
            out=model(received,anchor,innovation,variance,validate=False)
            loss=source_loss(out,anchor,clean,objective='dura_guard')
            weight,bias=torch.autograd.grad(loss,(model.out.weight,model.out.bias))
            vectors.append(torch.cat((weight.flatten(),bias.flatten())).detach().cpu())
            side_losses.append(float(loss.detach().cpu()))
        gradients.append(torch.stack(vectors));losses.append(sum(side_losses)/len(side_losses))
    result=summary(torch.stack(gradients))
    args.out.mkdir(parents=True)
    value=dict(status='SOURCE_LARGE_GRADIENT_COMPLETE',commit=args.commit,source_ids_per_side=IDS_PER_SIDE,
               views_per_id=3,batch=BATCH,model_seed=MODEL_SEED,order_seed=ORDER_SEED,
               model_state='fresh_initialization_not_live_training_checkpoint',loss_means=losses,metrics=result,
               TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
               caveat='Gradient is only of output head at initialization; three views share physical records; not TX accuracy or proof of training behavior.')
    (args.out/'result.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print('SOURCE_LARGE_GRADIENT_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
