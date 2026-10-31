"""Phase1 source-only semantic trajectories with hard-channel weighting."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import experiment as core
from .semantic import semantic_objective

RUN='20261101-phase1-trisigtime-hard-semantic-E30-s2026092701-r01'
VARIANCE_CUTOFF=0.004452520050108433
HARD_WEIGHT=3.0
read,write=core.read,core.write


def configure():
    core.RUN=RUN
    core.rollout_objective=semantic_objective


def source_weights(variance,clean_prefix):
    if variance.ndim!=1 or not 0<=clean_prefix<=len(variance) or not torch.isfinite(variance).all():
        raise ValueError('Finite source variance and valid clean-prefix size required')
    weights=torch.where(variance>=VARIANCE_CUTOFF,torch.full_like(variance,HARD_WEIGHT),torch.ones_like(variance))
    weights[:clean_prefix]=1.
    return weights


def smoke(root,out,commit):
    configure();teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract)
    idx=np.flatnonzero(roles=='L_s')[:2]
    x=core.tensor(clean[idx],'cpu');y=core.tensor(np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[idx],'cpu')
    model=core.make_model();noise=torch.randn(y.shape,generator=torch.Generator().manual_seed(2026110105))
    _,parts=semantic_objective(model,teacher,x,y,noise)
    grad=torch.autograd.grad(parts['semantic'].sum(),model.output.weight)[0]
    if not torch.isfinite(grad).all() or not grad.abs().sum()>0:raise ValueError('Actual semantic guidance gradient required')
    values=source_weights(torch.tensor([0.,1.,1.]),clean_prefix=1)
    if values.tolist()!=[1.,3.,3.] or source_weights(torch.tensor([1.,1.]),clean_prefix=1).tolist()!=[1.,3.]:
        raise ValueError('Hard-source and clean weights differ')
    write(out/'semantic_smoke.json',dict(status='PASS',semantic_gradient_norm=float(grad.norm()),
          F_run=core.F_RUN,F_row=core.F_ROW,source_records=2,TX_labels_read=0,test_IQ_reads=0,
          hard_weight=HARD_WEIGHT,variance_cutoff=VARIANCE_CUTOFF))
    core.smoke(root,out,commit)


def train(root,out,commit):
    configure();f,contract=core.teacher(root,'cuda:0');cache,clean,roles=core.data(root,contract)
    frozen={k:v.detach().cpu().clone() for k,v in f.state_dict().items()}
    model=core.make_model().cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(2026110104);torch.manual_seed(2026110105);torch.cuda.manual_seed_all(2026110105)
    row=out/'flow';row.mkdir();updates=0;started=time.perf_counter()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,core.EPOCHS+1):
            view=(epoch-1)%20+1;folder=cache/'epochs'/f'{view:02d}'
            received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape or received.dtype!=np.float32 or variance.shape!=(core.COUNT,) or not np.isfinite(variance).all():
                raise ValueError('Aligned Phase1 source IQ and variance required')
            order=rng.permutation(core.COUNT);clean_count=core.COUNT//4;total=0.;hard_count=0
            for offset in range(0,core.COUNT,core.BATCH):
                ix=order[offset:offset+core.BATCH];a=core.tensor(clean[ix],'cuda:0');b=core.tensor(received[ix],'cuda:0')
                n=max(0,min(len(ix),clean_count-offset))
                if n:b[:n]=a[:n]
                v=core.tensor(variance[ix],'cuda:0');weights=source_weights(v,n);hard_count+=int((weights>1).sum())
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=core.lr(updates)
                loss,parts=semantic_objective(model,f,a,b,torch.randn_like(a))
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source objective')
                ((loss*weights).sum()/weights.sum()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1
                total+=float(loss.detach().sum())
            record=dict(epoch=epoch,view=view,updates=updates,unweighted_loss=total/core.COUNT,
                        hard_received_count=hard_count,hard_weight=HARD_WEIGHT,variance_cutoff=VARIANCE_CUTOFF,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(record),flush=True)
    if updates!=core.STEPS or any(p.grad is not None for p in f.parameters()) or any(not torch.equal(v.detach().cpu(),frozen[k]) for k,v in f.state_dict().items()):
        raise ValueError('Training budget or frozen Tri-SIG-TIME invariant failed')
    core.save(row/'final.pt',model,commit,core.EPOCHS)
    write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=core.EPOCHS,updates=updates,F_run=core.F_RUN,F_row=core.F_ROW,
          F_frozen=True,TX_labels_read=0,test_IQ_reads=0,source_roles=['L_s','U_s'],parameters=sum(p.numel() for p in model.parameters()),
          hard_weight=HARD_WEIGHT,variance_cutoff=VARIANCE_CUTOFF,clean_identity_weight=1.))


def predict(root,out,commit):
    configure();return core.predict(root,out,commit)


def score(root,out,commit):
    configure();return core.score(root,out,commit)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'smoke':smoke,'score':score}[a.mode](a.root,a.root/'runs'/RUN,a.commit)


if __name__=='__main__':main()
