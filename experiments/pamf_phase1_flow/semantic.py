"""Source-only class-distribution-guided conditional trajectory candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import experiment as core

RUN='20261101-phase1-trisigtime-semantic-rollout-E30-s2026092701-r01'
KL_WEIGHT=.1


def semantic_objective(model,teacher,clean,received,noise):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):raise ValueError('Frozen teacher required')
    a=model.integrate(received,noise,noise_scale=.02,steps=4)
    b=model.integrate(received,-noise,noise_scale=.02,steps=4)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    iq=.5*((a-clean.detach()).square().mean((1,2))+(b-clean.detach()).square().mean((1,2)))/energy
    with torch.no_grad():
        target=teacher.features(clean);logp=F.log_softmax(teacher.classify_features(target),dim=1);p=logp.exp()
        reference=F.normalize(target,dim=1)
    fa=teacher.features(a);fb=teacher.features(b)
    na=F.normalize(fa,dim=1);nb=F.normalize(fb,dim=1)
    anchor=.5*((1-(na*reference).sum(1))+(1-(nb*reference).sum(1)))
    agreement=1-(na*nb).sum(1)
    la=F.log_softmax(teacher.classify_features(fa),dim=1);lb=F.log_softmax(teacher.classify_features(fb),dim=1)
    semantic=.5*((p*(logp-la)).sum(1)+(p*(logp-lb)).sum(1))
    identity=anchor+.25*agreement+KL_WEIGHT*semantic
    return iq+identity,dict(iq=iq,identity=identity,anchor=anchor,agreement=agreement,semantic=semantic,path_a=a,path_b=b)


# Reuse the exact source schedule, checkpoint validation, prediction and scorer.
# Configure only in the dedicated candidate process, never on import.
def configure():
    core.RUN=RUN
    core.rollout_objective=semantic_objective


def train(root,out,commit):
    configure();return core.train(root,out,commit)


def predict(root,out,commit):
    configure();return core.predict(root,out,commit)


def score(root,out,commit):
    configure();return core.score(root,out,commit)


read,write=core.read,core.write


def smoke(root,out,commit):
    configure()
    teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract)
    idx=np.flatnonzero(roles=='L_s')[:2]
    x=core.tensor(clean[idx],'cpu');y=core.tensor(np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[idx],'cpu')
    model=core.make_model();noise=torch.randn(y.shape,generator=torch.Generator().manual_seed(2026110105))
    _,parts=semantic_objective(model,teacher,x,y,noise)
    grad=torch.autograd.grad(parts['semantic'].sum(),model.output.weight)[0]
    if not torch.isfinite(grad).all() or not grad.abs().sum()>0:raise ValueError('Actual semantic guidance gradient required')
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Frozen teacher acquired gradients')
    core.write(out/'semantic_smoke.json',dict(status='PASS',semantic_gradient_norm=float(grad.norm()),
          F_run=core.F_RUN,F_row=core.F_ROW,source_records=2,TX_labels_read=0,test_IQ_reads=0))
    core.smoke(root,out,commit)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'runs'/RUN
    {'smoke':smoke,'score':score}[a.mode](a.root,out,a.commit)


if __name__=='__main__':main()
