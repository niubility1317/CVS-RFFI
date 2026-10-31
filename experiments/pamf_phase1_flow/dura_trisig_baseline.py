"""Reclassify existing Phase1 DURA IQ with the frozen Tri-SIG-TIME reader."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import experiment as core

RUN='20261101-phase1-dura-trisigtime-baseline-s2026092701-r01'
PRIOR='20261101-phase1-trisigtime-source-rollout-E30-s2026092701-r02'
DURA='runs/20261021-phase1-restoration-stage2-graph-s2026092701-r01/stage2_residual_graph/target_confirmation/native_map_prior/prediction'
read,write=core.read,core.write


def configure():core.RUN=RUN


def smoke(root,out,commit):
    configure();f,contract=core.teacher(root,'cpu');_,clean,roles=core.data(root,contract)
    idx=np.flatnonzero(roles=='L_s')[:2];x=core.tensor(clean[idx],'cpu')
    with torch.no_grad():scores=f.classify_features(f.features(x))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(p.requires_grad for p in f.parameters()):
        raise ValueError('Actual frozen Tri-SIG-TIME source checkpoint required')
    write(out/'smoke.json',dict(status='PASS',F_run=core.F_RUN,F_row=core.F_ROW,source_rows=2,F_frozen=True,TX_labels_read=0,test_IQ_reads=0))


@torch.no_grad()
def predict(root,out,commit):
    configure();prior=root/'runs'/PRIOR/'flow/test_prediction';prior_done=read(prior/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,TX_truth_reads=0,model_updates=0)
    if any(prior_done.get(k)!=v for k,v in expected.items()):raise ValueError('All fixed same-reader raw decisions required')
    dura=root/DURA;d=read(dura/'complete.json')
    expected_dura=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=168000,views=list(core.VIEWS),TX_truth_reads=0,state_updates=0,head_fit_updates=0)
    if any(d.get(k)!=v for k,v in expected_dura.items()):raise ValueError('Complete fixed Phase1 DURA IQ required')
    _,ids=core.reference(root)
    if not np.array_equal(ids,np.load(prior/'physical_ids.npy',allow_pickle=False)) or not np.array_equal(ids,np.load(dura/'physical_ids.npy',allow_pickle=False)):
        raise ValueError('Same ordered Phase1 physical records required')
    f,_=core.teacher(root,'cuda:0');row=out/'flow';row.mkdir();dst=row/'test_prediction';dst.mkdir();np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    for view in core.VIEWS:
        x=np.load(dura/(view+'.restored.iq.npy'),mmap_mode='r',allow_pickle=False)
        raw=np.load(prior/(view+'.raw.npy'),mmap_mode='r',allow_pickle=False)
        if x.shape!=(len(ids),2,256) or x.dtype!=np.float32 or raw.shape!=(len(ids),) or raw.dtype!=np.int64 or np.any((raw<0)|(raw>=6)):
            raise ValueError('Aligned complete IQ and raw decisions required')
        np.save(dst/(view+'.raw.npy'),raw,allow_pickle=False)
        pred=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        for offset in range(0,len(ids),core.BATCH):
            sample=core.tensor(x[offset:offset+core.BATCH],'cuda:0');scores=f.classify_features(f.features(sample))
            if not torch.isfinite(sample).all() or not torch.isfinite(scores).all() or scores.shape!=(len(sample),6):raise ValueError('Finite baseline inference required')
            pred[offset:offset+len(sample)]=np.from_dlpack(scores.argmax(1).cpu())
        pred.flush();print('DURA_TRISIG_VIEW_COMPLETE',view,len(ids),flush=True)
    write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,
          commit=commit,TX_truth_reads=0,model_updates=0))


def score(root,out,commit):
    configure();return core.score(root,out,commit)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'smoke':smoke,'score':score}[a.mode](a.root,a.root/'runs'/RUN,a.commit)


if __name__=='__main__':main()
