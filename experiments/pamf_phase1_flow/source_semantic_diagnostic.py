"""Phase1 source-only, label-free frozen teacher consistency diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from . import experiment as core

RUN='20261101-diagnostic-phase1-trisig-source-semantics-s2026092701-r01'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2)
    reader,contract=core.teacher(a.root,'cpu');cache,clean,roles=core.data(a.root,contract)
    variance=np.load(cache/'epochs/01/variance.npy',mmap_mode='r',allow_pickle=False)
    received=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    if variance.shape!=(core.COUNT,) or received.shape!=clean.shape or not np.isfinite(variance).all():raise ValueError('Complete Phase1 source action view required')
    cuts=np.quantile(variance,[0,.25,.5,.75,1]);rng=np.random.default_rng(2026110111);groups=[]
    for i in range(4):
        eligible=np.flatnonzero((variance>=cuts[i]) & (variance<=cuts[i+1] if i==3 else variance<cuts[i+1]))
        if len(eligible)<128:raise ValueError('Fixed noise strata too small')
        groups.append(rng.choice(eligible,128,replace=False))
    out=a.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique diagnostic output required')
    out.mkdir();result=[]
    with torch.no_grad():
        for k,ids in enumerate(groups):
            target=[];observe=[];feature=[]
            for offset in range(0,len(ids),32):
                ix=ids[offset:offset+32];x=core.tensor(clean[ix],'cpu');y=core.tensor(received[ix],'cpu')
                fx=reader.features(x);fy=reader.features(y)
                target.append(reader.classify_features(fx));observe.append(reader.classify_features(fy))
                feature.append((F.normalize(fx,dim=1)*F.normalize(fy,dim=1)).sum(1))
            x=torch.cat(target);y=torch.cat(observe);cos=torch.cat(feature)
            prob=x.softmax(1);kl=(prob*(x.log_softmax(1)-y.log_softmax(1))).sum(1)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(kl).all():raise ValueError('Finite source predictions required')
            result.append(dict(stratum=k,count=len(ids),variance_min=float(variance[ids].min()),variance_max=float(variance[ids].max()),
                L_s=int((roles[ids]=='L_s').sum()),U_s=int((roles[ids]=='U_s').sum()),
                top1_agreement=float((x.argmax(1)==y.argmax(1)).float().mean()),
                clean_mean_confidence=float(prob.max(1).values.mean()),
                mean_feature_cosine=float(cos.mean()),mean_teacher_KL=float(kl.mean())))
    core.write(out/'complete.json',dict(status='ANALYZED',run=RUN,commit=a.commit,source_count=512,
        source_view='Phase1 R07 epoch01',source_only=True,seed=2026110111,variance_cutpoints=cuts.tolist(),
        F_run=core.F_RUN,F_row=core.F_ROW,F_updates=0,TX_labels_read=0,test_IQ_reads=0,results=result,
        interpretation='Agreement and confidence are not accuracy; no source or test TX labels used.'))
    print(json.dumps(dict(status='ANALYZED',run=RUN,results=result)),flush=True)


if __name__=='__main__':main()
