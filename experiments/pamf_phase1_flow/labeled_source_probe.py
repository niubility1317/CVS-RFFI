"""Measure frozen clean teacher and DURA on legally labeled Phase1 L_s only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import anchor_balanced as prior, experiment as core

RUN='20261101-diagnostic-phase1-labeled-source-teacher-s2026092701-r01'


def align_labeled_positions(roles,physical,ids):
    roles=np.asarray(roles);physical=np.asarray(physical);ids=np.asarray(ids)
    if roles.ndim!=1 or physical.shape!=roles.shape or ids.ndim!=1:
        raise ValueError('One-dimensional physical IDs and roles required')
    labeled=np.flatnonzero(roles=='L_s')
    lookup={str(physical[i]):int(i) for i in labeled}
    if len(lookup)!=len(labeled) or len(ids)!=len(labeled) or len(set(ids.tolist()))!=len(ids) or set(map(str,ids))!=set(lookup):
        raise ValueError('Exact distinct L_s physical IDs required')
    return np.asarray([lookup[str(item)] for item in ids],dtype=np.int64)


def source_counts(truth,clean,dura,mask):
    arrays=[np.asarray(x) for x in (truth,clean,dura,mask)]
    if any(x.ndim!=1 or x.shape!=arrays[0].shape for x in arrays) or arrays[3].dtype!=np.bool_ or not arrays[3].any():
        raise ValueError('Aligned nonempty source labels and boolean mask required')
    y,a,b,m=arrays;ac=(a==y);bc=(b==y);diff=(a!=b)
    return dict(count=int(m.sum()),clean_correct=int((m&ac).sum()),dura_correct=int((m&bc).sum()),
        clean_dura_disagree=int((m&diff).sum()),disagree_clean_only_correct=int((m&diff&ac&~bc).sum()),
        disagree_dura_only_correct=int((m&diff&~ac&bc).sum()),disagree_both_wrong=int((m&diff&~ac&~bc).sum()))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2)
    teacher,contract=core.teacher(args.root,'cpu');cache,clean,roles=core.data(args.root,contract);dura=prior.frozen_dura(args.root,'cpu')
    f_complete=core.read(args.root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    source_cache=Path(f_complete['config']['source_cache'])
    with np.load(source_cache/'L_s_metadata.npz',allow_pickle=False) as data:
        ids=data['ids'];truth=data['y']
    all_ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    positions=align_labeled_positions(roles,all_ids,ids)
    from experiments.cric_dura_comparison.source_iq import aligned_source_iq
    labeled_x,binding=aligned_source_iq(source_cache,'L_s',ids)
    if (positions.shape!=(6300,) or truth.shape!=(6300,) or not np.array_equal(ids,all_ids[positions])
        or list(ids)!=contract['role_ids']['L_s'] or labeled_x.shape!=(6300,2,256)
        or not np.allclose(labeled_x,clean[positions],rtol=1e-5,atol=1e-6) or np.any(truth<0) or np.any(truth>=6)):
        raise ValueError('Exact legal L_s labels, IDs, class range and clean IQ required')
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura)
    def classify(x):return teacher.classify_features(teacher.features(x))
    clean_pred=[]
    for offset in range(0,len(positions),32):
        idx=positions[offset:offset+32]
        scores=classify(core.tensor(clean[idx],'cpu'))
        if not torch.isfinite(scores).all():raise ValueError('Finite clean teacher scores required')
        clean_pred.extend(scores.argmax(1).tolist())
    clean_pred=np.asarray(clean_pred,dtype=np.int64);results=[]
    for view in (1,20):
        folder=cache/'epochs'/f'{view:02d}'
        received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
        with np.load(folder/'groups.npz',allow_pickle=False) as data:
            groups={key:data[key][positions] for key in ('scene','state')}
        if received.shape!=clean.shape or received.dtype!=np.float32:raise ValueError('Aligned received source IQ required')
        dura_pred=[]
        for offset in range(0,len(positions),32):
            idx=positions[offset:offset+32]
            scores=classify(dura(core.tensor(received[idx],'cpu')))
            if not torch.isfinite(scores).all():raise ValueError('Finite restored teacher scores required')
            dura_pred.extend(scores.argmax(1).tolist())
        dura_pred=np.asarray(dura_pred,dtype=np.int64)
        masks={'all':np.ones(len(positions),dtype=bool)}
        for key in ('scene','state'):
            for value in np.unique(groups[key]):masks[f'{key}:{value}']=groups[key]==value
        entries=[]
        for name,mask in masks.items():
            counts=source_counts(truth,clean_pred,dura_pred,mask);counts['group']=name
            entries.append(counts)
        results.append(dict(source_view=view,groups=entries))
    if not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d):raise ValueError('Frozen networks changed')
    out=args.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique diagnostic output required')
    out.mkdir()
    core.write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=args.commit,F_row=core.F_ROW,
        labeled_source_role='L_s',source_unique_records=len(positions),source_views=[1,20],source_observations=2*len(positions),
        L_s_labels_read=len(positions),U_s_labels_read=0,V_reads=0,test_IQ_reads=0,model_updates=0,
        physical_iq_binding=binding,interpretation='L_s source train labels only; same source IDs across views; not independent test accuracy.',results=results))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY',run=RUN)),flush=True)


if __name__=='__main__':main()
