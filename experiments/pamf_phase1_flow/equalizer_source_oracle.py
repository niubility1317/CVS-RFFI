"""Privileged complex-scalar diagnostic on paired Phase1 training records."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import anchor_balanced as prior, experiment as core
from .dura_source_trisig_diagnostic import PARENT

RUN='20261101-diagnostic-phase1-complex-scalar-oracle-s2026092701-r01'


def scalar_oracle(received, clean):
    received=np.asarray(received,dtype=np.complex128);clean=np.asarray(clean,dtype=np.complex128)
    if received.ndim!=2 or received.shape!=clean.shape or not np.isfinite(received).all() or not np.isfinite(clean).all():
        raise ValueError('Aligned finite complex source windows required')
    power=np.sum(np.abs(received)**2,axis=1)
    if np.any(power<=1e-12):raise ValueError('Nonzero received energy required')
    coefficient=np.sum(np.conj(received)*clean,axis=1)/power
    return received*coefficient[:,None]


def to_complex(iq):
    x=np.asarray(iq,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Two-channel source IQ windows required')
    return x[:,0]+1j*x[:,1]


def to_iq(z):return np.stack((z.real,z.imag),axis=1).astype(np.float32)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2);teacher,contract=core.teacher(a.root,'cpu');cache,clean,roles=core.data(a.root,contract)
    before=prior.snapshot(teacher);parent=a.root/'runs'/PARENT
    launch=core.read(parent/'launch.json')
    if (launch.get('commit'),launch.get('CUDA_VISIBLE_DEVICES'),launch.get('GPU_workers'))!=('7e2d22ed1fe0feeac1df8381be89e4bb527108c5','',0):
        raise ValueError('Pinned CPU Phase1 source producer required')
    all_ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)]
                          for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
    results=[];common=None
    for view in (1,20):
        with np.load(parent/f'DURA_default-{view}.npz',allow_pickle=False) as z:
            saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','scene','state')}
        ix=saved['source_positions'];ids=saved['physical_ids']
        if (ix.shape!=(512,) or not np.array_equal(ix,fixed) or not np.array_equal(ids,all_ids[ix])
                or not np.array_equal(saved['roles'],roles[ix]) or not np.array_equal(saved['clean'],clean[ix])):
            raise ValueError('Exact paired source records required')
        if common is not None and not np.array_equal(ids,common):raise ValueError('Both views must share source IDs')
        common=ids
        raw=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
        if raw.shape!=clean.shape or raw.dtype!=np.float32:raise ValueError('Exact source IQ required')
        r=to_complex(raw[ix]);target=to_complex(saved['clean']);dura=to_complex(saved['native'])
        restored=scalar_oracle(r,target);values={'raw':r,'dura':dura,'scalar_oracle':restored}
        scores={key:[] for key in ('clean',*values)}
        for off in range(0,512,32):
            sl=slice(off,off+32)
            for key,data in [('clean',target[sl]),*((k,v[sl]) for k,v in values.items())]:
                logits=teacher.classify_features(teacher.features(core.tensor(to_iq(data),'cpu')))
                if not torch.isfinite(logits).all():raise ValueError('Finite teacher scores required')
                scores[key].extend(logits.argmax(1).tolist())
        reference=np.asarray(scores['clean']);energy=np.maximum(np.mean(abs(target)**2,axis=1),1e-4)
        match={k:np.asarray(scores[k])==reference for k in values}
        nmse={k:np.mean(abs(v-target)**2,axis=1)/energy for k,v in values.items()}
        if np.any(nmse['scalar_oracle']>nmse['raw']+1e-10):raise ValueError('Scalar least-squares objective must not exceed raw')
        masks={'all':np.ones(512,dtype=bool)}
        for key in ('scene','state','roles'):
            for value in np.unique(saved[key]):masks[key+':'+str(value)]=saved[key]==value
        groups=[]
        for name,mask in masks.items():
            entry=dict(group=name,count=int(mask.sum()))
            for key in values:
                entry[key]=dict(clean_teacher_agreement=float(match[key][mask].mean()),iq_nmse=float(nmse[key][mask].mean()))
            entry['oracle_repairs_vs_dura']=int((mask&match['scalar_oracle']&~match['dura']).sum())
            entry['oracle_damages_vs_dura']=int((mask&~match['scalar_oracle']&match['dura']).sum())
            groups.append(entry)
        results.append(dict(source_view=view,count=512,groups=groups))
    if not prior.unchanged(teacher,before):raise ValueError('Frozen teacher changed')
    out=a.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique source diagnostic output required')
    out.mkdir();core.write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=a.commit,F_run=core.F_RUN,F_row=core.F_ROW,
        source_unique_records=512,source_observations=1024,source_views=[1,20],TX_labels_read=0,V_reads=0,test_IQ_reads=0,F_updates=0,G_updates=0,
        source_dura_native_parent=PARENT,results=results,interpretation='Scalar uses unavailable paired clean IQ per record; privileged training-source counterfactual, clean-teacher agreement not true accuracy; not an upper bound for other methods.'))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY',run=RUN)),flush=True)


if __name__=='__main__':main()
