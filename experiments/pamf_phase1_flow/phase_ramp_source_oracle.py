"""Privileged gain-plus-phase-ramp counterfactual on Phase1 source IQ."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import anchor_balanced as prior, experiment as core
from .dura_source_trisig_diagnostic import PARENT
from .equalizer_source_oracle import scalar_oracle, to_complex, to_iq

RUN='20261101-diagnostic-phase1-phase-ramp-oracle-s2026092701-r01'


def phase_ramp_oracle(received, clean, *, nfft=4096):
    received=np.asarray(received,dtype=np.complex128);clean=np.asarray(clean,dtype=np.complex128)
    if (received.ndim!=2 or received.shape!=clean.shape or not np.isfinite(received).all()
            or not np.isfinite(clean).all() or not isinstance(nfft,int) or nfft<received.shape[1]):
        raise ValueError('Aligned finite complex windows and sufficient grid required')
    power=np.sum(abs(received)**2,axis=1)
    if np.any(power<=1e-12):raise ValueError('Nonzero received energy required')
    spectrum=np.fft.fft(np.conj(received)*clean,n=nfft,axis=1)
    index=np.abs(spectrum).argmax(axis=1)
    coefficient=spectrum[np.arange(len(received)),index]/power
    rotation=np.exp(2j*np.pi*index[:,None]*np.arange(received.shape[1])[None,:]/nfft)
    return received*rotation*coefficient[:,None]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2);teacher,contract=core.teacher(a.root,'cpu');cache,clean,roles=core.data(a.root,contract)
    before=prior.snapshot(teacher);parent=a.root/'runs'/PARENT;launch=core.read(parent/'launch.json')
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
        common=ids;raw=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
        if raw.shape!=clean.shape or raw.dtype!=np.float32:raise ValueError('Exact source IQ required')
        received=to_complex(raw[ix]);target=to_complex(saved['clean']);dura=to_complex(saved['native'])
        scalar=scalar_oracle(received,target);ramp=phase_ramp_oracle(received,target)
        values={'raw':received,'dura':dura,'scalar_oracle':scalar,'ramp_oracle':ramp};scores={k:[] for k in ('clean',*values)}
        for off in range(0,512,32):
            sl=slice(off,off+32)
            for key,data in [('clean',target[sl]),*((k,v[sl]) for k,v in values.items())]:
                logits=teacher.classify_features(teacher.features(core.tensor(to_iq(data),'cpu')))
                if not torch.isfinite(logits).all():raise ValueError('Finite teacher scores required')
                scores[key].extend(logits.argmax(1).tolist())
        reference=np.asarray(scores['clean']);energy=np.maximum(np.mean(abs(target)**2,axis=1),1e-4)
        match={k:np.asarray(scores[k])==reference for k in values}
        nmse={k:np.mean(abs(v-target)**2,axis=1)/energy for k,v in values.items()}
        if np.any(nmse['ramp_oracle']>nmse['scalar_oracle']+1e-10):raise ValueError('Grid includes scalar zero-frequency case')
        masks={'all':np.ones(512,dtype=bool)}
        for key in ('scene','state','roles'):
            for value in np.unique(saved[key]):masks[key+':'+str(value)]=saved[key]==value
        groups=[]
        for name,mask in masks.items():
            entry=dict(group=name,count=int(mask.sum()))
            for key in values:entry[key]=dict(clean_teacher_agreement=float(match[key][mask].mean()),iq_nmse=float(nmse[key][mask].mean()))
            entry['ramp_repairs_vs_dura']=int((mask&match['ramp_oracle']&~match['dura']).sum())
            entry['ramp_damages_vs_dura']=int((mask&~match['ramp_oracle']&match['dura']).sum())
            groups.append(entry)
        results.append(dict(source_view=view,count=512,groups=groups))
    if not prior.unchanged(teacher,before):raise ValueError('Frozen teacher changed')
    out=a.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique source diagnostic output required')
    out.mkdir();core.write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=a.commit,F_run=core.F_RUN,F_row=core.F_ROW,
        source_unique_records=512,source_observations=1024,source_views=[1,20],phase_grid=4096,TX_labels_read=0,V_reads=0,test_IQ_reads=0,F_updates=0,G_updates=0,
        source_dura_native_parent=PARENT,results=results,interpretation='Grid search uses unavailable clean IQ and unrestricted phase ramp; source-training counterfactual, not accuracy or deployable recovery.'))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY',run=RUN)),flush=True)


if __name__=='__main__':main()
