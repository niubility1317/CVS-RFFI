"""Source-only identity evidence for a frozen DURA-conditioned flow."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import experiment as core

RUN='20261101-diagnostic-phase1-dura-trisig-source-s2026092701-r01'
PARENT='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);f,contract=core.teacher(a.root,'cpu');cache,clean,roles=core.data(a.root,contract)
    parent=a.root/'runs'/PARENT;launch=core.read(parent/'launch.json')
    if (launch.get('commit'),launch.get('CUDA_VISIBLE_DEVICES'),launch.get('GPU_workers')) != ('7e2d22ed1fe0feeac1df8381be89e4bb527108c5','',0):
        raise ValueError('Pinned CPU Phase1 source producer required')
    all_ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);results=[];common=None
    fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)]
                          for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
    for view in (1,20):
        with np.load(parent/f'DURA_default-{view}.npz',allow_pickle=False) as z:
            saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','scene','state')}
        ix=saved['source_positions'];ids=saved['physical_ids']
        if ix.shape!=(512,) or not np.array_equal(ix,fixed) or not np.array_equal(ids,all_ids[ix]) or not np.array_equal(saved['roles'],roles[ix]) or not np.array_equal(saved['clean'],clean[ix]):
            raise ValueError('Source physical rows, roles and clean pairs must match')
        if (saved['roles']=='L_s').sum()!=128 or (saved['roles']=='U_s').sum()!=384:raise ValueError('Original source sample roles required')
        if common is not None and not np.array_equal(ids,common):raise ValueError('Two source views must use same physical IDs')
        common=ids
        if saved['native'].shape!=(512,2,256) or saved['native'].dtype!=np.float32 or not np.isfinite(saved['native']).all():raise ValueError('Original frozen DURA source output required')
        raw=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
        if raw.shape!=(core.COUNT,2,256) or raw.dtype!=np.float32:raise ValueError('Original matched source received IQ required')
        logits={'clean':[],'raw':[],'dura':[]};features={'clean':[],'raw':[],'dura':[]}
        for offset in range(0,512,32):
            sl=slice(offset,offset+32)
            for key,data in (('clean',saved['clean'][sl]),('raw',raw[ix[sl]]),('dura',saved['native'][sl])):
                feat=f.features(core.tensor(data,'cpu'));scores=f.classify_features(feat)
                if not torch.isfinite(feat).all() or not torch.isfinite(scores).all():raise ValueError('Finite frozen F outputs required')
                features[key].append(feat);logits[key].append(scores)
        logits={k:torch.cat(v) for k,v in logits.items()};features={k:torch.cat(v) for k,v in features.items()}
        reference=logits['clean'].argmax(1);pclean=F.softmax(logits['clean'],dim=1);logp=F.log_softmax(logits['clean'],dim=1)
        values={}
        for key in ('raw','dura'):
            prediction=logits[key].argmax(1);correct=prediction==reference
            kl=(pclean*(logp-F.log_softmax(logits[key],dim=1))).sum(1)
            cosine=(F.normalize(features['clean'],dim=1)*F.normalize(features[key],dim=1)).sum(1)
            iq=core.tensor(raw[ix] if key=='raw' else saved['native'],'cpu');cl=core.tensor(saved['clean'],'cpu')
            nmse=(iq-cl).square().mean((1,2))/cl.square().mean((1,2)).clamp_min(1e-4)
            values[key]=dict(match=correct,kl=kl,cosine=cosine,nmse=nmse)
        masks={'all':np.ones(512,dtype=bool)}
        for key in ('scene','state','roles'):
            for value in np.unique(saved[key]):masks[key+':'+str(value)]=saved[key]==value
        groups=[]
        for group,mask in masks.items():
            t=torch.tensor(mask.tolist(),dtype=torch.bool)
            entry=dict(group=group,count=int(mask.sum()))
            for key in ('raw','dura'):
                entry[key]=dict(clean_teacher_agreement=float(values[key]['match'][t].float().mean()),
                    mean_KL=float(values[key]['kl'][t].mean()),feature_cosine=float(values[key]['cosine'][t].mean()),
                    iq_nmse=float(values[key]['nmse'][t].mean()))
            entry['dura_repairs']=int((values['dura']['match'] & ~values['raw']['match'] & t).sum())
            entry['dura_damages']=int((~values['dura']['match'] & values['raw']['match'] & t).sum())
            groups.append(entry)
        results.append(dict(source_view=view,count=512,groups=groups))
    out=a.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique source diagnostic output required')
    out.mkdir();core.write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=a.commit,F_run=core.F_RUN,F_row=core.F_ROW,
        source_unique_records=512,source_observations=1024,source_views=[1,20],TX_labels_read=0,V_reads=0,test_IQ_reads=0,F_updates=0,G_updates=0,
        source_dura_native_parent=PARENT,results=results,interpretation='Teacher agreement is not accuracy; DURA native only, no oracle-aligned IQ or old classifier outputs consumed.'))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY',run=RUN),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
