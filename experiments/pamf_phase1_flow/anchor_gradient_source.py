"""Fixed source-only gradient scaling study for anchored trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import experiment as core
from .anchor_flow import source_objective

RUN='20261101-diagnostic-phase1-anchor-gradient-scale-s2026092701-r01'
PARENT='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);f,contract=core.teacher(args.root,'cpu');cache,clean,roles=core.data(args.root,contract)
    parent=args.root/'runs'/PARENT;launch=core.read(parent/'launch.json')
    if launch.get('commit')!='7e2d22ed1fe0feeac1df8381be89e4bb527108c5':raise ValueError('Pinned Phase1 source producer required')
    with np.load(parent/'DURA_default-1.npz',allow_pickle=False) as z:
        saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','state')}
    ix=saved['source_positions'];fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)]
                         for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
    ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    if not np.array_equal(ix,fixed) or not np.array_equal(saved['physical_ids'],ids[ix]) or not np.array_equal(saved['roles'],roles[ix]) or not np.array_equal(saved['clean'],clean[ix]):
        raise ValueError('Exact Phase1 source IDs, roles and clean alignment required')
    if saved['native'].shape!=(512,2,256) or saved['native'].dtype!=np.float32 or not np.isfinite(saved['native']).all():raise ValueError('Finite frozen DURA source IQ required')
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    if raw.shape!=(core.COUNT,2,256) or raw.dtype!=np.float32:raise ValueError('Complete Phase1 received source view required')
    f_before={k:v.clone() for k,v in f.state_dict().items()};results=[]
    for i,state in enumerate(('clean','LOS','BLOCKED')):
        chosen=np.flatnonzero(saved['state']==state)[:32]
        if len(chosen)!=32:raise ValueError('Fixed source state group incomplete')
        x=core.tensor(raw[ix[chosen]],'cpu');anchor=core.tensor(saved['native'][chosen],'cpu');target=core.tensor(saved['clean'][chosen],'cpu')
        field=core.make_model();noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(2026110113+i))
        _,terms=source_objective(field,f,x,anchor,target,noise)
        gradients={key:torch.autograd.grad(terms[key].mean(),field.output.weight,retain_graph=True)[0] for key in ('iq','semantic','feature','agreement','deviation')}
        if any(not torch.isfinite(v).all() for v in gradients.values()):raise FloatingPointError('Finite source gradients required')
        norms={key:float(v.norm()) for key,v in gradients.items()}
        if state!='clean' and norms['semantic']<=0:raise ValueError('Nonzero corrupted-source semantic gradient required')
        cosine=float(F.cosine_similarity(gradients['iq'].flatten(),gradients['semantic'].flatten(),dim=0))
        results.append(dict(source_state=state,count=len(chosen),L_s=int((saved['roles'][chosen]=='L_s').sum()),U_s=int((saved['roles'][chosen]=='U_s').sum()),
            gradient_norms=norms,iq_semantic_gradient_cosine=cosine,iq_to_semantic_ratio=norms['iq']/norms['semantic'] if norms['semantic']>0 else None,
            losses={k:float(terms[k].detach().mean()) for k in ('iq','semantic','feature','agreement','deviation')}))
    if any(v.grad is not None for v in f.parameters()) or any(not torch.equal(v,f_before[k]) for k,v in f.state_dict().items()):raise ValueError('Frozen Tri-SIG-TIME changed')
    ratios=[item['iq_to_semantic_ratio'] for item in results if item['source_state'] in ('LOS','BLOCKED')]
    # Predeclared, source-only ratio rule; fixed before inspecting these gradients.
    suggested=float(np.clip(np.median(ratios),1e-5,1e-2))
    out=args.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique source diagnostic output required')
    out.mkdir();core.write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=args.commit,F_run=core.F_RUN,F_row=core.F_ROW,
         source_records=96,source_IQ_materialized=512,source_states=['clean','LOS','BLOCKED'],source_view=1,source_seed=2026110113,source_dura_native_parent=PARENT,
         TX_labels_read=0,V_reads=0,test_IQ_reads=0,F_updates=0,G_updates=0,field_updates=0,
         rule='clip(median(IQ_grad_norm/KL_grad_norm over LOS and BLOCKED),1e-5,1e-2)',suggested_KL_weight=suggested,results=results,
         caveat='Zero-field output-layer gradient norms on fixed source records; no accuracy, test-performance, or generalization claim.'))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY',run=RUN,suggested_KL_weight=suggested)),flush=True)


if __name__=='__main__':main()
