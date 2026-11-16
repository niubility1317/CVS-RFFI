"""Diagnose DA-to-Orthogonal score geometry using saved B and support only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .clean_mopc_diagnostic import support_only_clean
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group2_b_state import load_b_state
from .group2_cil_bridge import register
from .io import activate, read, require
from .vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import compute_class_prototypes


def summarize_scores(scores,labels,old_count):
    require(scores.ndim==2 and len(scores)==len(labels) and scores.shape[1]>old_count,
            'Invalid support score shape')
    require(bool(torch.isfinite(scores).all()),'Nonfinite support scores')
    choice=scores.argmax(1);restricted=scores[:,:old_count].argmax(1)
    return dict(count=len(labels),all_correct=int((choice==labels).sum()),
                old_only_correct=int((restricted==labels).sum()),predicted_new=int((choice>=old_count).sum()),
                old_max_median=float(scores[:,:old_count].max(1).values.median()),
                new_max_median=float(scores[:,old_count:].max(1).values.median()))


def run(rt,job,k,n,device):
    require(job.get('methods')==['mopc_hr'] and job.get('save_B_for_mopc') and [k,n] in job['points'],
            'Expected saved B task')
    require(job.get('mopc_bridge','original')=='original',
            'Orthogonal diagnostic requires original bridge B, not affine MoPC bridge')
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cap=Path(rt['data_root'])/job['capsule']
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    old,new=specs[k,0],specs[k,n]
    require(old['support_indices']==new['support_indices'][:6*k],'Support identity differs')
    b_seed=seed_for(job['seed'],job['capsule'],k,'B');c_seed=seed_for(job['seed'],job['capsule'],k,n,'C')
    bpath=Path(job['output'])/f'B-K{k}.pt'
    payload=load_b_state(bpath,job,k,old['split_id'],b_seed,device);base=payload['anchor']
    ox,oy=support_only_clean(rt['data_root'],rt['clean_root'],job['capsule'],old['split_id'],new_only=False)
    nx,ny=support_only_clean(rt['data_root'],rt['clean_root'],job['capsule'],new['split_id'])
    ox=to_tensor(ox,device,np.float32);oy=to_tensor(oy,device,np.int64).long()
    nx=to_tensor(nx,device,np.float32);ny=to_tensor(ny,device,np.int64).long()
    with torch.no_grad():
        old_out=base.model(ox);new_out=base.model(nx)
        zold=F.normalize(old_out['features'],dim=1);znew=new_out['features']
        prototypes,ids=compute_class_prototypes(znew,ny)
        original=F.normalize(base.model.fc.weight.detach(),dim=1)
        initial=F.normalize(prototypes,dim=1)
        b_correct=int((old_out['tx_logits'].argmax(1)==oy).sum())
    results=[]
    for variant in ('affine_weight_cosine','support_prototype_cosine'):
        anchor=copy.deepcopy(base);anchor.method='orthogonal'
        weights=base.model.fc.weight.detach() if variant=='affine_weight_cosine' else base.prototypes.detach()
        anchor.model.classifier_weight=nn.Parameter(weights.clone(),requires_grad=False)
        init_scores=zold @ torch.cat((F.normalize(weights,dim=1),initial),dim=0).t()
        before=summarize_scores(init_scores,oy,6)
        state=register(anchor,nx,ny,seed=c_seed)
        with torch.no_grad():after=summarize_scores(state.score(ox),oy,6)
        losses=[float(r['loss']) for r in state.loss_trace]
        require(len(losses)==50 and all(np.isfinite(losses)),'Invalid registration loss trace')
        results.append(dict(variant=variant,old_support_before=before,old_support_after=after,
                            loss_first=losses[0],loss_last=losses[-1],updates=len(losses)))
    print(json.dumps(dict(event='ORTHOGONAL_SUPPORT_DIAGNOSTIC',profile=job['profile'],capsule=job['capsule'],
        seed=job['seed'],k=k,n=n,B_state_ref=str(bpath),saved_bridge_old_support_correct=b_correct,
        B_old_support_count=len(oy),variants=results,query_read=False,truth_read=False,
        prediction_artifacts_written=False),allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--job',required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    a=p.parse_args();run(read(a.runtime),read(a.job),a.k,a.n,a.device)


if __name__=='__main__':main()
