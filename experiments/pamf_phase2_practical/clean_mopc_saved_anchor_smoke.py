"""Verify a versioned MoPC run can reuse its exact saved B without query."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .clean_mopc_diagnostic import support_only_clean
from .clean_mopc_savedb_diagnostic import require_finite_registration
from .clean_saved_anchor_reuse import prior_anchor
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group2_cil_bridge import register
from .io import activate,read,require


def run(runtime_file,job_file,k,n,device):
    rt=read(runtime_file);old=read(job_file)
    job=dict(old,mopc_stabilization='clip1000_v1',reuse_saved_b=dict(job=job_file,runtime=runtime_file))
    require([k,n] in job['points'] and job.get('methods')==['mopc_hr'],'Wrong saved-anchor smoke task')
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cap=Path(rt['data_root'])/job['capsule'];splits=[read(x) for x in (cap/'splits').glob('*.json')]
    old_split=[x for x in splits if x['k']==k and x['new_count']==0]
    new_split=[x for x in splits if x['k']==k and x['new_count']==n]
    require(len(old_split)==len(new_split)==1 and old_split[0]['support_indices']==new_split[0]['support_indices'][:6*k],
            'Saved-anchor smoke support identity differs')
    b_seed=seed_for(job['seed'],job['capsule'],k,'B')
    prior=prior_anchor(job,rt,k,old_split[0]['split_id'],b_seed,device)
    require(prior is not None,'Saved-anchor smoke has no B state')
    x,y=support_only_clean(rt['data_root'],rt['clean_root'],job['capsule'],new_split[0]['split_id'])
    x=to_tensor(x,device,np.float32);y=to_tensor(y,device,np.int64).long()
    state=register(prior['payload']['anchor'],x,y,seed=seed_for(job['seed'],job['capsule'],k,n,'C'),
                   mopc_semantics='public_code',mopc_max_grad_norm=1000.)
    require_finite_registration(state)
    print(json.dumps(dict(event='SAVED_ANCHOR_SMOKE_PASS',profile=job['profile'],k=k,n=n,
        B_state_ref=prior['path'],updates=len(state.loss_trace),
        clipped_updates=state.protocol_metadata['gradient_clipped_updates'],
        query_read=False,truth_read=False,predictions_written=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--job',required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--device',choices=['cuda:0'],required=True)
    a=p.parse_args();run(a.runtime,a.job,a.k,a.n,a.device)


if __name__=='__main__':main()
