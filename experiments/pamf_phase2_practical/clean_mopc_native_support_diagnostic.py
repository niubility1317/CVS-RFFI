"""Run a source-anchored native MoPC registration on support only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .clean_mopc_diagnostic import support_only_clean
from .clean_source_smoke import check
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group1 import source_scope
from .group1_source import load_source
from .group1_target import fit_C
from .io import activate, read, require


def run(rt, job, k, n, device):
    require(job.get('profile')=='mopc_hr' and [k,n] in job.get('points',[]), 'Wrong native diagnostic job')
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle=load_source(source_scope(rt),'mopc_hr',job['seed'],device);check(rt,bundle)
    cap=Path(rt['data_root'])/job['capsule']
    matches=[read(p) for p in (cap/'splits').glob('*.json')]
    selected=[s for s in matches if s['k']==k and s['new_count']==n]
    require(len(selected)==1,'Wrong native support split')
    x,y=support_only_clean(rt['data_root'],rt['clean_root'],job['capsule'],selected[0]['split_id'])
    x=to_tensor(x,device,np.float32);y=to_tensor(y,device,np.int64).long()
    seed=seed_for(job['seed'],job['capsule'],k,n,'C')
    result=fit_C(rt,bundle,x,y,tuple(range(6,6+n)),seed,None,mopc_max_grad_norm=1000.)
    state=result['native_state']
    require(all(bool(torch.isfinite(v).all()) for v in state.model.state_dict().values()),'Nonfinite final state')
    print(json.dumps(dict(event='DIAGNOSTIC_RESULT',status='COMPLETE',profile='native_mopc',k=k,n=n,
        updates=len(state.loss_trace),clipped_updates=state.protocol_metadata['gradient_clipped_updates'],
        optimizer_changed=True,query_read=False,truth_read=False,predictions_written=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','job'):p.add_argument('--'+name,required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--device',choices=['cuda:0'],required=True)
    a=p.parse_args();run(read(a.runtime),read(a.job),a.k,a.n,a.device)


if __name__=='__main__':main()
