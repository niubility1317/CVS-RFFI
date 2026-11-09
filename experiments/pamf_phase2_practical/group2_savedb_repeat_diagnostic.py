"""Compare two deterministic source/support-only MoPC traces from one saved B."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .io import activate, read, require
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group2_b_state import load_b_state
from .group2_c_backbone_worker import support_only
from .group2_dadda_mopc_diagnostic import trace_registration
from .group2_savedb_diagnostic import observe_support_forward


def run(rt, job, k, n, device, emit):
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8',':16:8'), 'Deterministic CUDA workspace missing')
    require(job['profile'] in ('protonet_cda','dadda_sda') and job['methods']==['mopc_hr'] and
            job.get('load_B_for_mopc') and [k,n] in [list(p) for p in job['points']], 'Wrong diagnostic task')
    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    from .group2_schedule import smoke
    result=smoke(rt)
    require(result['status']=='PASS' and result['query_rows']==0 and result['truth_read'] is False,
            'Source-only smoke failed')
    emit(dict(event='SOURCE_SMOKE_PASS',profiles=len(result['profiles']),query_rows=0,truth_read=False))
    cap=Path(rt['data_root'])/job['capsule']
    manifest=read(cap/'manifest.json')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE', 'Wrong capsule')
    splits={(v['k'],v['new_count']):v for v in (read(p) for p in (cap/'splits').glob('*.json'))}
    old_id,new_id=splits[k,0]['split_id'],splits[k,n]['split_id']
    _,_,old_positions,_=support_only(cap,old_id)
    x,y,positions,_=support_only(cap,new_id)
    mask=y<6
    require(np.array_equal(old_positions,positions[mask]), 'B/C old support positions differ')
    b_seed=seed_for(int(job['seed']),job['capsule'],k,'B')
    c_seed=seed_for(int(job['seed']),job['capsule'],k,n,'C')
    anchor=load_b_state(Path(job['load_B_for_mopc'])/f'B-K{k}.pt',job,k,old_id,b_seed,device)['anchor']
    x=to_tensor(x[~mask],device,np.float32);y=to_tensor(y[~mask],device,np.int64).long()
    emit(dict(event='SAVED_B_SUPPORT_READY',key=job['key'],k=k,n=n,b_seed=b_seed,c_seed=c_seed,
              query_rows=0,truth_read=False))
    results=[];states=[]
    for repetition in (1,2):
        events=[]
        try:
            with observe_support_forward(events.append):
                state=trace_registration({'mopc_hr':anchor},x,y,c_seed,events.append)
            status='COMPLETE';error=None;states.append(state)
        except FloatingPointError as exc:
            status='NONFINITE';error=str(exc);states.append(None)
        result=dict(repetition=repetition,status=status,error=error,observations=events)
        results.append(result)
        emit(dict(event='REPETITION_DONE',**result))
    exact=None
    if all(state is not None for state in states):
        first,second=states
        exact=(all(torch.equal(value,second.model.state_dict()[key]) for key,value in first.model.state_dict().items())
               and torch.equal(first.prototypes,second.prototypes))
    emit(dict(event='DIAGNOSTIC_COMPARE',statuses=[r['status'] for r in results],state_exact=exact,
              query_rows=0,truth_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--k',type=int,required=True);parser.add_argument('--n',type=int,required=True)
    parser.add_argument('--device',choices=('cuda:0',),required=True)
    args=parser.parse_args()
    run(read(args.runtime),read(args.job),args.k,args.n,args.device,
        lambda event: print(json.dumps(event,allow_nan=False),flush=True))
