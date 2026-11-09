"""Trace support-only MoPC from an actual saved B anchor, without predictions."""
import argparse
from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .io import activate, read, require
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group2_b_state import load_b_state
from .group2_c_backbone_worker import support_only
from .group2_dadda_mopc_diagnostic import trace_registration


@contextmanager
def observe_support_forward(emit):
    """Record first observed leaf-module nonfinite boundary; return values unchanged."""
    from .vendor_native.comparison_suite import registration as native
    original = native._logits
    found = False
    recent = deque(maxlen=8)

    def inspect(layer, boundary, tensor):
        nonlocal found
        if found or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return
        data = tensor.detach()
        if not bool(torch.isfinite(data).all()):
            found = True
            emit(dict(event='FIRST_FORWARD_NONFINITE', layer=layer, boundary=boundary, recent=list(recent)))
        elif data.numel():
            recent.append(dict(layer=layer, boundary=boundary, absmax=float(data.abs().max())))

    def wrapped(model, method, x):
        handles = []
        for name, module in model.named_modules():
            if any(module.children()):
                continue
            def pre(_module, args, layer=name):
                for value in args:
                    inspect(layer, 'input', value)
            def post(current, _args, result, layer=name):
                inspect(layer, 'output', result)
                for key, value in current.named_buffers(recurse=False):
                    inspect(layer, 'buffer.'+key, value)
            handles.extend((module.register_forward_pre_hook(pre), module.register_forward_hook(post)))
        try:
            return original(model, method, x)
        finally:
            for handle in handles:
                handle.remove()

    with patch.object(native, '_logits', wrapped):
        yield


def run(rt, job, k, n, device, emit):
    """Load only frozen B and registered support; never load query/truth."""
    require(job['profile'] in ('protonet_cda','dadda_sda') and job['methods']==['mopc_hr'] and
            job.get('load_B_for_mopc') and [k,n] in [list(p) for p in job['points']], 'Wrong diagnostic task')
    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from .group2_schedule import smoke
    smoke_result=smoke(rt)
    require(smoke_result['status']=='PASS' and smoke_result['query_rows']==0 and
            smoke_result['truth_read'] is False, 'Source-only smoke failed')
    emit(dict(event='SOURCE_SMOKE_PASS',profiles=len(smoke_result['profiles']),query_rows=0,truth_read=False))
    cap=Path(rt['data_root'])/job['capsule']
    manifest=read(cap/'manifest.json')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE',
            'Wrong capsule')
    lookup={(v['k'],v['new_count']):v for v in (read(p) for p in (cap/'splits').glob('*.json'))}
    old_id,new_id=lookup[k,0]['split_id'],lookup[k,n]['split_id']
    _,_,old_positions,_=support_only(cap,old_id)
    x,y,positions,_=support_only(cap,new_id)
    mask=y<6
    require(np.array_equal(old_positions,positions[mask]), 'B/C old support positions differ')
    b_seed=seed_for(int(job['seed']),job['capsule'],k,'B')
    c_seed=seed_for(int(job['seed']),job['capsule'],k,n,'C')
    b_path=Path(job['load_B_for_mopc'])/f'B-K{k}.pt'
    anchor=load_b_state(b_path,job,k,old_id,b_seed,device)['anchor']
    new_x=to_tensor(x[~mask],device,np.float32)
    new_y=to_tensor(y[~mask],device,np.int64).long()
    emit(dict(event='SAVED_B_SUPPORT_READY',key=job['key'],k=k,n=n,b_seed=b_seed,c_seed=c_seed,
              query_rows=0,truth_read=False))
    with observe_support_forward(emit):
        trace_registration({'mopc_hr':anchor},new_x,new_y,c_seed,emit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--k',type=int,required=True)
    parser.add_argument('--n',type=int,required=True)
    parser.add_argument('--device',choices=('cuda:0',),required=True)
    args=parser.parse_args()
    run(read(args.runtime),read(args.job),args.k,args.n,args.device,
        lambda event: print(json.dumps(event,allow_nan=False),flush=True))
