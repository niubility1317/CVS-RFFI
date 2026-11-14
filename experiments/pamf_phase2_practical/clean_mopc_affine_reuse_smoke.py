"""Smoke-test the production affine bridge from an actual retained B state."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .clean_affine_b_reuse import prior_b
from .clean_mopc_diagnostic import support_only_clean
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group1_source import embedding
from .group2_b_state import save_b_state
from .group2_cross_fit import prepare_b, register_c
from .group2_proto_mopc_affine import require_equivalent_b
from .io import activate, read, require


def run(rt, old_job, old_job_file, old_runtime_file, output, k, n, device):
    require(old_job.get('profile') == 'protonet_cda' and old_job.get('methods') == ['mopc_hr'],
            'Expected ProtoNet MoPC job')
    activate(rt); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    job = dict(old_job, output=output, mopc_bridge='proto_affine_v1',
               reuse_original_b=dict(job=old_job_file, runtime=old_runtime_file))
    cap = Path(rt['data_root'])/job['capsule']
    lookup={(v['k'],v['new_count']):v for v in (read(p) for p in (cap/'splits').glob('*.json'))}
    old, new = lookup[k,0], lookup[k,n]
    require(old['support_indices']==new['support_indices'][:6*k], 'Wrong support binding')
    bs=seed_for(job['seed'],job['capsule'],k,'B'); cs=seed_for(job['seed'],job['capsule'],k,n,'C')
    source=prior_b(job,rt,k,old['split_id'],bs,device)
    require(source is not None, 'Expected retained B state')
    payload=source['payload']; model=payload['model']; prototypes=payload['old_prototypes']
    ox,oy=support_only_clean(rt['data_root'],rt['clean_root'],job['capsule'],old['split_id'],new_only=False)
    nx,ny=support_only_clean(rt['data_root'],rt['clean_root'],job['capsule'],new['split_id'])
    ox=to_tensor(ox,device,np.float32);oy=to_tensor(oy,device,np.int64).long()
    nx=to_tensor(nx,device,np.float32);ny=to_tensor(ny,device,np.int64).long()
    extractor=lambda m,x:embedding(m,'protonet',x)
    prepared=prepare_b(model,'protonet_cda',extractor,ox,oy,old_prototypes=prototypes,
                       mopc_bridge='proto_affine_v1')
    with torch.no_grad():
        actual=-torch.cdist(extractor(model,ox),prototypes).square()
        require_equivalent_b(actual,prepared['mopc_hr'].model(ox)['tx_logits'])
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    save_b_state(root/f'B-K{k}.pt',job,k,old['split_id'],bs,model,prepared['mopc_hr'],prototypes)
    state=register_c(prepared,nx,ny,seed=cs,methods=('mopc_hr',))['mopc_hr']
    require(all(bool(torch.isfinite(p).all()) for p in state.model.parameters()),'Nonfinite candidate')
    print(json.dumps(dict(event='AFFINE_REUSE_SMOKE_COMPLETE',source_B=source['path'],k=k,n=n,
        updates=len(state.loss_trace),bridge='proto_affine_v1',query_read=False,truth_read=False,
        predictions_written=False),allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','job','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--device',choices=['cuda:0'],required=True)
    a=p.parse_args();run(read(a.runtime),read(a.job),a.job,a.runtime,a.output,a.k,a.n,a.device)


if __name__=='__main__':main()
