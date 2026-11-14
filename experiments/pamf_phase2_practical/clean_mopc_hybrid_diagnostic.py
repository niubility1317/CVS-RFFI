"""Check clean B-to-MoPC fitting with source/support only, without query access."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .io import activate, read, require
from .clean_mopc_diagnostic import support_only_clean
from .clean_source_smoke import check
from .external_execution import seed_for,source_runtime
from .fast_transfer import to_tensor
from .group1 import row,source_scope
from .group1_source import load_source,embedding
from .group1_target import fit_B
from .group2_cross_fit import prepare_b,register_c
from . import external_additional as extra


def run(rt,profile,capsule,seed,k,n,device,*,mopc_bridge='original'):
    require(profile in ('protonet_cda','dadda_sda','tweak_m1'), 'Unsupported diagnostic profile')
    require(mopc_bridge=='original' or (mopc_bridge=='proto_affine_v1' and profile=='protonet_cda'),
            'Unsupported diagnostic bridge')
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if profile=='tweak_m1':bundle=extra.load_source(source_runtime(rt),profile,seed,device)
    else:
        source=source_scope(rt);source['run_root']=rt['source_run_root']
        bundle=load_source(source,profile,seed,device)
    check(rt,bundle)
    cap=Path(rt['data_root'])/capsule
    matches={(v['k'],v['new_count']):v for v in (read(p) for p in (cap/'splits').glob('*.json'))}
    old,new=matches[k,0],matches[k,n]
    require(old['support_indices']==new['support_indices'][:6*k], 'B/C old support differs')
    ox,oy=support_only_clean(rt['data_root'],rt['clean_root'],capsule,old['split_id'],new_only=False)
    nx,ny=support_only_clean(rt['data_root'],rt['clean_root'],capsule,new['split_id'])
    ox=to_tensor(ox,device,np.float32);oy=to_tensor(oy,device,np.int64).long()
    nx=to_tensor(nx,device,np.float32);ny=to_tensor(ny,device,np.int64).long()
    b_seed=seed_for(seed,capsule,k,'B');c_seed=seed_for(seed,capsule,k,n,'C')
    if profile=='tweak_m1':
        b=extra.fit_B(rt,bundle,ox,oy,b_seed)
        prepared=prepare_b(b['model'],profile,extra.native_features,ox,oy,b_scores=b)
    else:
        b=fit_B(rt,bundle,ox,oy,b_seed,None);native=b.get('native_state')
        prototypes=getattr(native,'prototypes',None) if profile=='protonet_cda' else None
        extractor=lambda model,x:embedding(model,row(profile)['method'],x)
        prepared=prepare_b(b['model'],profile,extractor,ox,oy,old_prototypes=prototypes,b_scores=b.get('score_fn'),
                           mopc_bridge=mopc_bridge)
        if mopc_bridge=='proto_affine_v1':
            from .group2_proto_mopc_affine import require_equivalent_b
            with torch.no_grad():
                require_equivalent_b(b['score_fn'](b['model'],ox),prepared['mopc_hr'].model(ox)['tx_logits'])
    state=register_c(prepared,nx,ny,seed=c_seed,methods=('mopc_hr',))['mopc_hr']
    require(all(bool(torch.isfinite(p).all()) for p in state.model.parameters()),'Nonfinite registered model')
    print(json.dumps(dict(event='HYBRID_SUPPORT_COMPLETE',profile=profile,capsule=capsule,seed=seed,k=k,n=n,
        b_seed=b_seed,c_seed=c_seed,updates=len(state.loss_trace),semantics='public_code',
        mopc_bridge=mopc_bridge,gradient_clipping=False,query_read=False,truth_read=False,
        predictions_written=False),allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','profile','capsule'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--k',type=int,required=True)
    p.add_argument('--n',type=int,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    p.add_argument('--mopc-bridge',choices=['original','proto_affine_v1'],default='original')
    a=p.parse_args();run(read(a.runtime),a.profile,a.capsule,a.seed,a.k,a.n,a.device,mopc_bridge=a.mopc_bridge)


if __name__=='__main__':main()
