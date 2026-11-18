"""Tweak M1 B followed by compatible, explicitly named registration cores."""
from pathlib import Path
import time

import numpy as np
import torch

from . import external_additional as extra
from .clean_source_smoke import check_worker
from .episodic_experiment import split_rows
from .episodic_io import load_task
from .external_execution import seed_for,source_runtime,plain
from .features import tensor
from .group2_cross_fit import prepare_b,register_c
from .group2_worker import method_arm, score_fixed
from .io import activate,read,require,write


def run(rt,job):
    methods=job.get('methods',[])
    from .mopc_stabilization import policy
    max_grad_norm=policy(job,'hybrid');stabilization=job.get('mopc_stabilization','none')
    require(job.get('profile')=='tweak_m1' and methods and len(set(methods))==len(methods)
            and set(methods)<={'lora','orthogonal','mopc_hr'},'Tweak registration method is unavailable')
    require(not job.get('save_B_for_mopc') or methods==['mopc_hr'], 'Invalid saved-B method subset')
    points=sorted(set(tuple(map(int,p)) for p in job['points']))
    require(points and all(k in (1,5,10,20) and n in (2,5,10,20) for k,n in points),'Wrong workpoints')
    activate(rt);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle=extra.load_source(source_runtime(rt),'tweak_m1',job['seed'],'cuda:0');check_worker(rt,bundle,job)
    model=bundle['model'];before={k:v.detach().clone() for k,v in model.state_dict().items()};device=next(model.parameters()).device
    cap=Path(rt['data_root'])/job['capsule'];manifest=read(cap/'manifest.json')
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    require(all((k,0) in specs and (k,n) in specs for k,n in points),'Missing split')
    output=Path(job['output']);output.mkdir(parents=True,exist_ok=False);rows=[];start=time.perf_counter()
    def get(spec,stage):
        if rt.get('input_condition')=='satellite_received':
            require(not rt.get('clean_root'),'Satellite task cannot use clean input')
            return load_task(cap,spec['split_id'],stage=stage)
        return load_task(cap,spec['split_id'],stage=stage,clean_root=rt['clean_root'],task_root=rt['data_root'])
    for k in sorted({k for k,n in points}):
        old=get(specs[k,0],'B');x=tensor(old['support_x'],device)
        y=torch.tensor(old['support_y'].tolist(),device=device,dtype=torch.long)
        b_seed=seed_for(job['seed'],job['capsule'],k,'B')
        prior=None
        if job.get('reuse_author_b'):
            from .mopc_author_b_reuse import pilot_b
            prior=pilot_b(job,rt,k,specs[k,0]['split_id'],specs[5,5]['split_id'],b_seed,str(device))
        if job.get('reuse_saved_b'):
            require(prior is None,'Conflicting B reuse sources')
            from .clean_saved_anchor_reuse import prior_anchor
            prior=prior_anchor(job,rt,k,specs[k,0]['split_id'],b_seed,str(device))
        if prior is None:
            b=extra.fit_B(rt,bundle,x,y,b_seed)
            prepared=prepare_b(b['model'],'tweak_m1',extra.native_features,x,y,old_count=6,b_scores=b)
        else:
            b=dict(model=prior['payload']['model'])
            prepared={'mopc_hr':prior['payload']['anchor']}
        if job.get('save_B_for_mopc'):
            from .group2_b_state import save_b_state
            save_b_state(output/f'B-K{k}.pt',job,k,specs[k,0]['split_id'],b_seed,b['model'],prepared['mopc_hr'])
        for _,n in (p for p in points if p[0]==k):
            spec=specs[k,n];task=get(spec,'C');mask=task['support_y']<6
            require(np.array_equal(old['support_ids'],task['support_ids'][mask]),'B/C old support mismatch')
            nx=tensor(task['support_x'][~mask],device)
            ny=torch.tensor(task['support_y'][~mask].tolist(),device=device,dtype=torch.long)
            options={} if max_grad_norm is None else dict(mopc_max_grad_norm=max_grad_norm)
            states=register_c(prepared,nx,ny,seed=seed_for(job['seed'],job['capsule'],k,n,'C'),methods=methods,**options)
            for method,state in states.items():
                pred,restricted=score_fixed(state,task['query_x'],count=6+n,old_count=6,batch_size=1,device=str(device))
                query=split_rows(cap,spec);require(len(query)==len(pred),'Wrong query complement')
                name=f'C-K{k}-N{n}-{method}.npz'
                np.savez_compressed(output/name,query_indices=query,predicted_indices=pred,old_restricted_indices=restricted)
                rows.append(dict(split_id=spec['split_id'],stage='C',arm=method_arm('tweak_m1',method,mopc_stabilization=stabilization),input_path='S0Q0',
                    input_condition=rt.get('input_condition','clean_ground_pre_satellite_normalized'),prediction=name,k=k,new_count=n,
                    registered_classes=spec['registered_classes'],query_count=len(query),query_records_per_decision=1,
                    claim='Tweak_M1_B_to_'+{'lora':'LoRa','orthogonal':'Orthogonal','mopc_hr':'MoPC'}[method]+'_core_transplant',
                    knn_k=state.k if method=='lora' else None,
                    resources=dict(B_state_ref=str(output/f'B-K{k}.pt') if job.get('save_B_for_mopc') else None,
                                   B_original_state_ref=prior['path'] if prior is not None else None,
                                   C_protocol=plain(getattr(state,'protocol_metadata',{})))))
    require(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Source state changed')
    require(len(rows)==len(points)*len(methods),'Incomplete C rows')
    write(output/'predictions.json',dict(capsule_id=manifest['capsule_id'],capsule=job['capsule'],
          profile='tweak_m1',model_seed=job['seed'],rows=rows))
    write(output/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],
          profile='tweak_m1',model_seed=job['seed'],rows=len(rows),truth_read=False,query_fit=False,seconds=time.perf_counter()-start))
