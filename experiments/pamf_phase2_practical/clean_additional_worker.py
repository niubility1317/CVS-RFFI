"""Clean direct comparisons for Tweak M1 and repaired long-budget ISSL."""
import copy
from pathlib import Path
import time

import numpy as np
import torch

from . import external_additional as extra
from .episodic_experiment import split_rows
from .episodic_io import load_task
from .external_execution import infer, seed_for, source_runtime
from .features import tensor
from .io import activate, read, require, write


def direct_issl(bundle, new_x, new_y, seed, logger=None):
    """Original-source anchor to fixed-v1 C, without target-old adaptation."""
    require(bundle['profile']=='issl_fixed_v1', 'Expected repaired ISSL')
    extra._seed(seed)
    model=copy.deepcopy(bundle['model']).requires_grad_(True)
    count=6+len(torch.unique(new_y))
    require(tuple(torch.unique(new_y,sorted=True).tolist())==tuple(range(6,count)), 'Invalid new classes')
    extra._issl_expand(model,count)
    replay=bundle['native_state'];rx,ry=replay['source_replay_x'],replay['source_replay_y']
    transfer_x=torch.cat((rx,new_x.detach().cpu())); transfer_y=torch.cat((ry,new_y.detach().cpu()))
    spec=extra.source_training_spec('issl_fixed_v1')
    require(spec['C_SSL_epochs']==2000 and spec['C_transfer_epochs']==300, 'Wrong ISSL long budget')
    extra._issl_ssl(model,torch.cat((rx,transfer_x)),6,spec,seed,logger)
    extra._train_ce(model,transfer_x,transfer_y,spec['C_transfer_epochs'],128,spec['C_transfer_lr'],seed,logger,
                    crop=218,stage='C_transfer')
    return dict(model=model.eval(),score_fn=extra.native_logits,classes=tuple(range(count)),profile='issl_fixed_v1',
                resources=dict(C_initialization='A_no_B',C_SSL_epochs=2000,C_transfer_epochs=300,
                               source_replay_count=len(rx),actual_target_old_labels=0,
                               actual_target_new_labels=len(new_y)))


def run(rt,job):
    require(rt.get('clean_root') and rt.get('data_root'), 'Clean and parent roots required')
    profile=job['profile'];require(profile in ('tweak_m1','issl_fixed_v1'), 'Unexpected method')
    points=sorted(set(tuple(map(int,p)) for p in job['points']))
    require(points and all(k in (1,5,10,20) and n in (2,5,10,20) for k,n in points), 'Invalid workpoints')
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle=extra.load_source(source_runtime(rt),profile,job['seed'],'cuda:0')
    from .clean_source_smoke import check
    check(rt,bundle)
    model=bundle['model'].eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    device=next(model.parameters()).device
    cap=Path(rt['data_root'])/job['capsule'];manifest=read(cap/'manifest.json')
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    require(all((k,0) in specs and (k,n) in specs for k,n in points),'Missing split')
    root=Path(job['output']);root.mkdir(parents=True,exist_ok=False);rows=[];start=time.perf_counter()
    def get(s,stage):
        return load_task(cap,s['split_id'],stage=stage,clean_root=rt['clean_root'],task_root=rt['data_root'])
    def save(s,stage,task,state):
        # M1 is a physical query decision of exactly one IQ record per call.
        batch=1 if profile=='tweak_m1' else rt.get('batch_size',128)
        pred,old=infer(state,task['query_x'],len(s['registered_classes']),batch=batch)
        q=split_rows(cap,s);require(len(q)==len(pred),'Query rows mismatch')
        name=f'{stage}-K{s["k"]}-N{s["new_count"]}.npz'
        payload=dict(query_indices=q,predicted_indices=pred)
        if s['new_count']:payload['old_restricted_indices']=old
        np.savez_compressed(root/name,**payload)
        rows.append(dict(split_id=s['split_id'],stage=stage,arm=profile,input_path='S0Q0',
            input_condition='clean_ground_pre_satellite_normalized',prediction=name,k=s['k'],
            new_count=s['new_count'],registered_classes=s['registered_classes'],query_count=len(q),
            query_records_per_decision=batch if profile=='tweak_m1' else 1))
    for k in sorted({k for k,n in points}):
        s=specs[k,0];task=get(s,'A');save(s,'A0',task,bundle)
        if profile=='tweak_m1':
            task=get(s,'B');sx=tensor(task['support_x'],device)
            sy=torch.tensor(task['support_y'].tolist(),device=device,dtype=torch.long)
            state=extra.fit_B(rt,bundle,sx,sy,seed_for(job['seed'],job['capsule'],k,'B'))
            save(s,'B',task,state)
    if profile=='issl_fixed_v1':
        for k,n in points:
            s=specs[k,n];task=get(s,'C');mask=task['support_y']>=6
            x=tensor(task['support_x'][mask],device)
            y=torch.tensor(task['support_y'][mask].tolist(),device=device,dtype=torch.long)
            state=direct_issl(bundle,x,y,seed_for(job['seed'],job['capsule'],k,n,'C'))
            save(s,'C',task,state)
    require(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'A source anchor changed')
    write(root/'predictions.json',dict(capsule_id=manifest['capsule_id'],capsule=job['capsule'],
          profile=profile,model_seed=job['seed'],rows=rows))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],
          profile=profile,model_seed=job['seed'],rows=len(rows),truth_read=False,query_fit=False,
          source_weights_unchanged=True,seconds=time.perf_counter()-start))
