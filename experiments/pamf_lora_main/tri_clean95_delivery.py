"""N607 release and independent clean-only score for LoRa PAMF-Tri."""
import argparse
from pathlib import Path, PurePosixPath
import shlex
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import receive, wait_for_predictions
from .data import CLASSES, TARGET, label_ids, packet_indices, recording
from .full_packet_data import LIMIT
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit
from .scoring import macro_f1
from .tri_clean95 import (
    BUDGET_RUN, DRIFT_RUN, NOISE_RUN, QUICK_RUN, ROWS, RUN, SCRATCH_ROWS, SCRATCH_RUN, SEED, TIMEFREQ_RUN, TRANSFER_RUN,
)


def publish(repo, output, context, ssh_config, quick=False, transfer=False, scratch=False, timefreq=False,
            noise=False,drift=False,budget=False):
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    project=PurePosixPath(rt['project'])
    parent=project/'runs'/DATA_RUN
    native=project/'releases'/f'{DATA_RUN}-release'/'data'
    studies=((budget,BUDGET_RUN),(drift,DRIFT_RUN),(noise,NOISE_RUN),(timefreq,TIMEFREQ_RUN),(scratch,SCRATCH_RUN),
             (transfer,TRANSFER_RUN),(quick,QUICK_RUN))
    name=next((name for enabled,name in studies if enabled),RUN)
    rt.update(run_id=name,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.tri_clean95',
              release_root=str(project/'releases'/f'{name}-release'),run_root=str(project/'runs'/name),
              parent_root=str(parent),source_root=str(native/'source'),
              native_target_root=str(native/'target'),target_views=str(native/'target'),
              allowed_gpus=[3,5,6,7,0,1,2,4],max_workers=6,max_compute_processes=12)
    if scratch or timefreq or noise or drift or budget:
        rt.update(scratch=True,fixed_clean_cache=str(project/'runs'/RUN/'clean_cache'),
                  max_workers=4)
    if timefreq:
        del rt['native_target_root'],rt['target_views']
        rt.update(timefreq=True,allowed_gpus=[0,5,4,6,2,1,7],max_workers=2,min_free_mib=10000)
    if noise:
        del rt['native_target_root'],rt['target_views']
        rt.update(noise_study=True,allowed_gpus=[2,4,1,7],max_workers=2,min_free_mib=10000)
    if drift:
        del rt['native_target_root'],rt['target_views'],rt['fixed_clean_cache']
        rt.update(drift_study=True,allowed_gpus=[2,4,1],max_workers=2,min_free_mib=9000)
    if budget:
        del rt['native_target_root'],rt['target_views']
        rt.update(budget_study=True,allowed_gpus=[5,0,6,4],max_workers=2,min_free_mib=9500)
    if quick or transfer:
        del rt['native_target_root'],rt['target_views']
        rt.update(quick=quick,transfer=transfer,fixed_clean_cache=str(project/'runs'/RUN/'clean_cache'))
        if transfer:
            rt.update(allowed_gpus=[3,6,5,7,0,2,4,1],max_workers=4,min_free_mib=12288)
        else:
            rt.update(allowed_gpus=[0,2,4,1],max_workers=4,min_free_mib=8000)
    release(repo,output,rt,ssh_config)


def frozen_predictions(output,names):
    if not names or names[0]!='control' or len(set(names))!=len(names) or not set(names)<=set(ROWS)|set(SCRATCH_ROWS):
        raise ValueError('Invalid clean-only source selection')
    result={}
    for name in names:
        folder=output/'predictions'/name
        if read(folder/'complete.json')!=dict(method=name,seed=SEED,status='PREDICTIONS_FROZEN',
                target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=['clean']):
            raise ValueError('Incomplete clean-only prediction declaration')
        chunks=[]
        for rx in TARGET:
            pred=np.load(folder/f'{rx}__clean__direct.npy',allow_pickle=False)
            if pred.shape!=(LIMIT,) or pred.dtype.kind not in 'iu' or (pred<0).any() or (pred>=len(CLASSES)).any():
                raise ValueError(f'Incomplete clean prediction: {name}/{rx}')
            chunks.append(pred)
        result[name]=np.concatenate(chunks)
    return result


def score(submit, ssh_config, dataset, output):
    root=shlex.quote(read(submit)['run_root'])
    conn=['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    state=wait_for_predictions(conn,root)
    output.mkdir(parents=True,exist_ok=False)
    receive(conn,root,output,main=False)
    preds=frozen_predictions(output,state['names'])
    truth=[]
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i,rx in enumerate(TARGET):
            with recording(archive,f'{rx}_test.h5',output) as h5:
                truth.append(label_ids(h5,LIMIT)[packet_indices(LIMIT,i)])
    truth=np.concatenate(truth)
    result={}
    base=preds['control']==truth
    for name,p in preds.items():
        ok=p==truth
        result[name]=dict(n=len(ok),correct=int(ok.sum()),accuracy=float(ok.mean()),
            macro_f1=macro_f1(truth,p),repairs=int((ok&~base).sum()),harms=int((base&~ok).sum()),
            by_rx=dict(zip(TARGET,ok.reshape(len(TARGET),LIMIT).mean(1).tolist(),strict=True)),
            by_tx={str(tx):float(ok[truth==i].mean()) for i,tx in enumerate(CLASSES)})
    write(output/'scores.json',dict(status='SCORED_COMPLETE',views=['clean'],truth_last=True,seed=SEED,results=result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    actions=parser.add_subparsers(dest='kind',required=True)
    for command, fields in [('publish',('repo','output','context','ssh-config')),
                            ('score',('submit','output','dataset','ssh-config'))]:
        p=actions.add_parser(command)
        for k in fields:p.add_argument('--'+k,type=Path,required=True)
        if command=='publish':
            study=p.add_mutually_exclusive_group()
            study.add_argument('--quick',action='store_true')
            study.add_argument('--transfer',action='store_true')
            study.add_argument('--scratch',action='store_true')
            study.add_argument('--timefreq',action='store_true')
            study.add_argument('--noise',action='store_true')
            study.add_argument('--drift',action='store_true')
            study.add_argument('--budget',action='store_true')
    args=vars(parser.parse_args())
    kind=args.pop('kind')
    {'publish':publish,'score':score}[kind](**args)
