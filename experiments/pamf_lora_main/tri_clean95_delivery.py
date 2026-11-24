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
from .tri_clean95 import ROWS, RUN, SEED


def publish(repo, output, context, ssh_config):
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    project=PurePosixPath(rt['project'])
    parent=project/'runs'/DATA_RUN
    native=project/'releases'/f'{DATA_RUN}-release'/'data'
    rt.update(run_id=RUN,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.tri_clean95',
              release_root=str(project/'releases'/f'{RUN}-release'),run_root=str(project/'runs'/RUN),
              parent_root=str(parent),source_root=str(native/'source'),
              native_target_root=str(native/'target'),target_views=str(native/'target'),
              allowed_gpus=[3,5,6,7,0,1,2,4],max_workers=6,max_compute_processes=12)
    release(repo,output,rt,ssh_config)


def frozen_predictions(output,names):
    if not names or names[0]!='control' or len(set(names))!=len(names) or not set(names)<=ROWS.keys():
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
    args=vars(parser.parse_args())
    kind=args.pop('kind')
    {'publish':publish,'score':score}[kind](**args)
