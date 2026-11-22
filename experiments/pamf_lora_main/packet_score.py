"""Independent packet scoring after all corrected-readout predictions freeze."""
import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import wait_for_predictions
from .data import CLASSES, SOURCE, TARGET, VIEWS, label_ids, packet_indices, recording
from .packet_data import TARGET_LIMIT
from .packet_models import SEED
from .packet_runner import READOUTS
from .scoring import macro_f1


def load_matrix(root, readouts=READOUTS):
    predictions={}
    for name,modes in readouts.items():
        path=root/name
        state=read(path/'complete.json')
        if (state['status']!='PREDICTIONS_FROZEN' or state['target_truth_read'] or state['seed']!=SEED
                or state['receivers']!=list(TARGET) or state['views']!=list(VIEWS)
                or state['modes']!=list(modes) or state['per_receiver']!=TARGET_LIMIT):
            raise ValueError('Packet score input matrix differs')
        for v in VIEWS:
            for rx in TARGET:
                for mode in modes:
                    x=np.load(path/f'{rx}__{v}__{mode}.npy',allow_pickle=False)
                    if x.shape!=(TARGET_LIMIT,) or x.dtype.kind not in 'iu' or x.min()<0 or x.max()>=len(CLASSES):
                        raise ValueError('Packet predictions incomplete')
                    predictions[name,mode,v,rx]=x
    return predictions


def score(dataset, output, readouts=READOUTS):
    preds=load_matrix(output/'predictions',readouts)
    truth={}
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as f:
        for i,r in enumerate(TARGET):
            with recording(f,f'{r}_test.h5',output) as h5:
                truth[r]=label_ids(h5,2000)[packet_indices(2000,i)[:TARGET_LIMIT]]
    target=np.concatenate([truth[rx] for rx in TARGET])
    results=[]
    for method,modes in readouts.items():
        for mode in modes:
            for view in VIEWS:
                rows={r:preds[method,mode,view,r] for r in TARGET}
                a=np.concatenate(list(rows.values()))
                ok=a==target
                base=np.concatenate([preds[method,'direct',view,r] for r in TARGET])==target
                results.append(dict(method=method,readout=mode,view=view,n=len(target),
                    accuracy=float(ok.mean()),macro_f1=macro_f1(target,a),
                    by_rx={r:float((x==truth[r]).mean()) for r,x in rows.items()},
                    gained=int((ok&~base).sum()),lost=int((~ok&base).sum())))
    write(output/'scores.json',dict(status='SCORED_COMPLETE',truth_last=True,results=results))


def score_refined_u(dataset,output,source,methods):
    c=read(source/'contract.json')
    with np.load(source/'U_s_metadata.npz',allow_pickle=False) as m:
        if 'y' in m.files or m['ids'].tolist()!=c['role_ids']['U_s']:
            raise ValueError('Incorrect blind source metadata for offline U scoring')
        rx,base=m['rx_i'],m['base_index']
    truth=np.full(len(rx),-1,dtype=np.int64)
    with zipfile.ZipFile(dataset/'multiple_receiver_train.zip') as archive:
        for i,r in enumerate(SOURCE):
            with recording(archive,f'{r}_train.h5',output) as h5:
                order=packet_indices(8000,i)
                mask=rx==i
                truth[mask]=label_ids(h5,8000)[order[base[mask]-8000*i]]
    if (truth<0).any():
        raise ValueError('Incomplete U physical label lookup')
    result={}
    for method in methods:
        p=np.load(output/'F'/method/'accepted_U.npy',allow_pickle=False)
        if p.ndim!=2 or p.shape[1]!=3 or p.dtype.kind!='i':
            raise ValueError('Invalid accepted U records')
        ep,idx,cls=p.T
        if ((idx<0)|(idx>=len(truth))|(cls<0)|(cls>=len(CLASSES))|(ep<1)|(ep>18)).any():
            raise ValueError('Invalid U accepted physical index or prediction')
        good=truth[idx]==cls
        result[method]=dict(accepted=len(p),correct=int(good.sum()),unique=int(len(np.unique(idx))),
            by_epoch=[dict(epoch=i,total=int((ep==i).sum()),correct=int((good&(ep==i)).sum())) for i in range(1,19)],
            by_predicted_class=[dict(tx=label,total=int((cls==i).sum()),correct=int((good&(cls==i)).sum()))
                                for i,label in enumerate(CLASSES)])
    write(output/'u_offline_scores.json',dict(status='SCORED_COMPLETE',truth_last=True,records=result))


def collect(submit, ssh_config, dataset, output, refinement=False, source=None):
    root=shlex.quote(read(submit)['run_root'])
    conn=['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    wait_for_predictions(conn,root)
    readouts=READOUTS
    extra=''
    if refinement:
        from .refine_runner import READOUTS as readouts
        if source is None:
            raise ValueError('Independent U scoring requires local source IDs')
        extra=' '+ ' '.join(f'F/{m}/accepted_U.npy F/{m}/epochs.jsonl' for m in readouts)
        extra+=' G/phase/epochs.jsonl G/chirp/epochs.jsonl'
    raw=subprocess.check_output([*conn,f'tar -czf - -C {root} predictions source_audit.json{extra}'],timeout=120)
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=BytesIO(raw),mode='r:gz') as archive:
        for member in archive:
            path=PurePosixPath(member.name)
            allowed=('predictions','source_audit.json','F','G') if refinement else ('predictions','source_audit.json')
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in allowed:
                raise ValueError('Unexpected packet result archive')
            if member.isdir():
                continue
            if not member.isfile() or path.suffix not in ('.json','.npy','.jsonl'):
                raise ValueError('Unexpected packet result artifact')
            dest=output.joinpath(*path.parts)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:
                f.write(archive.extractfile(member).read())
    score(dataset,output,readouts)
    if refinement:
        score_refined_u(dataset,output,source,readouts)
    print('PACKET_SCORED',output/'scores.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('submit','ssh-config','dataset','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--refinement',action='store_true')
    p.add_argument('--source',type=Path)
    collect(**vars(p.parse_args()))
