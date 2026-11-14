"""MoPC public-code source/clean-support diagnostic; never reads query or truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .io import activate, read, require


def support_only_clean(data_root, clean_root, capsule, split_id):
    data=Path(data_root).resolve();clean=Path(clean_root).resolve();cap=(data/capsule).resolve()
    require(cap.is_relative_to(data/'capsules') and len(split_id)==24 and
            all(c in '0123456789abcdef' for c in split_id), 'Invalid capsule/split')
    manifest=read(cap/'manifest.json');split=read(cap/'splits'/(split_id+'.json'))
    require(manifest['storage']=='episodic-pool-complement-v1' and
            manifest['protocol_schema']==split['protocol_schema']=='p2_min_v1' and
            manifest['phase2_data_status']==split['phase2_data_status']=='VALIDATED_ONCE' and
            manifest['capsule_id']==split['capsule_id'] and split['split_id']==split_id, 'Wrong task binding')
    done=read(clean/'completion.json')
    require(done.get('status')=='CLEAN_PAIRED_COMPLETE' and Path(done['parent_task_root']).resolve()==data,
            'Clean pairing binding mismatch')
    k,n=int(split['k']),int(split['new_count']);classes=split['registered_classes']
    require(k in (1,5,10,20) and n in (2,5,10,20) and len(classes)==6+n, 'Wrong K/N')
    idx=np.asarray(split['support_indices'],dtype=np.int64)
    require(len(idx)==k*(6+n) and len(np.unique(idx))==len(idx) and
            (idx>=0).all() and (idx<manifest['received_count']).all(), 'Wrong support positions')
    labels=np.load(cap/'support_labels'/(split_id+'.npy'),allow_pickle=False)
    require(labels.tolist()==np.repeat(np.arange(6+n),k).tolist(), 'Wrong support labels')
    cc=clean/'capsules'/cap.relative_to(data/'capsules');cm=read(cc/'manifest.json')
    require(cm['capsule_id']==manifest['capsule_id'] and cm['receiver']==manifest['receiver'] and
            cm['input_condition']=='clean_ground_pre_satellite_normalized', 'Wrong clean capsule')
    mapping=np.load(cc/'row_to_clean.npy',mmap_mode='r',allow_pickle=False)
    bank=np.load(clean/'banks'/manifest['receiver']/'clean.npy',mmap_mode='r',allow_pickle=False)
    require(mapping.dtype==np.int32 and mapping.shape==(manifest['received_count'],) and
            bank.dtype==np.float32 and bank.ndim==3 and bank.shape[1:]==(2,256), 'Wrong clean data layout')
    new=labels>=6;selected=np.asarray(mapping[idx[new]],dtype=np.int64)
    require((selected>=0).all() and (selected<len(bank)).all(), 'Wrong clean support mapping')
    return np.array(bank[selected],dtype=np.float32,copy=True),labels[new].astype(np.int64)


def run(rt, source_root, capsule, seed, k, n, device):
    from .group1 import source_scope
    from .group1_source import load_source
    from .group1_target import fit_C
    from .clean_source_smoke import check
    from .external_execution import seed_for
    from .fast_transfer import to_tensor
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=source_scope(rt);source['run_root']=str(Path(source_root).resolve())
    bundle=load_source(source,'mopc_hr',seed,device);check(rt,bundle)
    print(json.dumps(dict(event='SOURCE_SMOKE_PASS',seed=seed,query_read=False,truth_read=False)),flush=True)
    cap=Path(rt['data_root'])/capsule
    matches=[v for v in (read(p) for p in (cap/'splits').glob('*.json')) if v['k']==k and v['new_count']==n]
    require(len(matches)==1,'K/N split is not unique')
    x,y=support_only_clean(rt['data_root'],rt['clean_root'],capsule,matches[0]['split_id'])
    sx=to_tensor(x,device,np.float32);sy=to_tensor(y,device,np.int64).long()
    c_seed=seed_for(seed,capsule,k,n,'C');losses=[]
    def observe(row):
        losses.append(float(row['loss']))
        if len(losses)%40==0:
            print(json.dumps(dict(event='PROGRESS',updates=len(losses),loss=losses[-1],
                                  stage=row.get('stage'),epoch=row.get('epoch')),allow_nan=False),flush=True)
    state=fit_C(rt,bundle,sx,sy,tuple(range(6,6+n)),c_seed,observe)
    require(losses and all(torch.isfinite(p).all().item() for p in state['model'].parameters()),'Nonfinite final model')
    print(json.dumps(dict(event='DIAGNOSTIC_COMPLETE',seed=seed,c_seed=c_seed,k=k,n=n,
        semantics='public_code',updates=len(losses),loss_first=losses[0],loss_last=losses[-1],
        loss_max=max(losses),query_read=False,truth_read=False,predictions_written=False),allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('runtime','source-root','capsule'):p.add_argument('--'+x,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--k',type=int,required=True)
    p.add_argument('--n',type=int,required=True);p.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    a=p.parse_args();run(read(a.runtime),a.source_root,a.capsule,a.seed,a.k,a.n,a.device)


if __name__=='__main__':main()
