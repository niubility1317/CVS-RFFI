"""One label-free builder: fixed received observations shared by all eight rows."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import numpy as np
from leo_practical import Config,apply_leo_practical_channel_batch
from experiments.cvs_sixscene_eval.common import SCENES,read,write

def configuration(scene):
    return Config(fs_hz=25e6,scenario=scene,processing_route='residual',mode='post_sync',
        equalization_enabled=False,input_processing_state='WiSig_equalized1_center256_unit_rms')

def generate(task):
    capsule,scene,start,stop=task
    with np.load(Path(capsule)/'index.npz',allow_pickle=False) as ix:ids=ix['ids'][start:stop].tolist()
    array=np.load(Path(capsule)/'clean.npy',mmap_mode='r',allow_pickle=False)[start:stop]
    y,records,states=apply_leo_practical_channel_batch(np.asarray(array,dtype=np.float64),configuration(scene),
        seed=392005,sample_ids=ids,session_ids=['virtual_target_session0']*len(ids),
        realization_namespace='phase1_full_six_fixed_20261002_v1',receiver_seed=2027,return_meta=True)
    if y.shape!=array.shape or not np.isfinite(y).all():raise ValueError('Invalid generated observations')
    if [r['sample_id'] for r in records]!=ids or any(r['channel_equalization_applied'] for r in records):raise ValueError('Identity/route mismatch')
    elevations=[r['geometry']['elevation_deg'] for r in records]
    return start,y,dict(count=len(ids),locks=dict(Counter(r['receiver_lock_status'] for r in records)),
        elevation_min=min(elevations),elevation_max=max(elevations),example=records[0] if start==0 else None)

def build(spec):
    capsule=Path(spec['capsule']);m=read(capsule/'manifest.json')
    if m['status']!='VALIDATED_ONCE' or m['count']!=168000 or not m['source_target_disjoint'] or m['channel']!='residual/post_sync/noeq':raise ValueError('Input contract mismatch')
    with np.load(capsule/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
    if len(ids)!=168000 or len(set(ids.tolist()))!=len(ids):raise ValueError('Physical coverage mismatch')
    out=Path(spec['views_root']);out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'index.npz',ids=ids)
    evidence=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=24) as pool:
        for scene in SCENES:
            tic=time.perf_counter();array=np.lib.format.open_memmap(out/(scene+'.npy'),mode='w+',dtype=np.float32,shape=(len(ids),2,256))
            counts=Counter();lock_counts=Counter();examples=[];low=90.;high=0.
            tasks=[(str(capsule),scene,s,min(s+1000,len(ids))) for s in range(0,len(ids),1000)]
            for start,y,meta in pool.map(generate,tasks,chunksize=1):
                array[start:start+len(y)]=y;counts['count']+=len(y);lock_counts.update(meta['locks'])
                low=min(low,meta['elevation_min']);high=max(high,meta['elevation_max'])
                if meta['example']:examples.append(meta['example'])
            array.flush();del array
            if counts['count']!=len(ids):raise ValueError('Missing received observations')
            item=dict(scene=scene,count=counts['count'],config=asdict(configuration(scene)),locks=dict(lock_counts),
                elevation_min=low,elevation_max=high,example=examples[0],seconds=time.perf_counter()-tic)
            evidence.append(item);print('VIEW_VALIDATED '+json.dumps({k:v for k,v in item.items() if k not in {'config','example'}}),flush=True)
    write(out/'manifest.json',dict(status='VALIDATED_ONCE',classes=m['classes'],count=len(ids),scenes=list(SCENES),
        source_capsule=str(capsule),clean_ref=str(capsule/'clean.npy'),source_target_disjoint=True,unique_ids=True,
        seed=392005,receiver_seed=2027,namespace='phase1_full_six_fixed_20261002_v1',
        session='virtual_target_session0',channel='residual/post_sync/noeq',fs_hz=25000000,
        truth_read=False,rows_share_observations=True,seconds=time.perf_counter()-started,evidence=evidence))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();build(read(a.spec))
