"""Offline shared-pool episodes, each with a fresh practical residual overlay."""
import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import pickle
import time
import zipfile
import numpy as np
from .data import identity,observe
from .episodic_io import support_order,task_indices,observation_id
from .full_data import full_members
from .io import read,write,atomic,require


def prepare_pool(rt):
    require(not set(rt['source_receivers'])&set(rt['target_receivers']),'Source/target RX overlap')
    classes=rt['old_classes']+rt['new_classes']
    require(len(rt['old_classes'])==6 and len(classes)==len(set(classes))==26,'Wrong original TX sets')
    report=read(rt['validated_inventory_ref'])
    require(report['status']=='RAW_ELIGIBILITY_VERIFIED','Original Full inventory not validated')
    counts={(v['receiver'],v['transmitter'],v['date']):v['full'] for v in report['cells']}
    sizes={r:sum(counts[(r,c,d)] for c in classes for d in rt['dates']) for r in rt['target_receivers']}
    require(sum(sizes.values())==rt['expected_pool_records'],'Wrong full original-class pool count')
    for r in sizes:
        for c in classes:
            require(sum(counts[(r,c,d)] for d in rt['dates'])>max(rt['shots']),'Insufficient pool for maximum K')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    private=root/'builder_only/pool';private.mkdir(parents=True)
    members=full_members(rt['raw_directory'],rt['dates'],rt['target_receivers'])
    for r in rt['target_receivers']:
        dest=private/r;dest.mkdir()
        n=sizes[r];raw=np.lib.format.open_memmap(dest/'raw_iq.npy',mode='w+',dtype=np.float32,shape=(n,2,256))
        labels=np.empty(n,np.int16);days=np.empty(n,np.uint8);signals=np.empty(n,np.int32);ids=np.empty(n,'U24')
        pos=0
        for di,day in enumerate(rt['dates']):
            archive,member=members[(r,day)]
            with zipfile.ZipFile(archive) as z:data=pickle.loads(z.read(member))
            for ci,c in enumerate(classes):
                x=data['data'][data['node_list'].index(c)];m=len(x)
                require(x.shape==(counts[(r,c,day)],256,2) and x.dtype==np.float64,'Changed validated Full member')
                raw[pos:pos+m]=x.transpose(0,2,1).astype(np.float32)
                labels[pos:pos+m]=ci;days[pos:pos+m]=di;signals[pos:pos+m]=np.arange(m)
                ids[pos:pos+m]=[identity(['WiSig/Full/eq1',c,r,day,i]) for i in range(m)]
                pos+=m
            del data,x;gc.collect()
        require(pos==n and len(set(ids.tolist()))==n,'Raw pool total/identity differs')
        raw.flush();del raw
        np.savez(dest/'records.npz',labels=labels,days=days,signals=signals,physical_ids=ids)
        write(dest/'manifest.json',dict(receiver=r,physical_records=n,class_counts=np.bincount(labels,minlength=26).tolist()))
        print('RAW_POOL_READY',r,n,flush=True)
    write(private/'manifest.json',dict(status='FULL_SHARED_RAW_POOL_READY',physical_records=sum(sizes.values()),per_RX=sizes,
                                      classes=classes,old_classes=rt['old_classes'],new_classes=rt['new_classes'],
                                      legacy_exclusion=False,source_roles_unchanged=True))


def build_bank(rt,receiver,episode_id):
    from leo_practical.channel import stable_seed
    started=time.time();root=Path(rt['run_root']);episode=next(v for v in rt['episodes'] if v['id']==episode_id)
    classes=rt['old_classes']+rt['new_classes'];mix=read(rt['mixture_config'])
    private=root/'builder_only/pool'/receiver;meta=read(private/'manifest.json');n=meta['physical_records']
    raw=np.load(private/'raw_iq.npy',mmap_mode='r',allow_pickle=False)
    with np.load(private/'records.npz',allow_pickle=False) as records:
        original_labels=records['labels'];original_days=records['days'];original_signals=records['signals'];original_ids=records['physical_ids']
    order=np.random.default_rng(stable_seed(rt['bank_order_seed'],episode_id,receiver)).permutation(n)
    labels=original_labels[order];days=original_days[order];signals=original_signals[order];physical_ids=original_ids[order]
    capsule=root/'capsules'/episode_id/receiver;capsule.mkdir(parents=True,exist_ok=False)
    for name in ('stage_rows','splits','support_labels'):(capsule/name).mkdir()
    progress=root/'progress'/(episode_id+'-'+receiver+'.json')
    tmp=capsule/'received.partial.npy'
    bank=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(n,2,256))
    ids=np.empty(n,'U24');scenes=np.empty(n,np.uint8);overlay_seeds=np.empty(n,np.uint64)
    input_rms=np.empty(n,np.float64);elevation=np.empty(n,np.float64);scene_counts=Counter()
    for j,index in enumerate(order):
        sid=str(physical_ids[j]);day=rt['dates'][int(days[j])]
        value,channel=observe(raw[int(index)],mix['scenarios'],sid=sid,session=f'rx:{receiver}/day:{day}',pool=episode,mixture=mix)
        require(np.isfinite(value).all(),'Nonfinite received observation')
        bank[j]=value;ids[j]=observation_id(sid,rt['run_id']+':'+episode_id)
        scene=next(i for i,s in enumerate(mix['scenarios']) if s['id']==channel['scenario'])
        scenes[j]=scene;overlay_seeds[j]=channel['overlay_seed'];input_rms[j]=channel['input_rms']
        g=channel['geometry'];elevation[j]=float(g.get('elevation_deg',np.nan)) if g else np.nan
        scene_counts[channel['scenario']]+=1
        if (j+1)%10000==0:
            bank.flush()
            atomic(progress,dict(status='GENERATING',pid=os.getpid(),run_root=rt['run_root'],episode=episode_id,receiver=receiver,
                                 completed=j+1,total=n,elapsed_seconds=time.time()-started))
            print('OBSERVATIONS',episode_id,receiver,j+1,n,flush=True)
    require(len(set(ids.tolist()))==n,'Observation ID collision within episode')
    bank.flush();del bank;tmp.rename(capsule/'received.npy')
    np.save(capsule/'ids.npy',ids,allow_pickle=False)
    orders=[support_order(np.flatnonzero(labels==ci),stable_seed(rt['support_seed'],episode_id,receiver,c)) for ci,c in enumerate(classes)]
    require(all(len(v)>max(rt['shots']) for v in orders),'Raw pool too small')
    capsule_id=rt['run_id']+':'+episode_id+':'+receiver
    split_count=0;query_counts={}
    for nc in rt['new_counts']:
        registered=classes[:6+nc]
        rows=np.flatnonzero(labels<6+nc).astype(np.int32)
        rows=rows[np.random.default_rng(stable_seed(rt['query_order_seed'],episode_id,receiver,nc)).permutation(len(rows))]
        np.save(capsule/'stage_rows'/f'N{nc:02d}.npy',rows,allow_pickle=False)
        for k in rt['shots']:
            support,query,sy=task_indices(orders,6+nc,k)
            split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capsule_id,
                       episode_id=episode_id,receiver=receiver,new_count=nc,k=k,registered_classes=registered,
                       support_indices=support.tolist(),query_rule='all_registered_pool_rows_minus_current_support',
                       query_count=len(query),support_seed=stable_seed(rt['support_seed'],episode_id,receiver))
            split['split_id']=identity(split)
            write(capsule/'splits'/(split['split_id']+'.json'),split)
            np.save(capsule/'support_labels'/(split['split_id']+'.npy'),sy,allow_pickle=False)
            query_counts[f'K{k}-N{nc}']=len(query);split_count+=1
    # No TX, date, raw path or query labels are written inside the capsule.
    score=root/'score_only'/episode_id;score.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(score/(receiver+'.npz'),observation_ids=ids,physical_ids=physical_ids,labels=labels,
                        days=days,signal_indices=signals,scene_indices=scenes,overlay_seeds=overlay_seeds,
                        input_rms=input_rms,elevation_deg=elevation)
    manifest=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capsule_id,
                  storage='episodic-pool-complement-v1',received_count=n,signal_shape=[2,256],dtype='float32',
                  episode_id=episode_id,receiver=receiver,split_count=split_count,source_roles_unchanged=True,
                  physical_pool_reused_across_episodes=True,channel_redrawn_per_episode=True,
                  input_visibility='Only rows selected by the current split/stage reader; whole-bank model export forbidden',
                  channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25e6))
    write(capsule/'manifest.json',manifest)
    worker_report=dict(status='VALIDATED_ONCE',pid=os.getpid(),episode=episode_id,receiver=receiver,
                       observations=n,splits=split_count,query_counts=query_counts,scene_counts=dict(scene_counts),
                       elapsed_seconds=time.time()-started,capsule=str(capsule.relative_to(root)),
                       model_reads=0,identity_predictions=0,within_task_support_query_disjoint=True,
                       query_is_full_complement=True,query_labels_outside_capsule=True)
    write(root/'worker_reports'/(episode_id+'-'+receiver+'.json'),worker_report)
    atomic(progress,worker_report)
    print('BANK_COMPLETE',episode_id,receiver,n,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['pool','worker'])
    p.add_argument('--runtime',required=True);p.add_argument('--receiver');p.add_argument('--episode');a=p.parse_args()
    rt=read(a.runtime)
    import sys
    sys.path.insert(0,str(Path(rt['phase1_release'])/'code'))
    if a.mode=='pool':prepare_pool(rt)
    else:build_bank(rt,a.receiver,a.episode)


if __name__=='__main__':main()
