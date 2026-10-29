"""Three disjoint physical pools, repeated actual-K complement tasks per pool.

Reuse the validated r09 practical observations. No model or scoring is loaded.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
import numpy as np
from .data import identity
from .episodic_io import support_order,task_indices
from .io import read,write,atomic,require


def seed_for(*parts):
    return int(identity(list(parts))[:16],16)


def input_ready(source,episode,receiver):
    # r09 publishes this atomic marker only after its ordinary report closes.
    marker=Path(source)/'progress'/(episode+'-'+receiver+'.json')
    return marker.exists() and read(marker)['status']=='VALIDATED_ONCE'


def assign_pools(labels,days,*,classes,receiver,seed,pool_count=3):
    """Random within TX/day; round-robin remainders balance each TX total too."""
    labels=np.asarray(labels);days=np.asarray(days)
    require(pool_count==3 and labels.ndim==days.ndim==1 and len(labels)==len(days),'Wrong partition input')
    require(len(labels)>0 and len(classes)==len(set(classes)),'Empty/duplicate class input')
    require(np.isin(labels,np.arange(len(classes))).all() and np.isin(days,np.arange(4)).all(),'Unknown TX/day')
    assigned=np.full(len(labels),255,np.uint8)
    for ci,c in enumerate(classes):
        offset=seed_for(seed,receiver,c,'offset')%pool_count
        for day in range(4):
            rows=np.flatnonzero((labels==ci)&(days==day))
            order=np.random.default_rng(seed_for(seed,receiver,c,day)).permutation(rows)
            assigned[order]=(np.arange(len(order))+offset)%pool_count
            offset=(offset+len(order))%pool_count
        sizes=np.bincount(assigned[labels==ci],minlength=pool_count)
        require(sizes.max()-sizes.min()<=1,'Class partition imbalance')
    require((assigned<pool_count).all(),'Unassigned physical record')
    return assigned


def prepare(rt):
    source=Path(rt['input_data_root']);classes=rt['old_classes']+rt['new_classes']
    require(len(rt['physical_pools'])==3 and len(set(rt['physical_pools']))==3,'Exactly three unique physical pools required')
    require(not set(rt['target_receivers'])&set(rt['source_receivers']),'Source/target RX overlap')
    pool=read(source/'builder_only/pool/manifest.json')
    require(pool['status']=='FULL_SHARED_RAW_POOL_READY' and pool['classes']==classes,'Wrong validated raw pool')
    require(pool['physical_records']==rt['expected_pool_records'],'Wrong original physical total')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    plans={};counts={};minimum=10**9;maximum=0;total=0
    for r in rt['target_receivers']:
        with np.load(source/'builder_only/pool'/r/'records.npz',allow_pickle=False) as z:
            labels=z['labels'];days=z['days'];physical_ids=z['physical_ids']
        require(len(physical_ids)==len(set(physical_ids.tolist())),'Duplicate input physical IDs')
        assigned=assign_pools(labels,days,classes=classes,receiver=r,seed=rt['physical_pool_seed'])
        dest=root/'builder_only/partition'/r;dest.mkdir(parents=True)
        np.savez(dest/'partition.npz',physical_ids=physical_ids,pool_indices=assigned)
        per_pool=[]
        for pi,pid in enumerate(rt['physical_pools']):
            cell=np.array([[np.count_nonzero((assigned==pi)&(labels==ci)&(days==d)) for d in range(4)] for ci in range(len(classes))])
            class_counts=cell.sum(axis=1)
            require((class_counts-max(rt['shots'])>=rt['minimum_query_per_class']).all(),'Three-pool class has fewer than minimum query after max K')
            minimum=min(minimum,int(class_counts.min()));maximum=max(maximum,int(class_counts.max()))
            total+=int(cell.sum());per_pool.append(int(cell.sum()))
            plans[(r,pid)]={str(physical_ids[i]) for i in np.flatnonzero(assigned==pi)}
            counts[(r,pid)]=dict(physical_records=int(cell.sum()),class_counts=class_counts.tolist(),day_counts=cell.tolist())
        require(sum(per_pool)==len(physical_ids),'Partition lost/duplicated records')
        for a in range(3):
            for b in range(a):require(not plans[(r,rt['physical_pools'][a])]&plans[(r,rt['physical_pools'][b])],'Physical pools overlap')
        write(dest/'counts.json',dict(receiver=r,pools={p:counts[(r,p)] for p in rt['physical_pools']}))
        print('PARTITION_READY',r,per_pool,flush=True)
    require(total==rt['expected_pool_records'],'Wrong three-pool union')
    pool_totals={p:sum(counts[(r,p)]['physical_records'] for r in rt['target_receivers']) for p in rt['physical_pools']}
    report=dict(status='THREE_DISJOINT_POOLS_VALIDATED_ONCE',physical_records=total,pool_totals=pool_totals,
                min_class_pool=minimum,max_class_pool=maximum,min_query_at_max_K=minimum-max(rt['shots']),
                pool_count=3,stratification='RX/TX/day',each_TX_and_each_date_pool_count_difference_at_most_one=True,
                source_roles_unchanged=True,input_data_root=rt['input_data_root'],code_commit=rt['code_commit'])
    write(root/'partition_report.json',report)
    return plans,counts,report


def bank(rt,episode,receiver,plans,counts):
    source=Path(rt['input_data_root']);root=Path(rt['run_root']);classes=rt['old_classes']+rt['new_classes']
    worker=read(source/'worker_reports'/(episode+'-'+receiver+'.json'))
    require(worker['status']=='VALIDATED_ONCE' and worker['episode']==episode and worker['receiver']==receiver,'Wrong validated bank')
    src=source/worker['capsule'];manifest=read(src/'manifest.json')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE' and manifest['source_roles_unchanged'],'Wrong observation contract')
    require(manifest['capsule_id']==rt['input_run_id']+':'+episode+':'+receiver,'Wrong input run')
    received=np.load(src/'received.npy',mmap_mode='r',allow_pickle=False)
    obs_ids=np.load(src/'ids.npy',mmap_mode='r',allow_pickle=False)
    with np.load(source/'score_only'/episode/(receiver+'.npz'),allow_pickle=False) as z:
        truth={key:z[key] for key in z.files}
    require(len(received)==worker['observations']==len(truth['physical_ids']),'Wrong input bank length')
    require(np.array_equal(obs_ids,truth['observation_ids']),'Wrong input ID alignment')
    physical_ids=truth['physical_ids'];reports=[];used=set()
    for pid in rt['physical_pools']:
        positions=np.flatnonzero(np.fromiter((str(x) in plans[(receiver,pid)] for x in physical_ids),dtype=bool,count=len(physical_ids)))
        ids=np.asarray(obs_ids[positions]);labels=truth['labels'][positions];n=len(positions)
        require(n==counts[(receiver,pid)]['physical_records'],'Wrong physical pool membership')
        require(not used.intersection(positions.tolist()),'Repeated input row across physical pools');used.update(positions.tolist())
        capsule=root/'capsules'/pid/episode/receiver;capsule.mkdir(parents=True,exist_ok=False)
        tmp=capsule/'received.partial.npy'
        target=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(n,2,256))
        target[:]=received[positions];target.flush();del target;tmp.rename(capsule/'received.npy')
        np.save(capsule/'ids.npy',ids,allow_pickle=False)
        for d in ('stage_rows','splits','support_labels'):(capsule/d).mkdir()
        orders=[support_order(np.flatnonzero(labels==ci),seed_for(rt['support_seed'],pid,episode,receiver,c)) for ci,c in enumerate(classes)]
        capid=rt['run_id']+':'+pid+':'+episode+':'+receiver;query_counts={};split_count=0
        for nc in rt['new_counts']:
            rows=np.flatnonzero(labels<6+nc).astype(np.int32)
            rows=rows[np.random.default_rng(seed_for(rt['query_order_seed'],pid,episode,receiver,nc)).permutation(len(rows))]
            np.save(capsule/'stage_rows'/f'N{nc:02d}.npy',rows,allow_pickle=False)
            for k in rt['shots']:
                support,query,sy=task_indices(orders,6+nc,k)
                split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capid,
                           physical_pool=pid,episode_id=episode,receiver=receiver,new_count=nc,k=k,
                           registered_classes=classes[:6+nc],support_indices=support.tolist(),
                           query_rule='all_registered_physical_pool_rows_minus_current_support',query_count=len(query),
                           support_seed=seed_for(rt['support_seed'],pid,episode,receiver))
                split['split_id']=identity(split);write(capsule/'splits'/(split['split_id']+'.json'),split)
                np.save(capsule/'support_labels'/(split['split_id']+'.npy'),sy,allow_pickle=False)
                query_counts[f'K{k}-N{nc}']=len(query);split_count+=1
        score=root/'score_only'/pid/episode;score.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(score/(receiver+'.npz'),**{key:value[positions] for key,value in truth.items()})
        write(capsule/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capid,
            storage='episodic-pool-complement-v1',received_count=n,signal_shape=[2,256],dtype='float32',
            physical_pool=pid,episode_id=episode,receiver=receiver,split_count=split_count,source_roles_unchanged=True,
            physical_pools_disjoint=True,channel_redrawn_per_episode=True,observation_values_and_ids_reused_exactly=True,
            input_visibility='Only current registered-stage rows minus actual support;whole-bank model export forbidden',
            channel=manifest['channel']))
        report=dict(status='VALIDATED_ONCE',physical_pool=pid,episode=episode,receiver=receiver,observations=n,
                    splits=split_count,query_counts=query_counts,capsule=str(capsule.relative_to(root)),
                    scene_counts=dict(Counter(truth['scene_indices'][positions].tolist())),
                    input_capsule_id=manifest['capsule_id'],observations_reused=True,model_reads=0,identity_predictions=0,scoring=0)
        write(root/'worker_reports'/(pid+'-'+episode+'-'+receiver+'.json'),report);reports.append(report)
    require(len(used)==len(received),'Three pools do not cover whole received bank')
    print('THREE_POOL_BANK_READY',episode,receiver,len(received),flush=True)
    return reports


def run(runtime):
    rt=read(runtime);release=Path.cwd();status=release/'preparation_status.json';root=Path(rt['run_root']);started=time.time()
    atomic(status,dict(status='RUNNING',stage='partition',controller_pid=os.getpid(),run_root=str(root),code_commit=rt['code_commit']))
    completed=[]
    try:
        plans,counts,partition=prepare(rt)
        pending=[(e['id'],r) for e in rt['episodes'] for r in rt['target_receivers']]
        while pending:
            require(time.time()-started<rt['input_wait_timeout_seconds'],'Input observation job exceeded preregistered wait budget')
            ready=[(e,r) for e,r in pending if input_ready(rt['input_data_root'],e,r)]
            for e,r in ready:
                completed.extend(bank(rt,e,r,plans,counts));pending.remove((e,r))
                atomic(status,dict(status='RUNNING',stage='repack',controller_pid=os.getpid(),run_root=str(root),code_commit=rt['code_commit'],
                    completed_source_banks=len(rt['episodes'])*len(rt['target_receivers'])-len(pending),
                    completed_banks=len(completed),total_banks=rt['expected_outputs']['capsules'],
                    completed_observations=sum(x['observations'] for x in completed),elapsed_seconds=time.time()-started))
            if pending:
                upstream=read(Path(rt['input_release_root'])/'preparation_status.json')
                require(upstream['status'] in ('RUNNING','DATA_COMPLETE'),'Upstream observation builder failed/stopped')
                require(upstream['status']!='DATA_COMPLETE' or ready,'Completed input missing expected source banks')
                atomic(status,dict(status='RUNNING',stage='waiting_for_input',controller_pid=os.getpid(),run_root=str(root),code_commit=rt['code_commit'],
                    completed_banks=len(completed),total_banks=rt['expected_outputs']['capsules'],
                    completed_observations=sum(x['observations'] for x in completed),pending_source_banks=len(pending),elapsed_seconds=time.time()-started))
                time.sleep(rt['poll_seconds'])
        expected=rt['expected_outputs'];observations=sum(x['observations'] for x in completed);splits=sum(x['splits'] for x in completed)
        require(len(completed)==expected['capsules'] and observations==expected['reused_observations'] and splits==expected['splits'],'Wrong completed totals')
        write(root/'catalog.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsules=completed))
        write(root/'score_only/label_map.json',dict(classes=rt['old_classes']+rt['new_classes'],dates=rt['dates'],
                                                   scenes=[x['id'] for x in read(rt['mixture_config'])['scenarios']]))
        result=dict(status='VALIDATED_ONCE',code_commit=rt['code_commit'],unique_physical_pool_records=partition['physical_records'],
                    physical_pools=3,episodes_per_pool=len(rt['episodes']),capsules=len(completed),splits=splits,
                    reused_observations=observations,additional_channel_generations=0,identity_training=0,identity_predictions=0,scoring=0,
                    query_rule='entire_current_registered_physical_pool_minus_K_support_per_class',elapsed_seconds=time.time()-started)
        write(root/'builder_report.json',result);atomic(status,dict(**{k:v for k,v in result.items() if k!='status'},status='DATA_COMPLETE'))
    except BaseException as exc:
        atomic(status,dict(status='FAILED',error=repr(exc),controller_pid=os.getpid(),run_root=str(root),completed_banks=len(completed)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);run(p.parse_args().runtime)
