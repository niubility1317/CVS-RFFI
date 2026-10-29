"""Task-relative support/query on one re-overlaid observation bank per episode."""
from pathlib import Path
import numpy as np
from .data import identity
from .io import read,require


def support_order(class_rows, seed):
    rows=np.asarray(class_rows,dtype=np.int32)
    require(len(rows)==len(set(rows.tolist())), 'Repeated raw pool row')
    return rows[np.random.default_rng(seed).permutation(len(rows))]


def task_indices(class_orders, registered_count, k):
    require(type(k) is int and k>0 and 0<registered_count<=len(class_orders), 'Invalid K/classes')
    require(all(len(x)>k for x in class_orders[:registered_count]), 'Pool must contain support and nonempty query')
    support=np.concatenate([x[:k] for x in class_orders[:registered_count]]).astype(np.int32)
    registered=np.concatenate([x for x in class_orders[:registered_count]]).astype(np.int32)
    require(len(set(registered.tolist()))==len(registered), 'Class pool overlap within an episode')
    query=registered[~np.isin(registered,support)]
    require(len(query)==len(registered)-k*registered_count and not np.isin(support,query).any(), 'Wrong query complement')
    labels=np.repeat(np.arange(registered_count,dtype=np.int16),k)
    return support,query,labels


def query_complement(registered, support):
    registered=np.asarray(registered);support=np.asarray(support)
    require(len(set(support.tolist()))==len(support) and np.isin(support,registered).all(), 'Illegal support positions')
    return registered[~np.isin(registered,support)]


def observation_id(physical_id, episode_id):
    return identity(['Full-eq-episodic-observation-v1',physical_id,episode_id])


def load_task(capsule, split_id, *, stage):
    """Mmap metadata only; select permitted rows before any model sees IQ."""
    require(stage in ('A','B','C'),'Unknown stage')
    require(len(split_id)==24 and all(c in '0123456789abcdef' for c in split_id),'Invalid split ID')
    root=Path(capsule);manifest=read(root/'manifest.json')
    require(manifest['storage']=='episodic-pool-complement-v1','Unsupported storage')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE','Unvalidated episode')
    split=read(root/'splits'/(split_id+'.json'))
    require(split['capsule_id']==manifest['capsule_id'] and split['split_id']==split_id,'Wrong capsule/split')
    require(split['protocol_schema']=='p2_min_v1' and split['phase2_data_status']=='VALIDATED_ONCE','Wrong split contract')
    require(stage=='C' or split['new_count']==0,'A/B may open only the old-class task')
    n=split['new_count'];k=split['k'];classes=split['registered_classes']
    require(n in (0,2,5,10,20) and k in (1,5,10,20) and len(classes)==6+n,'Wrong K/registration stage')
    support=np.asarray(split['support_indices'],dtype=np.int32)
    require(len(support)==k*len(classes),'Wrong support size')
    bank=np.load(root/'received.npy',mmap_mode='r',allow_pickle=False)
    ids=np.load(root/'ids.npy',mmap_mode='r',allow_pickle=False)
    registered=np.load(root/'stage_rows'/f'N{n:02d}.npy',allow_pickle=False)
    require(bank.dtype==np.float32 and bank.shape==(manifest['received_count'],2,256),'Wrong received bank')
    require(ids.shape==(len(bank),) and registered.ndim==1 and registered.dtype==np.int32,'Wrong bank/index schema')
    require(len(registered)==len(set(registered.tolist())) and (registered>=0).all() and (registered<len(bank)).all(),'Illegal registered positions')
    query=query_complement(registered,support)
    require(len(query)==split['query_count'],'Wrong query complement count')
    out=dict(manifest=manifest,split=split,query_x=bank[query],query_ids=np.asarray(ids[query]))
    # A does not open the target support-label file or read support IQ.
    if stage!='A':
        labels=np.load(root/'support_labels'/(split_id+'.npy'),allow_pickle=False)
        require(labels.tolist()==np.repeat(np.arange(len(classes)),k).tolist(),'Wrong support labels')
        out.update(support_x=bank[support],support_ids=np.asarray(ids[support]),support_y=labels)
    return out
