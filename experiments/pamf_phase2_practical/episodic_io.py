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


def load_task(capsule, split_id, *, stage, clean_root=None, task_root=None, restored_root=None):
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
    ids=np.load(root/'ids.npy',mmap_mode='r',allow_pickle=False)
    registered=np.load(root/'stage_rows'/f'N{n:02d}.npy',allow_pickle=False)
    require((clean_root is None)==(task_root is None),'Clean input requires both roots')
    require(restored_root is None or clean_root is not None,'Restored clean input requires clean mapping')
    mapping=None
    if clean_root is None:
        bank=np.load(root/'received.npy',mmap_mode='r',allow_pickle=False)
        require(bank.shape==(manifest['received_count'],2,256),'Wrong received bank')
        condition='satellite_received'
    else:
        clean_root=Path(clean_root).resolve();task_root=Path(task_root).resolve()
        completed=read(clean_root/'completion.json')
        require(completed.get('status')=='CLEAN_PAIRED_COMPLETE' and Path(completed['parent_task_root']).resolve()==task_root,
                'Clean pairing is incomplete or bound to a different parent')
        relative=root.resolve().relative_to(task_root/'capsules')
        clean_cap=clean_root/'capsules'/relative
        clean_manifest=read(clean_cap/'manifest.json')
        require(clean_manifest['capsule_id']==manifest['capsule_id'] and
                clean_manifest['receiver']==manifest['receiver'] and
                clean_manifest['input_condition']=='clean_ground_pre_satellite_normalized', 'Wrong clean capsule')
        mapping=np.load(clean_cap/'row_to_clean.npy',mmap_mode='r',allow_pickle=False)
        bank=np.load(clean_root/'banks'/manifest['receiver']/'clean.npy',mmap_mode='r',allow_pickle=False)
        require(mapping.dtype==np.int32 and mapping.shape==(manifest['received_count'],) and
                (mapping>=0).all() and (mapping<len(bank)).all(), 'Wrong clean mapping')
        require(bank.ndim==3 and bank.shape[1:]==(2,256),'Wrong clean bank')
        condition='clean_ground_pre_satellite_normalized'
    require(bank.dtype==np.float32,'Wrong bank dtype')
    require(ids.shape==(manifest['received_count'],) and registered.ndim==1 and registered.dtype==np.int32,'Wrong bank/index schema')
    require(len(registered)==len(set(registered.tolist())) and (registered>=0).all() and (registered<len(ids)).all(),'Illegal registered positions')
    query=query_complement(registered,support)
    require(len(query)==split['query_count'],'Wrong query complement count')
    def rows(indices):
        return bank[indices if mapping is None else mapping[indices]]
    out=dict(manifest=manifest,split=split,query_x=rows(query),query_ids=np.asarray(ids[query]),input_condition=condition)
    # A does not open the target support-label file or read support IQ.
    if stage!='A':
        labels=np.load(root/'support_labels'/(split_id+'.npy'),allow_pickle=False)
        require(labels.tolist()==np.repeat(np.arange(len(classes)),k).tolist(),'Wrong support labels')
        out.update(support_x=rows(support),support_ids=np.asarray(ids[support]),support_y=labels)
    if restored_root is not None:
        restored_root=Path(restored_root).resolve();done=read(restored_root/'completion.json')
        require(done.get('status')=='CLEAN_DURA_COMPLETE' and Path(done['clean_root']).resolve()==clean_root and
                done.get('restorer_id')=='dura_v1', 'Restored bank is incomplete or mismatched')
        restored=np.load(restored_root/'banks'/manifest['receiver']/'restored.npy',mmap_mode='r',allow_pickle=False)
        require(restored.dtype==np.float32 and restored.shape==bank.shape,'Wrong restored clean bank')
        out['restored_query_x']=restored[mapping[query]]
        if stage!='A':out['restored_support_x']=restored[mapping[support]]
    return out
