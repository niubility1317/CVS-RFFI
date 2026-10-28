"""A row reader that opens only the requested registration stage and actual K."""
from pathlib import Path
import numpy as np
from .io import read,require,validate_split


def load_query(capsule):
    """A-stage access: query IQ/opaque IDs only; never opens target support."""
    root=Path(capsule);manifest=read(root/'manifest.json')
    require(manifest['storage']=='support-plus-shared-query-v1', 'Unknown Full data storage')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE', 'Unvalidated data')
    q=np.load(root/'query_iq.npy',mmap_mode='r',allow_pickle=False)
    qi=np.load(root/'query_ids.npy',allow_pickle=False)
    require(q.dtype==np.float32 and list(q.shape[1:])==[2,256], 'Wrong query IQ layout/type')
    require(len(q)==manifest['query_count']==len(qi) and len(set(qi))==len(qi), 'Wrong query IDs/count')
    return dict(manifest=manifest,query_x=q,query_ids=qi)


def load_row(capsule, split_id):
    root=Path(capsule)
    require(len(split_id)==24 and all(c in '0123456789abcdef' for c in split_id), 'Invalid split ID')
    manifest=read(root/'manifest.json')
    require(manifest['storage']=='support-plus-shared-query-v1', 'Unknown Full data storage')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE', 'Unvalidated data')
    split=read(root/'splits'/(split_id+'.json'))
    require(split['split_id']==split_id, 'Wrong split')
    require(manifest['registered_classes']==split['registered_classes'], 'Wrong registration stage')
    q=np.load(root/'query_iq.npy',mmap_mode='r',allow_pickle=False)
    qi=np.load(root/'query_ids.npy',allow_pickle=False)
    with np.load(root/'supports'/(split_id+'.npz'),allow_pickle=False) as data:
        require(set(data.files)=={'iq','ids','labels'}, 'Unexpected support fields')
        s=data['iq']; si=data['ids']; y=data['labels']
    validate_split(split,manifest,len(s)+len(q))
    require(list(s.shape[1:])==list(q.shape[1:])==[2,256], 'Wrong IQ layout')
    require(s.dtype==q.dtype==np.float32 and y.tolist()==split['support_labels'], 'Wrong IQ type/support labels')
    require(len(q)==manifest['query_count']==len(qi) and len(s)==len(si), 'Wrong record count')
    require(len(set(qi))==len(qi) and len(set(si))==len(si) and not set(si)&set(qi), 'Repeated/overlapping physical IDs')
    return dict(manifest=manifest,split=split,support_x=s,support_y=y,support_ids=si,query_x=q,query_ids=qi)
