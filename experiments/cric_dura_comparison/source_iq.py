"""Bind existing source IQ storage rows to declared physical IDs, without labels."""
from pathlib import Path
import numpy as np


def aligned_source_iq(root,role,expected_ids):
    if role not in ('L_s','U_s','V'):
        raise ValueError('Source roles only')
    root=Path(root);path=root/(role+'_x.npy');actual=path.resolve()
    x=np.load(path,mmap_mode='r',allow_pickle=False)
    expected=np.asarray(expected_ids)
    index=actual.parent/(role+'_index.npz')
    if not index.is_file():
        index=actual.parent/(role+'_metadata.npz')
    if not index.is_file():
        raise ValueError('Existing source IQ physical ID index missing')
    # Only ids is opened. U transmitter labels are never loaded or derived.
    with np.load(index,allow_pickle=False) as values:
        stored=values['ids'].copy()
    if (expected.ndim!=1 or stored.ndim!=1 or x.shape!=(len(stored),2,256) or
        len(set(stored.tolist()))!=len(stored) or len(set(expected.tolist()))!=len(expected) or
        set(expected.tolist())!=set(stored.tolist())):
        raise ValueError('Existing source IQ physical ID binding differs')
    if np.array_equal(stored,expected):
        changed=0
    else:
        lookup={str(s):i for i,s in enumerate(stored)}
        positions=np.asarray([lookup[str(s)] for s in expected],dtype=np.int64)
        changed=int(np.count_nonzero(positions!=np.arange(len(positions))))
        x=x[positions]
    return x,dict(role=role,count=len(expected),reordered_positions=changed,
        storage_index=str(index),binding='exact physical ID lookup',U_TX_truth_reads=0)
