"""Read original clean roles without importing historical model packages."""
from pathlib import Path
import numpy as np


def aligned_source_iq(root,role,expected_ids):
    if role not in ('L_s','U_s','V'):raise ValueError('Source roles only')
    root=Path(root);path=root/(role+'_x.npy');actual=path.resolve()
    x=np.load(path,mmap_mode='r',allow_pickle=False);expected=np.asarray(expected_ids)
    index=actual.parent/(role+'_index.npz')
    if not index.is_file():index=actual.parent/(role+'_metadata.npz')
    if not index.is_file():raise ValueError('Existing source IQ physical ID index missing')
    with np.load(index,allow_pickle=False) as values:stored=values['ids'].copy()
    if (expected.ndim!=1 or stored.ndim!=1 or x.shape!=(len(stored),2,256) or x.dtype!=np.float32
            or len(set(stored.tolist()))!=len(stored) or len(set(expected.tolist()))!=len(expected)
            or set(expected.tolist())!=set(stored.tolist())):raise ValueError('Existing source IQ physical ID binding differs')
    if np.array_equal(stored,expected):changed=0
    else:
        lookup={str(s):i for i,s in enumerate(stored)}
        positions=np.asarray([lookup[str(s)] for s in expected],np.int64)
        changed=int(np.count_nonzero(positions!=np.arange(len(positions))));previous=x;x=x[positions];previous._mmap.close()
    return x,dict(role=role,count=len(expected),reordered_positions=changed,storage_index=str(index),binding='exact physical ID lookup',U_TX_truth_reads=0)


class SourceDataset:
    """Original metadata contract with only ids read from the storage index."""
    def __init__(self,root,role,contract):
        if role not in ('L_s','U_s','V'):raise ValueError('Only source roles allowed')
        root=Path(root)
        with np.load(root/(role+'_metadata.npz'),allow_pickle=False) as values:
            names=set(values.files);allowed={'rx_i','day_i','eq_i','sig_i','base_index','domain','ids'}
            if role!='U_s':allowed.add('y')
            if names!=allowed:raise ValueError('U truth or missing physical metadata: '+repr(names))
            self.metadata={k:values[k].copy() for k in names}
        if list(self.metadata['ids'])!=contract['role_ids'][role]:raise ValueError('Source role physical order differs')
        self.x,self.iq_binding=aligned_source_iq(root,role,self.metadata['ids'])
        if self.x.shape!=(contract['counts'][role],2,256):raise ValueError('Source role count differs')
        self.role=role
