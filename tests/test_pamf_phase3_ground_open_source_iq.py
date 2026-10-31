"""The released source loader binds storage by opaque ID, never Us labels."""
import numpy as np
import pytest
from experiments.cric_dura_comparison.source_iq import aligned_source_iq


def test_source_storage_reorders_without_opening_hidden_labels(tmp_path,monkeypatch,subtests):
    stored=np.asarray(['b','c','a']);expected=np.asarray(['a','b','c'])
    x=np.arange(3*2*256,dtype=np.float32).reshape(3,2,256)
    np.save(tmp_path/'U_s_x.npy',x)
    np.savez(tmp_path/'U_s_index.npz',ids=stored,y=np.asarray([99,98,97]))
    original=np.load;opened=[]
    class PublicIndex:
        def __init__(self,z):self.z=z
        def __enter__(self):return self
        def __exit__(self,*args):self.z.close()
        def __getitem__(self,key):
            assert key=='ids','Source Us identity must remain hidden'
            opened.append(key);return self.z[key]
    def guarded(path,*args,**kwargs):
        z=original(path,*args,**kwargs)
        return PublicIndex(z) if str(path).endswith('_index.npz') else z
    monkeypatch.setattr(np,'load',guarded)
    actual,handle=aligned_source_iq(tmp_path,'U_s',expected)
    assert np.array_equal(actual,x[[2,0,1]]) and opened==['ids']
    assert handle['reordered_positions']==3 and handle['U_TX_truth_reads']==0
    for ids in (np.asarray(['a','b','d']),np.asarray(['a','b','b'])):
        with subtests.test(ids=ids.tolist()),pytest.raises(ValueError):aligned_source_iq(tmp_path,'U_s',ids)
