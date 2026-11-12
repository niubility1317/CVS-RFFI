import json
import numpy as np
import pytest

from experiments.pamf_phase2_practical.clean_restore import build


def test_canonical_cache_reuses_each_record_and_protects_output(tmp_path):
    clean=tmp_path/'clean';(clean/'banks'/'1-1').mkdir(parents=True)
    (clean/'completion.json').write_text(json.dumps({'status':'CLEAN_PAIRED_COMPLETE','receivers':{'1-1':5}}))
    raw=np.arange(5*2*256,dtype=np.float32).reshape(5,2,256)
    np.save(clean/'banks'/'1-1'/'clean.npy',raw)
    calls=[]
    def restore(x):calls.append(len(x));return x+np.float32(2)
    result=build(clean,tmp_path/'out',restore,batch_size=2,dura_spec={'id':'test'},receivers=('1-1',))
    np.testing.assert_array_equal(np.load(tmp_path/'out/banks/1-1/restored.npy'),raw+2)
    assert calls==[2,2,1] and result['status']=='CLEAN_DURA_SUBSET'
    with pytest.raises(FileExistsError):build(clean,tmp_path/'out',restore,batch_size=2,dura_spec={},receivers=('1-1',))


def test_nonfinite_output_never_publishes_completion(tmp_path):
    clean=tmp_path/'clean';(clean/'banks'/'1-1').mkdir(parents=True)
    (clean/'completion.json').write_text(json.dumps({'status':'CLEAN_PAIRED_COMPLETE','receivers':{'1-1':1}}))
    np.save(clean/'banks'/'1-1'/'clean.npy',np.ones((1,2,256),np.float32))
    with pytest.raises(ValueError):
        build(clean,tmp_path/'out',lambda x:x*np.float32(np.nan),batch_size=1,dura_spec={},receivers=('1-1',))
    assert not (tmp_path/'out/completion.json').exists()
