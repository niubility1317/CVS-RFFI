import pytest
import numpy as np

from experiments.pamf_phase2_practical import clean_orthogonal_source as source
from experiments.practical_full_2x2.cache import Reader, atomic, key, probability


def test_source_preparation_rejects_target_and_existing_output(tmp_path):
    release=tmp_path/'release';release.mkdir()
    data=tmp_path/'source';data.mkdir()
    contract=tmp_path/'contract.json';contract.write_text('{}')
    root=tmp_path/'run'
    config=dict(run_root=str(root),release_root=str(release),source_cache=str(data),
                source_contract=str(contract),augmentation_cache=str(root/'augmentation'))
    assert source.check(config,2026092701)['target_reads']==0
    with pytest.raises(ValueError,match='target/query'):
        source.check(dict(config,data_root='forbidden'),2026092701)
    with pytest.raises(ValueError,match='seed'):
        source.check(config,1)
    output=root/'source/orthogonal-s2026092701';output.mkdir(parents=True)
    with pytest.raises(ValueError,match='already exists'):
        source.check(config,2026092701)


def test_complete_source_augmentation_is_consumed_read_only(tmp_path):
    release=tmp_path/'release';release.mkdir()
    data=tmp_path/'source';data.mkdir()
    contract=tmp_path/'contract.json'
    atomic(contract,{'role_ids':{'L_s':['physical-1']}})
    cache=tmp_path/'existing'
    config=dict(run_root=str(tmp_path/'new'),release_root=str(release),source_cache=str(data),
                source_contract=str(contract),augmentation_cache=str(cache),protocol_id='F0',
                reuse_source_augmentation=True)
    for epoch in range(1,201):
        p=probability((epoch-1)/200)
        folder=cache/'F0'/key(epoch,1,p);folder.mkdir(parents=True)
        request=dict(protocol_id='F0',epoch=epoch,stage=1,probability=p,
                     source_cache=str(data),source_contract=str(contract))
        atomic(folder/'complete.json',dict(status='L_AUGMENTATION_READY',count=1,request=request,
            U_reads=0,V_reads=0,target_reads=0,dtype='float32',extra_normalization=False))
        np.save(folder/'ids.npy',np.array(['physical-1']))
        np.save(folder/'x.npy',np.ones((1,2,256),dtype=np.float32))
        np.save(folder/'scene.npy',np.array([-1],dtype=np.int8))
    assert source.check(config,2026092701)['reused_augmentation_epochs']==200
    values,_=Reader(config).get([0],1,1,0,200)
    assert np.array_equal(values,np.ones((1,2,256),dtype=np.float32))
    assert not (cache/'requests').exists()
    with pytest.raises(ValueError,match='cache mismatch'):
        source.check(dict(config,source_cache=str(release)),2026092701)
