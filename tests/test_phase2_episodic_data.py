import numpy as np
import pytest
from experiments.pamf_phase2_practical.episodic_io import support_order,task_indices,observation_id,query_complement


def test_query_is_full_pool_complement_and_roles_can_switch_across_k():
    orders=[support_order(np.arange(c*100,(c+1)*100),17+c) for c in range(3)]
    s1,q1,y1=task_indices(orders,2,1);s5,q5,y5=task_indices(orders,2,5)
    assert len(s1)==2 and len(q1)==198 and len(s5)==10 and len(q5)==190
    assert set(s1)<=set(s5) and set(s5)-set(s1)<=set(q1)
    assert not set(s5)&set(q5) and set(s5)|set(q5)==set(range(200))
    assert y5.tolist()==[0]*5+[1]*5
    assert not np.isin(np.arange(200,300),np.r_[s5,q5]).any()


def test_episodes_reuse_entire_raw_pool_but_resample_support():
    a=support_order(np.arange(5000),11);b=support_order(np.arange(5000),12)
    assert set(a)==set(b)==set(range(5000)) and not np.array_equal(a[:20],b[:20])
    assert np.array_equal(a,support_order(np.arange(5000),11))
    assert observation_id('physical-record','episode1')!=observation_id('physical-record','episode2')


def test_insufficient_pool_does_not_create_samples_or_drop_query():
    with pytest.raises(ValueError,match='nonempty query'):
        task_indices([np.arange(20)],1,20)
    with pytest.raises(ValueError,match='Illegal support'):
        query_complement(np.arange(10),np.array([9,10]))


def test_query_reader_excludes_support_and_unregistered_iq(tmp_path):
    import json
    from experiments.pamf_phase2_practical.episodic_io import load_task
    root=tmp_path; (root/'stage_rows').mkdir();(root/'splits').mkdir();(root/'support_labels').mkdir()
    ids=np.array([f'observation-{i}' for i in range(16)])
    bank=np.arange(16,dtype=np.float32)[:,None,None]*np.ones((16,2,256),np.float32)
    np.save(root/'received.npy',bank);np.save(root/'ids.npy',ids)
    np.save(root/'stage_rows/N00.npy',np.arange(12,dtype=np.int32))
    manifest=dict(storage='episodic-pool-complement-v1',protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='e1',received_count=16)
    (root/'manifest.json').write_text(json.dumps(manifest))
    split_id='a'*24
    split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='e1',split_id=split_id,new_count=0,k=1,
               registered_classes=list('abcdef'),support_indices=list(range(0,12,2)),query_count=6)
    (root/'splits'/(split_id+'.json')).write_text(json.dumps(split))
    # No label file yet: A must succeed and read only the six legal query rows.
    a=load_task(root,split_id,stage='A')
    assert a['query_x'][:,0,0].tolist()==list(range(1,12,2)) and 'support_x' not in a
    np.save(root/'support_labels'/(split_id+'.npy'),np.arange(6,dtype=np.int16))
    b=load_task(root,split_id,stage='B')
    assert b['support_x'][:,0,0].tolist()==list(range(0,12,2))
    assert not np.isin(bank[12:,0,0],np.r_[b['support_x'][:,0,0],b['query_x'][:,0,0]]).any()


def test_channel_replicates_are_fresh_and_reproducible():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    from experiments.pamf_phase2_practical.data import observe
    mixture=dict(route={'fs_hz':25e6},receiver_seed=2027)
    scenes=[dict(id='practical_low_urban',probability=1.0,mixed_conditional_elevation_range_deg=[10,30])]
    raw=np.stack([np.sin(np.arange(256)*.1),np.cos(np.arange(256)*.1)]).astype(np.float32)
    pool1=dict(id='E1',scenario='practical_low_urban',scene_seed=101,channel_seed=201)
    pool2=dict(id='E2',scenario='practical_low_urban',scene_seed=102,channel_seed=202)
    y,m=observe(raw,scenes,sid='one-raw-record',session='rx:r/day:d',pool=pool1,mixture=mixture)
    same,_=observe(raw,scenes,sid='one-raw-record',session='rx:r/day:d',pool=pool1,mixture=mixture)
    other,mo=observe(raw,scenes,sid='one-raw-record',session='rx:r/day:d',pool=pool2,mixture=mixture)
    assert np.array_equal(y,same) and not np.array_equal(y,other) and m['overlay_seed']!=mo['overlay_seed']
