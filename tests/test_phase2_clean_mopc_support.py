import json
import numpy as np
import pytest
from experiments.pamf_phase2_practical.clean_mopc_diagnostic import support_only_clean


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def test_clean_support_opens_only_registered_support(tmp_path, monkeypatch):
    data=tmp_path/'data'; clean=tmp_path/'clean'; cap=data/'capsules/P01/MIX01/1-1'
    sid='a'*24
    manifest=dict(storage='episodic-pool-complement-v1',protocol_schema='p2_min_v1',
                  phase2_data_status='VALIDATED_ONCE',capsule_id='cap',receiver='1-1',received_count=24)
    split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='cap',
               split_id=sid,k=1,new_count=2,registered_classes=list(range(8)),support_indices=list(range(8)))
    write(cap/'manifest.json',manifest);write(cap/'splits'/(sid+'.json'),split)
    (cap/'support_labels').mkdir();np.save(cap/'support_labels'/(sid+'.npy'),np.arange(8))
    write(clean/'completion.json',dict(status='CLEAN_PAIRED_COMPLETE',parent_task_root=str(data)))
    cc=clean/'capsules/P01/MIX01/1-1';write(cc/'manifest.json',dict(capsule_id='cap',receiver='1-1',
                                           input_condition='clean_ground_pre_satellite_normalized'))
    np.save(cc/'row_to_clean.npy',np.arange(24,dtype=np.int32))
    bank=clean/'banks/1-1';bank.mkdir(parents=True);np.save(bank/'clean.npy',np.arange(24*2*256,dtype=np.float32).reshape(24,2,256))
    original=np.load
    def only_support(path,*args,**kwargs):
        assert 'score_only' not in str(path) and 'stage_rows' not in str(path) and 'received.npy' not in str(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',only_support)
    x,y=support_only_clean(data,clean,'capsules/P01/MIX01/1-1',sid)
    assert x.shape==(2,2,256) and y.tolist()==[6,7]
    np.testing.assert_array_equal(x,original(bank/'clean.npy')[6:8])


def test_clean_support_rejects_mismatched_data_binding(tmp_path):
    data=tmp_path/'data';clean=tmp_path/'clean';cap=data/'capsules/P01/MIX01/1-1';sid='a'*24
    write(cap/'manifest.json',dict(capsule_id='cap',receiver='1-1',received_count=8,storage='episodic-pool-complement-v1',
          protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE'))
    write(cap/'splits'/(sid+'.json'),dict(split_id=sid,capsule_id='cap',k=1,new_count=2,
          registered_classes=list(range(8)),support_indices=list(range(8)),protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE'))
    write(clean/'completion.json',dict(status='CLEAN_PAIRED_COMPLETE',parent_task_root=str(tmp_path/'other')))
    with pytest.raises(ValueError,match='Clean pairing'):
        support_only_clean(data,clean,'capsules/P01/MIX01/1-1',sid)
