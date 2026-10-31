"""Source-Us replay keeps four old decisions and forwards only the new student."""
import importlib
import numpy as np
import pytest
import torch
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_us_run')


def test_controls_select_source_and_pcd_without_changing_public_identity():
    m=api();a=dict(ids=np.array(['a','b']),closed=np.array([0,1],np.int64),
        signal_features=np.arange(960,dtype=np.float32).reshape(2,3,160),
        raw_scores=np.arange(12,dtype=np.float64).reshape(2,6),scores=np.arange(12,dtype=np.float64).reshape(2,6)-2)
    b=m._controls(a)
    np.testing.assert_array_equal(b['signal_features'],a['signal_features'][:,1:3])
    for k in ('scores','raw_scores'):np.testing.assert_array_equal(b[k],a[k][:,2:6])
    np.testing.assert_array_equal(b['closed'],a['closed'])
    b['scores'][0,0]=99
    assert a['scores'][0,2]==0


def test_query_forward_uses_one_restoration_f0_and_new_student_only():
    m=api();seen=[]
    class Reader(torch.nn.Module):
        def features(self,x):seen.append('F0');return x.mean((1,2))[:,None].repeat(1,160)
        def classify_features(self,z):return z[:,:6]
    class Student(torch.nn.Module):
        def forward(self,x):seen.append('new');return torch.nn.functional.normalize(torch.arange(1,161,dtype=x.dtype)[None].repeat(len(x),1),dim=1)
    def restore(x):seen.append('G');return (x*.7,)
    a=m._forward(dict(reader=Reader(),dura=(None,restore),model=lambda *a:pytest.fail('No old model forward')),Student(),np.ones((3,2,256),np.float32),'cpu')
    assert seen==['G','F0','new'] and a['signal_features'].shape==(3,160)
    np.testing.assert_array_equal(a['closed'],a['baseline_closed'])


def test_extra_truth_fields_fail_before_any_training_or_source_read():
    m=api()
    with pytest.raises(ValueError):m.validate_config(dict(run_id=m.RUN_ID,code_commit='a'*40,train_config={},source_control_config={},Q_conditions={},output_root='new',batch_size=128,Us_truth='forbidden'))


@contextmanager
def fixture(root,mp):
    # New-checkpoint strict loading is covered by source_us_train's actual
    # two-step case. This handoff reuses a real frozen SignalOpen toy, without
    # pretending that this test runs the production 444-step training budget.
    from test_pamf_phase3_fresh_world_run import fixture as parent
    with parent(root,mp) as (old,ref,_,b,events,old_identity):
        m=api();ref['code_commit']=m.REFERENCE_CODE;ref['train_config']['code_commit']=m.REFERENCE_CODE;old_identity['code_commit']=m.REFERENCE_CODE
        pm=old.run(ref,'cpu');mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',old.Q_COUNT)
        tc=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=deepcopy(ref['source_control_config']),Us_public_root=str(root/'Us/public'),
            Us_capsule_id='test-source-Us',Us_split_id='test-source-Us-split',Us_expected_count=56700,output_root=str(root/'new_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,source_control_config=deepcopy(ref),Q_conditions=deepcopy(ref['Q_conditions']),output_root=str(root/'new_predict'),batch_size=128)
        train=SimpleNamespace(STEPS=444,STATUS='SOURCE_US_REPLAY_TRAINING_COMPLETE',validate_config=lambda c:c)
        mp.setattr(m,'_train',lambda:train);model=deepcopy(b['model'])
        with torch.no_grad():model.projection.network[2].bias.add_(.015)
        model.eval().requires_grad_(False)
        actual_train=importlib.import_module('experiments.pamf_phase3_open_world.source_us_train')
        iqroot=Path(tc['reference_config']['reference_config']['train_config']['IQ_root'])
        manifest=dict(source_manifests={role:m._read(iqroot/role/'manifest.json') for role in ('L','OE')})
        identity=actual_train._identity(tc,pm['model_identities']['E0'],manifest,
            dict(test_only_public_manifest=True),dict(identity=pm['model_identities']['SOURCE']))
        Path(tc['output_root']).mkdir()
        m._write(Path(tc['output_root'])/'complete.json',dict(status=train.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],config=tc,identity=identity,
            updates=444,total_optimizer_updates=444,native_updates=0,identity_head_updates=0,source_only=True,target_U_used=False,target_U_reads=0,
            source_sampling_matched=True,Us_ground_truth_reads=0,private_truth_reads=0,test_only_simulated_training=True))
        mp.setattr(m,'_load_model',lambda *a:(model,identity));mp.setattr(m,'_backend',lambda *a:b)
        mp.setattr(b['model'],'forward',lambda *a:pytest.fail('No old r03 model forward during new evaluation'))
        calls=[];load=m._iq_role;public=m._public;forward=m._forward
        def role(config,name,header):
            out=Path(c['output_root']);assert (out/'smoke.json').exists()
            if name=='V':assert (out/'models/complete.json').exists()
            if name=='novel':assert (out/'calibration/complete.json').exists()
            calls.append(name);return load(config,name,header)
        def query(config,name):
            assert Path(c['output_root'],'source/source_DURA/novel_predictions.npz').exists();calls.append(name);return public(config,name)
        forward_rows=[]
        def counted(*args):forward_rows.append(len(args[2]));return forward(*args)
        mp.setattr(m,'_iq_role',role);mp.setattr(m,'_public',query);mp.setattr(m,'_forward',counted)
        yield m,c,pm,calls,forward_rows
