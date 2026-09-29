"""Synthetic four-view cache boundaries; no real data or checkpoint."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import export_d92_branch_orbit_support_features as mod
import export_d92_branch_support_features as old
from test_export_d92_branch_support_features import fixture as base_fixture, dump


@pytest.fixture
def fixture(base_fixture, monkeypatch):
    monkeypatch.setattr(mod, 'load_native', old.load_native)
    return base_fixture


def binding(args):
    return dict(support_features=args['output'], capsule=args['capsule'],
        expected_capsule_id=args['expected_capsule_id'],
        expected_checkpoint_sha256=args['expected_checkpoint_sha256'], expected_model_seed=args['seed'],
        config=dict(algorithm={}, matrix=dict(receivers=['target-rx'], scenarios=['test-scene'],
                    ks=[1,2], new_counts=[1], support_seeds=[12])))


def test_export_only_support_single_view_calls_and_exact_original(fixture, monkeypatch):
    args, iq, ids, support, query, infer = fixture
    np_load = np.load
    original_get = np.memmap.__getitem__; reads = []
    def get(array, index):
        assert isinstance(index, np.ndarray) and set(index).issubset(support)
        reads.extend(index.tolist()); return original_get(array, index)
    monkeypatch.setattr(np.memmap, '__getitem__', get)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('No eager IQ load'))
    monkeypatch.setattr(torch.Tensor, 'numpy', lambda *a, **kw: pytest.fail('No Tensor.numpy ABI bridge'))
    monkeypatch.setattr(torch, 'as_tensor', lambda *a, **kw: pytest.fail('No torch.as_tensor ABI bridge'))
    forward = infer.forward.backbone_forward_compat
    def singleton(model, x, **kwargs):
        assert len(x) == 1
        return forward(model,x,**kwargs)
    monkeypatch.setattr(infer.forward, 'backbone_forward_compat', singleton)
    marker = mod.export(**args)
    assert len(reads) == len(support) and set(reads) == set(support)
    assert not set(reads)&set(query)
    assert marker['status'] == 'BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE'
    assert marker['count'] == marker['support_iq_rows_read'] == 6
    assert marker['native_view_forward_count'] == marker['native_batch_calls'] == 24
    assert marker['native_total_batch_calls'] == infer.batch_calls == 28
    assert marker['smoke_forward_count'] == 4
    assert marker['feature_array_bytes'] == 6*(4*4*160+96)*4
    assert marker['view_count_per_observation'] == 4
    monkeypatch.setattr(np, 'load', np_load)
    arrays, tasks, classes, producer, startup, provenance = mod.load_support(**binding(args))
    assert len(tasks) == 2 and classes == ['old-z','old-a']
    assert startup['provenance'] == provenance
    assert arrays['fft'].shape == (6,96)
    assert all(arrays[k].shape == (6,4,160) for k in mod.BRANCHES)
    assert all(not v.flags.writeable for v in arrays.values())
    positions = sorted(support,key=lambda i:ids[i])
    for row, index in enumerate(positions):
        expected = infer(iq[index:index+1])
        for key in mod.BRANCHES:
            np.testing.assert_array_equal(arrays[key][row,0], expected[key][0])
    np.testing.assert_array_equal(arrays['fft'], old.historical_fft96(iq[positions]))
    with pytest.raises(FileExistsError): mod.export(**args)


def test_phase_order_and_physical_permutation_are_exact(fixture):
    _,iq,_,support,_,infer=fixture
    rows=iq[support[:2]]; views=mod.make_received_views(rows)
    np.testing.assert_array_equal(views[:,0],rows)
    np.testing.assert_array_equal(views[:,1,0],-rows[:,1])
    np.testing.assert_array_equal(views[:,1,1],rows[:,0])
    np.testing.assert_array_equal(views[:,2],-rows)
    np.testing.assert_array_equal(views[:,3,0],rows[:,1])
    np.testing.assert_array_equal(views[:,3,1],-rows[:,0])
    batched=mod.transform_support(rows,infer)
    reversed_=mod.transform_support(rows[::-1],infer)
    for key in mod.BRANCHES:
        np.testing.assert_array_equal(batched[key],reversed_[key][::-1])
        np.testing.assert_array_equal(batched[key],np.concatenate([mod.transform_support(v[None],infer)[key] for v in rows]))


@pytest.mark.parametrize('fault',['capsule','sha','seed','query_flag','inherited','count','smoke','shape','labels','indices','partial','extra_member','dtype'])
def test_reader_rejects_wrong_binding_or_schema(fixture, fault):
    args,*_=fixture; mod.export(**args); kwargs=binding(args)
    markerpath=args['output']/'features_complete.json'; marker=mod.read(markerpath)
    if fault in ('capsule','sha','seed'):
        kwargs[{'capsule':'expected_capsule_id','sha':'expected_checkpoint_sha256','seed':'expected_model_seed'}[fault]]='wrong'
    elif fault in ('shape','labels','indices','extra_member','dtype'):
        path=args['output']/mod.CACHE_NAME
        with np.load(path,allow_pickle=False) as z: data={k:z[k] for k in z.files}
        if fault=='shape': data['z_id']=data['z_id'][:,0]
        if fault=='labels': data['labels'][0]='wrong'
        if fault=='indices': data['indices'][0]=999
        if fault=='extra_member': data['query_truth']=np.asarray([0])
        if fault=='dtype': data['z_id']=data['z_id'].astype(np.float64)
        np.savez(path,**data)
    elif fault=='partial':
        p=args['output']/'support_splits.json'; plan=mod.read(p); plan['splits'].pop(); dump(p,plan)
    else:
        if fault=='query_flag': marker['query_iq_access']=True
        if fault=='inherited': marker['adapted_state_inherited']=True
        if fault=='count': marker['native_batch_calls']-=1
        if fault=='smoke': marker['synthetic_smoke']['status']='FAIL'
        dump(markerpath,marker)
    with pytest.raises(ValueError):mod.load_support(**kwargs)


def test_smoke_failure_reads_no_received_and_preserves_failure(fixture, monkeypatch):
    args,*_=fixture
    monkeypatch.setattr(mod.SupportIQ,'take',lambda *a:pytest.fail('No support before smoke'))
    monkeypatch.setattr(mod,'transform_support',lambda *a: (_ for _ in ()).throw(ValueError('bad smoke')))
    with pytest.raises(ValueError,match='bad smoke'):mod.export(**args)
    assert mod.read(args['output']/'technical_failure.json')['status']=='TECHNICAL_FAILURE'
    assert not (args['output']/'features_complete.json').exists()


def test_reader_never_opens_iq_or_weights(fixture, monkeypatch):
    args,*_=fixture; mod.export(**args)
    args['capsule'].joinpath('received.npz').unlink()
    monkeypatch.setattr(mod,'load_native',lambda **kw:pytest.fail('No checkpoint reload'))
    arrays,*_=mod.load_support(**binding(args))
    assert arrays['z_id'].shape==(6,4,160)


def test_no_overwrite_on_output_race(fixture, monkeypatch):
    args,*_=fixture; original=mod.load_native
    def load(**kw):
        result=original(**kw); args['output'].mkdir(); (args['output']/'owner').write_text('preserve'); return result
    monkeypatch.setattr(mod,'load_native',load)
    with pytest.raises(FileExistsError):mod.export(**args)
    assert [p.name for p in args['output'].iterdir()]==['owner']
