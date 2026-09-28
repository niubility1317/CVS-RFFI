"""End-to-end synthetic received cache and support-only MV-KME predictions."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import predict_d92_mv_kme as mod
import export_d92_mv_kme_features as exporter
from test_export_d92_mv_kme_features import fixture,dump


def inputs(tmp_path,monkeypatch,k=1):
    args,_,_=fixture(tmp_path,monkeypatch,k)
    with threadpool_limits(limits=1):exporter.export(**args)
    return {**{key:args[key] for key in ('row_root','capsule','config','expected_capsule_id','expected_checkpoint_sha256')},
        'mv_features':args['output'],'output':tmp_path/'predictions'}


@pytest.mark.parametrize('k',[1,2])
def test_support_fit_query_invariance_and_logs(tmp_path,monkeypatch,k):
    args=inputs(tmp_path,monkeypatch,k)
    actual=mod.fit_mv_kme;states=[]
    with np.load(args['mv_features']/'received_mv_features.npz') as data:blocks=data['blocks']
    def fit(**values):
        np.testing.assert_array_equal(values['support_blocks'],blocks[:4*k])
        assert values['support_ids'].tolist()==[f'physical-{i}' for i in range(4*k)]
        state=actual(**values);states.append(state);return state
    monkeypatch.setattr(mod,'fit_mv_kme',fit)
    original=Path.open
    def guarded(path,*a,**kw):
        assert path.name not in ('truth.json','scores.json','final_ssdg.pth','source_l_features.npz')
        assert 'ground' not in path.parts
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    with threadpool_limits(limits=1):
        mod.predict(**args)
        query=blocks[4*k:];state=states[0]
        np.testing.assert_array_equal(state.score(query),np.concatenate([state.score(q[None]) for q in query]))
        np.testing.assert_array_equal(state.score(query),state.score(query[::-1])[::-1])
    record=mod.read(args['output']/'predictions.jsonl')
    assert record['mode']=='d92_mvkme_registration'
    assert record['query_ids']==[f'physical-{i}' for i in range(4*k,4*k+3)]
    small=mod.read(args['output']/'compact.jsonl')
    assert small['candidate_count']==(0 if k==1 else 9) and small['fold_count']==(0 if k==1 else 2)
    assert small['source_rows_used_for_fit']==small['query_rows_used_for_fit']==0
    assert (small['selected_old_nll'] is None)==(k==1)
    assert (small['selected_new_nll'] is None)==(k==1)
    assert small['learning_rate'] is small['gradient'] is small['source_validation'] is None
    assert 'Closed-form ridge' in small['unavailable_reason']
    assert (args['output']/'compact.csv').is_file()
    assert mod.read(args['output']/'predictions_complete.json')['payload_audit']['new_ground_statistics_bytes']==0
    with pytest.raises(FileExistsError):mod.predict(**args)


@pytest.mark.parametrize('fault',['sha','capsule','config','marker','provenance','cache_binding','ids','blocks','nonfinite','extra','truth','overlap'])
def test_illegal_cache_or_split_fails_before_fit(tmp_path,monkeypatch,fault):
    args=inputs(tmp_path,monkeypatch)
    if fault=='sha':args['expected_checkpoint_sha256']='b'*64
    elif fault=='capsule':args['expected_capsule_id']='other'
    elif fault=='config':args['config']['algorithm']['fourier_seed']=4
    elif fault in ('marker','provenance'):
        path=args['mv_features']/('features_complete.json' if fault=='marker' else 'checkpoint_provenance.json')
        value=mod.read(path)
        if fault=='marker':value['query_used_for_fitting']=True
        else:value['checkpoint_inheritance']=['other']
        dump(path,value)
    elif fault in ('truth','overlap'):
        path=args['capsule']/'splits/one.json';value=mod.read(path)
        if fault=='truth':value['query_labels']=[0,1,2]
        else:value['query_indices'][0]=0
        dump(path,value)
    else:
        path=args['mv_features']/'received_mv_features.npz'
        with np.load(path) as data:values={key:data[key] for key in data.files}
        if fault=='cache_binding':values['checkpoint_sha256']=np.asarray('c'*64)
        elif fault=='ids':values['ids'][0]='different'
        elif fault=='blocks':values['blocks']=values['blocks'][:,:2]
        elif fault=='nonfinite':values['blocks'][0,0,0]=np.nan
        else:values['query_truth']=np.zeros(3)
        np.savez(path,**values)
    monkeypatch.setattr(mod,'fit_mv_kme',lambda **_:pytest.fail('Invalid input cannot fit'))
    with pytest.raises(ValueError):mod.predict(**args)
    assert not args['output'].exists()


def test_ties_follow_physical_classes():
    assert mod.stable_predictions(np.ones((2,3)),['z','a','m']).tolist()==[1,1]
