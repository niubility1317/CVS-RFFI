"""BNNA input isolation and real core step logging using synthetic support."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import predict_d92_bnna as mod
import export_d92_bnna_features as exporter
from test_export_d92_bnna_features import fixture,dump


def inputs(tmp_path,monkeypatch,k=1):
    args,_,_=fixture(tmp_path,monkeypatch,k)
    with threadpool_limits(limits=1):exporter.export(**args)
    return {**{key:args[key] for key in ('row_root','capsule','config','expected_capsule_id','expected_checkpoint_sha256')},
        'bnna_features':args['output'],'output':tmp_path/'predictions'}


@pytest.mark.parametrize('k',[1,2])
def test_only_support_fit_and_real_step_logs(tmp_path,monkeypatch,k):
    args=inputs(tmp_path,monkeypatch,k)
    with np.load(args['bnna_features']/exporter.CACHE_NAME) as data:z,f=data['identity_views'],data['fft']
    original_fit=mod.fit_bnna;states=[]
    def fit(**values):
        np.testing.assert_array_equal(values['support_identity_views'],z[:4*k])
        np.testing.assert_array_equal(values['support_fft'],f[:4*k])
        assert values['support_ids'].tolist()==[f'physical-{i}' for i in range(4*k)]
        state=original_fit(**values);states.append(state);return state
    monkeypatch.setattr(mod,'fit_bnna',fit)
    original=Path.open
    def guarded(path,*a,**kw):
        assert path.name not in ('truth.json','scores.json','source_l_features.npz','final_ssdg.pth') and 'ground' not in path.parts
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    with threadpool_limits(limits=1):
        mod.predict(**args);state=states[0]
        score=state.score(z[4*k:],f[4*k:])
        singles=np.concatenate([state.score(z[i:i+1],f[i:i+1]) for i in range(4*k,len(z))])
        np.testing.assert_array_equal(score,singles)
        np.testing.assert_array_equal(score,state.score(z[4*k:][::-1],f[4*k:][::-1])[::-1])
    out=args['output'];record=mod.read(out/'predictions.jsonl');small=mod.read(out/'compact.jsonl');trace=mod.read(out/'fit_trace.jsonl')
    assert record['mode']=='d92_bnna_registration' and record['query_ids']==[f'physical-{i}' for i in range(4*k,len(z))]
    assert small['fold_count']==(0 if k==1 else 2)
    assert small['source_rows_used_for_fit']==small['query_rows_used_for_fit']==0
    assert small['source_validation'] is None and 'user prohibits' in small['source_validation_reason']
    assert small['epoch'] is None and small['epoch_reason']
    assert all(small[key]>=0 for key in ('fit_call_seconds','query_score_seconds','prediction_write_seconds'))
    steps=[json.loads(line) for line in (out/'training_steps.jsonl').read_text().splitlines()]
    assert len(steps)==len(trace['steps'])==small['optimizer_steps_recorded']
    assert steps and all(not isinstance(value,(dict,list)) for step in steps for value in step.values())
    with (out/'training_steps.csv').open(newline='',encoding='utf-8') as stream:assert len(list(csv.DictReader(stream)))==len(steps)
    for assignment in trace['physical_fold_assignment']:
        i=int(assignment['physical_id'].split('-')[-1]);assert assignment['class_id']==trace['registered_classes'][i//k]
    payload=mod.read(out/'startup.json')['payload_audit']
    assert payload['new_ground_statistics_bytes']==0 and payload['model_file_bytes']==123456
    assert payload['model_already_deployed'] is None and payload['model_incremental_transfer_bytes'] is None
    assert 'unknown' in payload['model_deployment_unknown_reason']
    assert 'Complete training checkpoint package' in payload['model_file_bytes_scope']
    assert mod.read(out/'predictions_complete.json')['payload_audit']==payload
    with pytest.raises(FileExistsError):mod.predict(**args)


@pytest.mark.parametrize('fault',['sha','capsule','config','marker','provenance','binding','ids','shape','nonfinite','extra','truth','overlap'])
def test_bad_cache_or_split_never_fits(tmp_path,monkeypatch,fault):
    args=inputs(tmp_path,monkeypatch)
    if fault=='sha':args['expected_checkpoint_sha256']='b'*64
    elif fault=='capsule':args['expected_capsule_id']='other'
    elif fault=='config':args['config']['algorithm']['unexpected']=True
    elif fault in ('marker','provenance'):
        path=args['bnna_features']/('features_complete.json' if fault=='marker' else 'checkpoint_provenance.json');value=mod.read(path)
        if fault=='marker':value['query_used_for_fitting']=True
        else:value['checkpoint_inheritance']=['other']
        dump(path,value)
    elif fault in ('truth','overlap'):
        path=args['capsule']/'splits/one.json';value=mod.read(path)
        if fault=='truth':value['query_labels']=[0,1,2]
        else:value['query_indices'][0]=0
        dump(path,value)
    else:
        path=args['bnna_features']/exporter.CACHE_NAME
        with np.load(path) as data:values={key:data[key] for key in data.files}
        if fault=='binding':values['checkpoint_sha256']=np.asarray('c'*64)
        elif fault=='ids':values['ids'][0]='different'
        elif fault=='shape':values['identity_views']=values['identity_views'][:,:3]
        elif fault=='nonfinite':values['fft'][0,0]=np.nan
        else:values['query_truth']=np.zeros(3)
        np.savez(path,**values)
    monkeypatch.setattr(mod,'fit_bnna',lambda **_:pytest.fail('Invalid input must not fit'))
    with pytest.raises(ValueError):mod.predict(**args)
    assert not args['output'].exists()


def test_physical_class_ties():
    assert mod.stable_predictions(np.ones((2,3)),['z','a','b']).tolist()==[1,1]


def test_zero_view_variation_records_zero_steps_without_fabricated_metrics(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch)
    path=args['bnna_features']/exporter.CACHE_NAME
    with np.load(path) as data:values={key:data[key] for key in data.files}
    values['identity_views']=np.repeat(values['identity_views'][:,:1],4,axis=1)
    np.savez(path,**values)
    with threadpool_limits(limits=1):mod.predict(**args)
    small=mod.read(args['output']/'compact.jsonl')
    assert small['active_rank']==0 and small['optimizer_steps_recorded']==0 and small['final_step'] is None
    assert small['final_fit']['status']=='NO_VIEW_VARIATION'
    assert (args['output']/'training_steps.jsonl').read_text()==''
    with (args['output']/'training_steps.csv').open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==mod.STEP_FIELDS and list(reader)==[]
