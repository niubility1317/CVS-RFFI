import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import predict_d92_sourcefree_head as mod


def dump(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path):
    row=tmp_path/'row';capsule=tmp_path/'capsule';out=tmp_path/'candidate'
    feature=row/'received_features';feature.mkdir(parents=True);(capsule/'splits').mkdir(parents=True)
    classes=[f'c{i}' for i in range(8)];ids=np.asarray([f'id{i}' for i in range(16)])
    rng=np.random.default_rng(31);x=rng.normal(size=(16,160));logits=rng.normal(size=(16,6))
    np.savez(feature/'received_features.npz',identity160=x,logits=logits,ids=ids)
    np.savez(capsule/'received.npz',ids=ids)
    dump(capsule/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='synthetic',split_count=1))
    split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='synthetic',split_id='one',
        registered_classes=classes,support_indices=list(range(8)),support_labels=list(range(8)),query_indices=list(range(8,16)),
        k=1,support_seed=23,receiver='rx',scenario='scene')
    dump(capsule/'splits/one.json',split)
    dump(row/'d92_startup.json',dict(checkpoint_sha256='frozen',capsule=str(capsule),features=str(feature/'received_features.npz'),
        query_fit_access=False,truth_read=False,seed=1))
    dump(feature/'features_complete.json',dict(status='FROZEN_FEATURES_COMPLETE',capsule_id='synthetic',query_used_for_fitting=False))
    dump(feature/'checkpoint_provenance.json',dict(target_access_before_freeze=False,checkpoint_inheritance=[],classes=classes[:6]))
    return dict(row_root=row,capsule=capsule,output=out,expected_capsule_id='synthetic',expected_checkpoint_sha256='frozen',config=dict(max_iter=300)),x


def test_only_support_fits_and_outputs_are_complete(tmp_path,monkeypatch):
    args,x=fixture(tmp_path);original=mod.fit_sourcefree_head;seen=[]
    def fit(**values):
        np.testing.assert_array_equal(values['support_features'],x[:8]);seen.append(len(values['support_features']))
        return original(**values)
    monkeypatch.setattr(mod,'fit_sourcefree_head',fit)
    mod.predict(**args)
    assert seen==[8]
    out=args['output'];marker=json.loads((out/'predictions_complete.json').read_text())
    assert marker['predictions']==1 and marker['source_data_access'] is False
    trace=json.loads((out/'fit_trace.jsonl').read_text())
    assert trace['query_rows_used_for_fit']==0 and trace['source_runtime_access'] is False
    record=json.loads((out/'predictions.jsonl').read_text())
    assert record['mode']=='d92_sfhead_registration' and len(record['classes'])==8
    assert record['query_ids']==[f'id{i}' for i in range(8,16)]
    assert (out/'compact.csv').is_file()
    with pytest.raises(FileExistsError):mod.predict(**args)


@pytest.mark.parametrize('fault',['truth','overlap','checkpoint','feature_marker'])
def test_bad_bindings_fail_before_training_or_output(tmp_path,monkeypatch,fault):
    args,_=fixture(tmp_path)
    if fault in ('truth','overlap'):
        p=args['capsule']/'splits/one.json';v=json.loads(p.read_text())
        if fault=='truth':v['query_labels']=[0]*8
        else:v['query_indices'][0]=0
        dump(p,v)
    elif fault=='checkpoint':args['expected_checkpoint_sha256']='wrong'
    else:
        p=args['row_root']/'received_features/features_complete.json';v=json.loads(p.read_text());v['query_used_for_fitting']=True;dump(p,v)
    def no_fit(**_):raise AssertionError('Training must not start')
    monkeypatch.setattr(mod,'fit_sourcefree_head',no_fit)
    with pytest.raises(ValueError):mod.predict(**args)
    assert not args['output'].exists()
