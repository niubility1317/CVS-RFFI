"""Input isolation, artifact binding, and byte reporting on synthetic observations."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
import predict_d92_summary_joint as mod
from test_d92_ground_summary import fixture_files, SHA, CLASSES


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path, k=1):
    row = tmp_path / 'row'
    feature = row / 'received_features'
    feature.mkdir(parents=True)
    ground = row / 'ground'
    ground.mkdir()
    fixture_files(ground)
    capsule = tmp_path / 'capsule'
    (capsule / 'splits').mkdir(parents=True)
    classes = [*CLASSES, 'new-0', 'new-1']
    support_count=4*k
    count=support_count+4
    ids = np.asarray([f'opaque-{i}' for i in range(count)])
    rng = np.random.default_rng(31)
    identity = rng.normal(size=(count, 160)).astype(np.float32)
    iq = rng.normal(size=(count, 2, 256)).astype(np.float32)
    np.savez(feature / 'received_features.npz', identity160=identity,
             logits=rng.normal(size=(count, 2)), ids=ids)
    np.savez(capsule / 'received.npz', ids=ids, iq=iq)
    dump(capsule / 'manifest.json', dict(protocol_schema='p2_min_v1',
        phase2_data_status='VALIDATED_ONCE', capsule_id='synthetic', split_count=1))
    split = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
        capsule_id='synthetic', split_id='one', registered_classes=classes,
        support_indices=list(range(support_count)), support_labels=np.repeat(np.arange(4),k).tolist(),
        query_indices=list(range(support_count,count)), k=k, support_seed=23, receiver='rx', scenario='scene')
    dump(capsule / 'splits/one.json', split)
    dump(row / 'd92_startup.json', dict(checkpoint_sha256=SHA, capsule=str(capsule),
        features=str(feature / 'received_features.npz'), query_fit_access=False, truth_read=False, seed=1))
    dump(feature / 'features_complete.json', dict(status='FROZEN_FEATURES_COMPLETE',
        capsule_id='synthetic', query_used_for_fitting=False))
    dump(feature / 'checkpoint_provenance.json', dict(target_access_before_freeze=False,
        checkpoint_inheritance=[], classes=list(CLASSES)))
    config = dict(algorithm=copy.deepcopy(mod.FROZEN_CONFIG), summary_already_deployed=False)
    return dict(row_root=row, capsule=capsule, output=tmp_path / 'output',
        expected_capsule_id='synthetic', expected_checkpoint_sha256=SHA, config=config), identity


def test_only_support_fit_and_actual_payload_bytes(tmp_path, monkeypatch):
    args, identity = fixture(tmp_path)
    fit = mod.fit_summary_joint
    calls = []
    def guarded_fit(**values):
        x = values['support_features'][:, :160]
        np.testing.assert_allclose(x / np.linalg.norm(x, axis=1, keepdims=True),
            identity[:4] / np.linalg.norm(identity[:4], axis=1, keepdims=True), atol=1e-7)
        assert values['support_ids'].tolist() == [f'opaque-{i}' for i in range(4)]
        assert values['old_classes'] == list(CLASSES)
        calls.append(len(x))
        return fit(**values)
    monkeypatch.setattr(mod, 'fit_summary_joint', guarded_fit)
    original_open = Path.open
    def no_truth(path, *a, **kw):
        assert 'truth' not in path.name and 'source_features' not in path.name
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', no_truth)
    mod.predict(**args)
    assert calls == [4]
    out = args['output']
    record = json.loads((out / 'predictions.jsonl').read_text())
    marker = json.loads((out / 'predictions_complete.json').read_text())
    assert record['mode'] == 'd92_sgjoint_registration'
    assert record['query_ids'] == [f'opaque-{i}' for i in range(4, 8)]
    assert marker['predictions'] == 1 and marker['truth_read'] is False
    assert marker['payload_audit']['source_samples_read'] is False
    ground = args['row_root'] / 'ground'
    expected_bytes = sum(p.stat().st_size for p in ground.iterdir())
    assert marker['payload_audit']['incremental_transfer_bytes'] == expected_bytes
    assert (out / 'compact.csv').is_file()
    assert json.loads((out / 'fit_trace.jsonl').read_text())['query_rows_used_for_fit'] == 0
    small=json.loads((out/'compact.jsonl').read_text())
    assert small['candidate_count']==small['fold_count']==0
    assert small['selected_objective'] is None and small['selected_old_nll'] is None
    assert small['learning_rate'] is None and small['gradient'] is None
    assert small['source_validation'] is None
    assert 'user prohibits' in small['source_validation_reason']
    startup=json.loads((out/'startup.json').read_text())
    assert startup['source_validation'] is None
    assert startup['source_validation_reason']==small['source_validation_reason']
    assert 'no learning rate/gradient' in small['unavailable_reason']
    with (out/'compact.csv').open(newline='',encoding='utf-8') as stream:
        csv_row=next(csv.DictReader(stream))
    assert csv_row['learning_rate']==csv_row['gradient']=='N/A'
    assert csv_row['selected_objective']=='N/A'
    assert csv_row['source_validation']=='N/A'
    with pytest.raises(FileExistsError):
        mod.predict(**args)


@pytest.mark.parametrize('fault', ['truth', 'overlap', 'checkpoint', 'ground_binding', 'source_path', 'config',
                                  'iq_layout','iq_nonfinite','physical_id','classes'])
def test_illegal_input_rejected_before_fit_or_output(tmp_path, monkeypatch, fault):
    args, _ = fixture(tmp_path)
    if fault in ('truth', 'overlap'):
        p = args['capsule'] / 'splits/one.json'
        v = json.loads(p.read_text())
        if fault == 'truth':
            v['query_labels'] = [0] * 4
        else:
            v['query_indices'][0] = 0
        dump(p, v)
    elif fault == 'checkpoint':
        args['expected_checkpoint_sha256'] = 'c' * 64
    elif fault in ('ground_binding', 'source_path'):
        p = args['row_root'] / 'ground/manifest.json'
        v = json.loads(p.read_text())
        if fault == 'ground_binding':
            v['checkpoint_sha256'] = 'c' * 64
        else:
            v['source_path'] = '/forbidden/source.npz'
        dump(p, v)
    elif fault == 'config':
        args['config']['algorithm']['unregistered'] = True
    elif fault=='classes':
        p=args['row_root']/'received_features/checkpoint_provenance.json'
        v=json.loads(p.read_text());v['classes']=list(reversed(CLASSES));dump(p,v)
    else:
        p = args['capsule'] / 'received.npz'
        with np.load(p) as d:
            ids, iq = d['ids'], d['iq']
        if fault=='iq_layout':iq=iq.transpose(0,2,1)
        elif fault=='iq_nonfinite':iq[0,0,0]=np.nan
        else:ids[0]='different'
        np.savez(p, ids=ids, iq=iq)
    def forbidden(**_):
        raise AssertionError('No fit allowed for invalid input')
    monkeypatch.setattr(mod, 'fit_summary_joint', forbidden)
    with pytest.raises(ValueError):
        mod.predict(**args)
    assert not args['output'].exists()


def test_stable_ties_follow_physical_class_id():
    scores = np.ones((3, 4))
    classes = ['z', 'a', 'b', 'c']
    assert mod.stable_predictions(scores, classes).tolist() == [1, 1, 1]
    order = [2, 0, 3, 1]
    assert mod.stable_predictions(scores[:, order], [classes[i] for i in order]).tolist() == [3, 3, 3]


def test_registered_features_and_query_scores_are_single_sample_invariant(tmp_path,monkeypatch):
    args,_=fixture(tmp_path)
    with np.load(args['row_root']/'received_features/received_features.npz') as d:
        identity=d['identity160'];logits=d['logits'];ids=d['ids']
    with np.load(args['capsule']/'received.npz') as d:iq=d['iq']
    features=mod.registered_features(iq,identity)
    individual=np.concatenate([mod.registered_features(iq[i:i+1],identity[i:i+1]) for i in range(len(ids))])
    np.testing.assert_array_equal(features,individual)
    states=[]
    original_fit=mod.fit_summary_joint
    def capture(**values):
        np.testing.assert_array_equal(values['support_features'],features[:4])
        state=original_fit(**values);states.append(state);return state
    monkeypatch.setattr(mod,'fit_summary_joint',capture)
    mod.predict(**args)
    first=json.loads((args['output']/'predictions.jsonl').read_text())
    state=states[0]
    batch=state.score(features[4:])
    singleton=np.concatenate([state.score(row[None]) for row in features[4:]])
    np.testing.assert_array_equal(batch,singleton)
    np.testing.assert_array_equal(batch,state.score(features[[7,5,4,6]])[[2,1,3,0]])
    np.testing.assert_array_equal(batch,first['scores'])
    # Change one query and append another while keeping support exactly fixed.
    identity[-1]*=-9;iq[-1]+=7
    identity=np.concatenate([identity,identity[-1:]*2]);iq=np.concatenate([iq,iq[-1:]*3])
    ids=np.append(ids,'opaque-new-query');logits=np.concatenate([logits,logits[-1:]])
    np.savez(args['row_root']/'received_features/received_features.npz',identity160=identity,logits=logits,ids=ids)
    np.savez(args['capsule']/'received.npz',iq=iq,ids=ids)
    path=args['capsule']/'splits/one.json';split=json.loads(path.read_text());split['query_indices'].append(8);dump(path,split)
    args['output']=tmp_path/'changed-query-output'
    mod.predict(**args)
    second=json.loads((args['output']/'predictions.jsonl').read_text())
    np.testing.assert_array_equal(first['scores'][:3],second['scores'][:3])
    np.testing.assert_array_equal(states[0].coefficient,states[1].coefficient)
    np.testing.assert_array_equal(states[0].intercept,states[1].intercept)
    assert states[0].audit_dict()['selected']==states[1].audit_dict()['selected']


def test_deployed_summary_reports_zero_incremental_transfer(tmp_path):
    args,_=fixture(tmp_path)
    args['config']['summary_already_deployed']=True
    mod.predict(**args)
    audit=json.loads((args['output']/'predictions_complete.json').read_text())['payload_audit']
    assert audit['total_file_bytes']>0 and audit['incremental_transfer_bytes']==0


def test_cross_validation_trace_contains_real_support_folds_and_measured_risk(tmp_path,capsys):
    from threadpoolctl import threadpool_limits
    args,_=fixture(tmp_path,k=2)
    with threadpool_limits(limits=2):
        mod.predict(**args)
    audit=json.loads((args['output']/'fit_trace.jsonl').read_text())
    small=json.loads((args['output']/'compact.jsonl').read_text())
    assert len(audit['candidate_trace'])==small['candidate_count']==16
    assert audit['fold_count']==small['fold_count']==2
    assert set(i for fold in audit['folds'] for i in fold['held_ids'])=={f'opaque-{i}' for i in range(8)}
    for candidate in audit['candidate_trace']:
        assert len(candidate['folds'])==2
        assert np.isfinite(candidate['objective']) and np.isfinite(candidate['macro_nll'])
        for fold in candidate['folds']:
            assert fold['train_per_class']==fold['held_per_class']==1
            assert np.isfinite(fold['held_before_temperature']['objective'])
            assert np.isfinite(fold['held_after_temperature']['objective'])
            assert fold['fit_elapsed_seconds']>=0
    assert small['selected_objective']==audit['selected_oof_risk']['objective']
    assert small['selected_old_nll']==audit['selected_oof_risk']['old_nll']
    assert small['selected_new_nll']==audit['selected_oof_risk']['new_nll']
    assert small['learning_rate'] is None and small['gradient'] is None
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sum(event['event']=='SUPPORT_CV_CANDIDATE' for event in events)==16
    assert audit['source_runtime_access'] is False and audit['query_rows_used_for_fit']==0
