"""Synthetic end-to-end input, fit, query independence and audit boundaries."""
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import evaluate_d92_branch_interaction as mod
import export_d92_branch_features as exporter
from test_export_d92_branch_features import full_fixture
from test_export_d92_branch_support_features import dump


@pytest.fixture
def ready(full_fixture):
    export_args, iq, ids, infer, support, query = full_fixture
    exporter.export(**export_args)
    args = {k:export_args[k] for k in ('row_root', 'capsule', 'config', 'expected_capsule_id', 'expected_checkpoint_sha256')}
    args.update(output=export_args['output'].parent/'prediction', branch_features=export_args['output'])
    args['config'] = dict(algorithm=deepcopy(mod.local_core().FROZEN_CONFIG))
    return args, ids, support, query


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_real_core_only_fits_each_support_and_query_is_batch_invariant(ready, monkeypatch):
    args, ids, support, query = ready
    core = mod.local_core(); actual = core.fit_branch_interaction; observed = []; score_counts = []
    cache_args = {k:args[k] for k in ('branch_features','capsule','row_root','expected_capsule_id','expected_checkpoint_sha256')}
    arrays, *_ = exporter.load_features(**cache_args,config=dict(algorithm=exporter.local_core().FROZEN_CONFIG))
    split_by_ids = {tuple(ids[s['support_indices']]):s for s in
                    [mod.read(p) for p in (args['capsule']/'splits').glob('*.json')]}
    def fit(**kw):
        split = split_by_ids[tuple(kw['support_ids'])]
        assert set(kw['support_ids']).isdisjoint(ids[query])
        for key,value in arrays.items(): np.testing.assert_array_equal(kw[key], value[split['support_indices']])
        state = actual(**kw); observed.append(state)
        q = {key:value[query] for key,value in arrays.items()}
        scores = state.score(**q)
        singles = np.concatenate([state.score(**{k:v[i:i+1] for k,v in q.items()}) for i in range(len(query))])
        np.testing.assert_array_equal(scores, singles)
        np.testing.assert_array_equal(scores[::-1], state.score(**{k:v[::-1] for k,v in q.items()}))
        def individual_score(**features):
            score_counts.append(len(features['z_id']))
            assert len(features['z_id']) == 1
            return state.score(**features)
        return SimpleNamespace(classes=state.classes,audit_dict=state.audit_dict,score=individual_score)
    monkeypatch.setattr(core, 'fit_branch_interaction', fit)
    original = np.lib.npyio.NpzFile.__getitem__
    def guard(archive,key):
        assert key != 'iq'
        return original(archive,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', guard)
    marker = mod.predict(**args)
    assert len(observed) == marker['factorization_count'] == marker['split_count'] == 2
    assert score_counts and set(score_counts) == {1}
    assert marker['status'] == 'PREDICTIONS_COMPLETE'
    assert marker['query_used_for_fitting'] is marker['truth_read'] is False
    traces = lines(args['output']/'fit_trace.jsonl'); compact = lines(args['output']/'compact.jsonl')
    for trace,small in zip(traces,compact):
        assert trace['folds'] == [] and trace['oof'] is None and trace['factorization_count'] == 1
        n = trace['support_count']; c = len(trace['classes'])
        assert trace['head_bytes'] == trace['persistent_state_bytes'] == 8*(n*(736+c+2)+c)+16
        assert trace['selected'] == 'interaction'
        assert [v['physical_id'] for v in trace['support_records']] == trace['final_fit']['training_physical_ids']
        assert trace['heldout_unavailable_reason'] == ('K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if trace['k']==1 else 'NO_CV_FIXED_CONFIG')
        assert small['gradient_norm'] == trace['final_fit']['gradient_norm'] >= 0
        assert small['fit_call_seconds'] >= trace['fit_seconds']
        assert small['query_score_seconds'] >= 0 and small['prediction_write_seconds'] >= 0
    startup = mod.read(args['output']/'startup.json')
    assert startup['payload_audit']['model_already_deployed'] is None
    assert startup['payload_audit']['native_total_physical_forward_count'] == 0
    assert startup['payload_audit']['existing_native_physical_forward_count'] == len(ids)
    assert startup['payload_audit']['feature_cache_reused'] is True
    assert all(record['mode'] == mod.MODE for record in lines(args['output']/'predictions.jsonl'))
    # Independent scorer validation reads only these synthetic predictions.
    import score_d92_confirmation as scorer
    splits = {p.stem:mod.read(p) for p in (args['capsule']/'splits').glob('*.json')}
    scorer.validate_method_records(args['output'],'D92-BranchInteraction-v1',mod.read(args['capsule']/'manifest.json'),ids,splits,
        dict(candidate_method='D92-BranchInteraction-v1', candidate_folder='branch_interaction',
             candidate_predictor='evaluate_d92_branch_interaction.py', candidate_mode=mod.MODE))


@pytest.mark.parametrize('fault',['query_labels','overlap','missing','config'])
def test_invalid_inputs_rejected_before_fit(ready, monkeypatch, fault):
    args, *_ = ready
    monkeypatch.setattr(mod.local_core(),'fit_branch_interaction',lambda **kw:pytest.fail('Invalid input must not fit'))
    path=args['capsule']/'splits/small.json'; split=mod.read(path)
    if fault=='query_labels':split['query_labels']=[0,1,2]
    if fault=='overlap':split['query_indices'][0]=split['support_indices'][0]
    if fault=='missing':path.unlink()
    elif fault=='config':args['config']['algorithm']['ridge_coefficient']=2
    else:dump(path,split)
    with pytest.raises(ValueError):mod.predict(**args)
    assert not args['output'].exists()


def test_fit_failure_has_no_completion_and_preserves_output(ready, monkeypatch):
    args, *_=ready
    def failed(**kw):raise RuntimeError('synthetic solver failure')
    monkeypatch.setattr(mod.local_core(),'fit_branch_interaction',failed)
    with pytest.raises(RuntimeError,match='solver'):mod.predict(**args)
    assert not (args['output']/'predictions_complete.json').exists()
    assert mod.read(args['output']/'technical_failure.json')['status']=='TECHNICAL_FAILURE'
    with pytest.raises(FileExistsError):mod.predict(**args)


def test_ties_are_physical_id_stable():
    assert mod.stable_predictions(np.zeros((2,3)),['z','a','m']).tolist()==[1,1]
