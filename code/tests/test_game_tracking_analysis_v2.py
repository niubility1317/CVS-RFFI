import json

import pytest

from scripts.analyze_core90_game import factorial_differences
from scripts.analyze_core90_conditional_robustness import analyze_pair


SCENES = ['clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']


def fixture_files(tmp_path):
    paths = []
    for model in ('a', 'b'):
        p = tmp_path / (model + '.jsonl')
        rows = [dict(sample_id=str(i), scene=s, prediction=(i % 2 if s == 'clean' else 0))
                for s in SCENES for i in range(4)]
        if model == 'b':
            rows[0]['prediction'] = 1
            rows[5]['prediction'] = 1
        p.write_text(''.join(json.dumps(r)+'\n' for r in reversed(rows)), encoding='utf-8')
        m = tmp_path / (model + '.manifest.json')
        m.write_text(json.dumps(dict(schema='core90_conditional_prediction_manifest_v1', complete=True,
            prediction_file_closed_before_scoring=True, truth_used_for_prediction=False,
            prediction_scope='all_registered_classes_argmax', samples_per_scene=4,
            prediction_count=16, num_classes=2, scenes=SCENES)), encoding='utf-8')
        paths.append((p, m))
    return paths


def test_conditional_shared_subset_and_opaque_join(tmp_path):
    paths = fixture_files(tmp_path)
    called = []
    def truth():
        called.append(True)
        return [dict(sample_id=str(i), truth=i % 2, rx_i=i // 2) for i in range(4)]
    out = analyze_pair(*paths[0], *paths[1], truth)
    assert called == [True]
    assert out['shared_clean_correct']['count'] == 3
    assert out['shared_clean_correct']['fraction'] == .75
    assert out['shared_clean_correct']['per_tx_counts'] == {'0': 1, '1': 2}
    assert out['scenes']['leo_clear_weak']['a_error_rate'] == pytest.approx(2/3)
    assert out['scenes']['leo_clear_weak']['b_error_rate'] == pytest.approx(1/3)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unclosed', 'wrong_count', 'wrong_ids', 'truth_leak'])
def test_truth_not_opened_before_complete_prediction_validation(tmp_path, mutation):
    paths = fixture_files(tmp_path)
    p, m = paths[1]
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    manifest = json.loads(m.read_text())
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows[-1] = rows[0]
    if mutation == 'unclosed': manifest['prediction_file_closed_before_scoring'] = False
    if mutation == 'wrong_count': manifest['prediction_count'] = 15
    if mutation == 'wrong_ids': rows[0]['sample_id'] = 'foreign'
    if mutation == 'truth_leak': rows[0]['truth'] = 1
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    m.write_text(json.dumps(manifest), encoding='utf-8')
    def forbidden():
        pytest.fail('truth was touched before closure')
    with pytest.raises(ValueError): analyze_pair(*paths[0], *paths[1], forbidden)


def test_factorial_is_per_seed_and_does_not_mix_missing_rows():
    runs = [dict(seed=seed, row='V2_'+row, source_performance={'scenes': {'clean': {'accuracy': val}}})
            for seed in (1, 2) for row, val in zip('ABCDEF', (.5, .6, .5, .8, .6, .85))]
    result = factorial_differences(runs)
    assert len(result['per_seed']) == 2
    assert result['per_seed'][0]['scenes']['clean']['accuracy']['b8_interaction'] == pytest.approx(.2)
    assert result['per_seed'][0]['scenes']['clean']['accuracy']['eg_interaction'] == pytest.approx(.15)
    assert factorial_differences(runs[:-1])['per_seed'][1]['status'] == 'pending_rows'


def test_native_target_manifest_requires_every_declared_model_closed(tmp_path):
    paths = fixture_files(tmp_path)
    for name, (path, _) in zip(('A','B'), paths):
        (tmp_path/(name+'_predictions.jsonl')).write_bytes(path.read_bytes())
    m = tmp_path/'frozen_manifest.json'
    m.write_text(json.dumps(dict(schema='core90_target_frozen_15_v1', rows=['A','B','C'],
        samples_per_scene=4, scenes=SCENES, num_classes=2, source_only=False,
        fitting=False, selection=False, all_rows_predicted_before_truth=True)), encoding='utf-8')
    (tmp_path/'predictions_complete.json').write_text(json.dumps(dict(complete=True, rows=['A','B','C'],
        counts={s:4 for s in SCENES}, truth_accessed_for_scoring=False)), encoding='utf-8')
    def forbidden(): pytest.fail('truth touched with missing third model')
    with pytest.raises(ValueError, match='declared'):
        analyze_pair(tmp_path/'A_predictions.jsonl', m, tmp_path/'B_predictions.jsonl', m, forbidden)


def test_dense_source_metrics_include_worst_rx_tx_and_histogram(tmp_path):
    from scripts.analyze_core90_conditional_robustness import closed_source_metrics
    paths = fixture_files(tmp_path)
    path, m = paths[0]
    m.write_text(json.dumps(dict(schema='core90_game_source_predictions_v1', complete=True, source_only=True,
        counts={s:4 for s in SCENES}, num_classes=2, prediction_scope='all_registered_classes_argmax',
        truth_used_for_prediction=False, prediction_file_closed_before_scoring=True)), encoding='utf-8')
    truth = tmp_path/'truth.jsonl'
    truth.write_text(''.join(json.dumps(dict(sample_id=str(i),truth=i%2,rx_i=i//2))+'\n' for i in range(4)), encoding='utf-8')
    result = closed_source_metrics(path, m, truth)
    assert result['leo_clear_weak']['worst_rx_tx_accuracy'] == 0
    assert result['leo_clear_weak']['confusion_matrix'] == [[2,0],[2,0]]
    assert result['leo_clear_weak']['prediction_histogram'] == [4,0]


def test_failure_diagnostics_do_not_invent_collapse_from_loss():
    from scripts.analyze_core90_game import failure_diagnostics
    result = failure_diagnostics([dict(epoch=1, step=0, loss=100.)], stream_complete=True)
    assert result['first_collapse']['status'] == 'UNKNOWN'
    assert result['optimistic']['gradient_predictive_value']['status'] == 'UNKNOWN'
    rows = [dict(step=0,epoch=79,prediction_histogram=[5,5],optimistic_history_used=False),
            dict(step=1,epoch=80,prediction_histogram=[10,0],optimistic_history_used=True,
                 raw_optimistic_cosine=-.2,history_next_gradient_cosine=.8)]
    result = failure_diagnostics(rows, stream_complete=True)
    assert result['first_collapse']['first_observed_step'] == 1
    assert result['optimistic']['history_used_observations'] == 1
    assert result['optimistic']['gradient_predictive_value']['values'] == [.8]
