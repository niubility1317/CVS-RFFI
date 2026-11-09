import copy
import csv
import json

import pytest

from tools import analyze_ground_restoration as subject


def matrix(n, value):
    result = [[0] * 6 for _ in range(6)]
    result[0][0], result[0][1] = value, n - value
    return result


def fixture_scores():
    results, paired = [], []
    for task, count, raw in [('cross_rx', 126000, 111019), ('cross_both', 42000, 34366)]:
        for view in subject.VIEWS:
            for stream in subject.STREAMS:
                damage = 2520 if stream == 'candidate' and view != 'clean' else 0
                row = dict(test=task, readout=stream, view=view, dimension='overall', stratum='ALL', count=count)
                results.append(dict(row, confusion=matrix(count, raw - damage)))
                if stream != 'raw':
                    paired.append(dict(row, repairs=0, damage=damage, net=-damage))
    return dict(status='SCORED', truth_last=True, target_feedback_forbidden=True,
                results=results, paired_raw=paired)


def find(rows, task, stream, view):
    return next(row for row in rows if (row['test'], row['readout'], row['view']) == (task, stream, view))


def test_threshold_uses_exact_counts_for_all_six_scenes():
    scores = fixture_scores()
    result = subject.summarize(scores)
    assert result['raw_clean_correct'] == 145385 and result['minimum_correct'] == 140345
    assert result['goal_met'] and all(q['target_met'] for q in result['scenes'])
    assert all(q['candidate_correct'] == 140345 for q in result['scenes'])
    low = find(scores['results'], 'cross_rx', 'candidate', 'practical_low_urban')
    low['confusion'] = matrix(low['count'], 108498)
    pair = find(scores['paired_raw'], 'cross_rx', 'candidate', 'practical_low_urban')
    pair.update(damage=2521, net=-2521)
    result = subject.summarize(scores)
    assert not result['goal_met'] and not result['scenes'][-1]['target_met']
    assert all(q['target_met'] for q in result['scenes'][:-1])


def test_clean_is_guarded_separately_for_both_populations():
    scores = fixture_scores()
    for task, correct, repairs, damage in [('cross_rx', 111018, 0, 1), ('cross_both', 34367, 1, 0)]:
        row = find(scores['results'], task, 'candidate', 'clean')
        row['confusion'] = matrix(row['count'], correct)
        pair = find(scores['paired_raw'], task, 'candidate', 'clean')
        pair.update(repairs=repairs, damage=damage, net=repairs-damage)
    result = subject.summarize(scores)
    assert not result['goal_met']
    assert result['clean_preserved'] == {'cross_rx': False, 'cross_both': True}


def test_missing_prediction_group_or_mismatched_pair_is_rejected():
    scores = fixture_scores()
    scores['results'].pop()
    with pytest.raises(ValueError, match='groups'):
        subject.summarize(scores)
    scores = fixture_scores()
    scores['paired_raw'][0]['net'] = 1
    with pytest.raises(ValueError, match='paired'):
        subject.summarize(scores)
    scores = fixture_scores()
    scores['truth_last'] = False
    with pytest.raises(ValueError, match='truth-last'):
        subject.summarize(scores)


def test_report_writes_group_counts_and_preserves_existing_output(tmp_path):
    result = subject.summarize(fixture_scores())
    output = tmp_path / 'analysis'
    subject.write_report(output, result)
    assert json.loads((output/'summary.json').read_text(encoding='utf-8'))['goal_met']
    with (output/'combined_group_metrics.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7 and rows[-1]['count'] == '168000'
    assert 'cross_rx' in (output/'analysis.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        subject.write_report(output, result)


def test_load_requires_completed_original_full_run(tmp_path):
    run = tmp_path/'run'; arm = 'TF_Candidate'; row = run/'rows'/arm
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    commit = 'commit'; identity = dict(source_code_commit=subject.F0_COMMIT, model_seed=2026092701, parameters=232993,
                    model='original_F0_E50', epoch=50, updates=11100,
                    source_run='20261027-phase1-pamf-repl-shuffled-views-m16-r02',
                    source_row='main_split-repl-shuffled_time-3view-s2026092701')
    save(row/'training/complete.json', dict(status='SOURCE_TRAINED', epoch=30, updates=7410, TX_labels_read=0,
         V_reads=0, OE_reads=0, target_training_reads=0, code_commit=commit, F0_identity=identity))
    tests = {key: dict(count=value, views=list(subject.VIEWS), readouts=list(subject.STREAMS))
             for key, value in subject.TASKS.items()}
    save(row/'prediction/complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', run_id='run', row_id=arm,
         truth_read=False, query_fit=False, views=list(subject.VIEWS), readouts=list(subject.STREAMS), tests=tests, code_commit=commit))
    score_done = dict(status='SCORED_COMPLETE', run_id='run', row_id=arm, tests=list(subject.TASKS),
                      views=list(subject.VIEWS), readouts=list(subject.STREAMS))
    save(row/'score/complete.json', score_done)
    scores = fixture_scores(); scores.update(run_id='run', row_id=arm, model_seed=2026092701)
    save(row/'score/scores.json', scores)
    save(row/'results.json', dict(status='ANALYZED', run_id='run', arm=arm, code_commit=commit, truth_last=True))
    save(run/'complete.json', dict(status='ARTIFACTS_COMPLETE', all_rows_trained30_and_main_seven_scored=True))
    assert subject.summarize(subject.load_completed(run, arm))['goal_met']
    score_done['status'] = 'PARTIAL'
    save(row/'score/complete.json', score_done)
    with pytest.raises(ValueError, match='Completed'):
        subject.load_completed(run, arm)
