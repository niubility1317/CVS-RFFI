"""Synthetic complete-matrix checks; no target results are loaded."""
from itertools import product
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_d92_confirmation import main, summarize


SEEDS = range(2026092701, 2026092705)
RECEIVERS = ['19-1', '8-14', '8-7']
SCENES = ['practical_high', 'practical_mid', 'practical_low_urban']
SUPPORT_SEEDS = range(2026092711, 2026092716)


def complete_scores(values):
    """Populate all 3,600 cells per method and the 36 frozen-DG records."""
    rows = []
    for method, seed, rx, scene, k, new_count, support_seed in product(
            ['D92', 'D92-SCV-v1'], SEEDS, RECEIVERS, SCENES,
            [1, 5, 10, 20], [0, 2, 5, 10, 20], SUPPORT_SEEDS):
        old, new = values(method, seed)
        # Old-only cells intentionally differ so their accidental inclusion is visible.
        old = old if new_count else 0.99
        new = new if new_count else None
        harmonic = 2 * old * new / (old + new) if new is not None else None
        rows.append(dict(method=method, model_seed=seed, receiver=rx, scenario=scene,
            k=k, new_count=new_count, support_seed=support_seed,
            accuracy=(6 * old + new_count * new) / (6 + new_count) if new_count else old,
            old_accuracy=old, new_accuracy=new, harmonic_mean=harmonic,
            macro_f1=(old + new) / 2 if new_count else old,
            old_macro_f1=old, new_macro_f1=new, old_floor=old, new_floor=new,
            forgetting=0.99 - old if new_count else None))
    for seed, rx, scene in product(SEEDS, RECEIVERS, SCENES):
        rows.append(dict(method='frozen_dg', model_seed=seed, receiver=rx, scenario=scene,
            k=1, new_count=0, support_seed=SUPPORT_SEEDS.start,
            accuracy=0.8, old_accuracy=0.8, new_accuracy=None, harmonic_mean=None,
            macro_f1=0.8, old_macro_f1=0.8, new_macro_f1=None,
            old_floor=0.8, new_floor=None, forgetting=None))
    return dict(status='SCORED', results=rows, selection_feedback_forbidden=True,
                claim_scope='Synthetic regression data; no real target observations')


@pytest.mark.parametrize('missing_method', ['D92', 'D92-SCV-v1', 'frozen_dg'])
def test_complete_coverage_rejects_missing_rows(missing_method):
    data = complete_scores(lambda method, seed: (0.5, 0.5))
    assert len(data['results']) == 7236
    index = next(i for i, row in enumerate(data['results']) if row['method'] == missing_method)
    data['results'].pop(index)
    with pytest.raises(ValueError, match='Incomplete paired matrix|Wrong final coverage'):
        summarize(data)


def test_h_is_mean_of_cell_h_and_excludes_old_only_rows():
    def values(method, seed):
        if method == 'D92-SCV-v1':
            return 0.6, 0.6
        return (0.9, 0.1) if seed % 2 else (0.1, 0.9)
    result = summarize(complete_scores(values))
    assert result['rows'] == 7236
    for k in [1, 5, 10, 20]:
        pair = {r['method']: r for r in result['tables']['per_k'] if r['k'] == k}
        baseline, candidate = pair['D92'], pair['D92-SCV-v1']
        assert baseline['cells'] == candidate['cells'] == 720
        assert baseline['old_accuracy'] == pytest.approx(0.5)
        assert baseline['new_accuracy'] == pytest.approx(0.5)
        assert baseline['harmonic_mean'] == pytest.approx(0.18)
        assert baseline['harmonic_mean'] != pytest.approx(
            2 * baseline['old_accuracy'] * baseline['new_accuracy'] /
            (baseline['old_accuracy'] + baseline['new_accuracy']))
        assert candidate['harmonic_mean'] == pytest.approx(0.6)
        comparison = next(r for r in result['comparisons'] if r['k'] == k)
        assert comparison['delta']['harmonic_mean'] == pytest.approx(0.42)
    assert result['preregistered_guard_pass'] is True


@pytest.mark.parametrize('baseline,candidate,failing_guard', [
    ((0.2, 0.8), (0.5, 0.75), 'new_guard'),
    ((0.8, 0.2), (0.75, 0.5), 'old_guard'),
])
def test_h_gain_cannot_hide_either_side_regression(baseline, candidate, failing_guard):
    result = summarize(complete_scores(lambda method, seed: baseline if method == 'D92' else candidate))
    for comparison in result['comparisons']:
        assert comparison['h_improved'] is True
        assert comparison[failing_guard] is False
        other = 'new_guard' if failing_guard == 'old_guard' else 'old_guard'
        assert comparison[other] is True
        assert comparison['delta']['harmonic_mean'] == pytest.approx(0.28)
    assert result['preregistered_guard_pass'] is False
    assert result['new_improved_every_k'] is (failing_guard == 'old_guard')


def sfhead_spec():
    return dict(rows=[dict(seeds=dict(model=s)) for s in SEEDS],
        data=dict(target_receivers=['20-19'],scenarios=SCENES,k=[1,5,10,20],
            new_class_counts=[0,2,5,10,20],support_seeds=list(SUPPORT_SEEDS)),
        confirmation=dict(candidate_method='D92-SFHead-v1',candidate_folder='sfhead',
            candidate_predictor='predict_d92_sourcefree_head.py',candidate_mode='d92_sfhead_registration',
            expected_split_count=300,splits_per_model=300,model_rows=4,predictions_total=2412),
        metrics_plan=dict(acceptance=dict(old_max_drop=0.01,require_new_improvement=True)),
        permissions=dict(claim_scope='Synthetic one-receiver confirmation only'))


def sfhead_scores():
    data=complete_scores(lambda method,seed:(0.5,0.5) if method=='D92' else (0.6,0.6))
    data['results']=[r for r in data['results'] if r['receiver']=='19-1']
    for row in data['results']:
        row['receiver']='20-19'
        if row['method']=='D92-SCV-v1':row['method']='D92-SFHead-v1'
    return data


def test_sfhead_spec_derives_exact_2412_matrix_and_real_method_labels(tmp_path,monkeypatch):
    data,spec=sfhead_scores(),sfhead_spec()
    result=summarize(data,spec)
    assert result['rows']==2412
    assert result['methods']==['D92','D92-SFHead-v1']
    assert result['matrix']['receiver']==['20-19']
    assert result['preregistered_guard_pass'] is True
    assert result['acceptance']['require_new_improvement'] is True
    assert len(result['tables']['per_k'])==8
    assert len(result['tables']['per_k_new'])==40
    assert len(result['tables']['per_seed_k'])==32
    assert len(result['tables']['per_rx_scene_k'])==24
    assert all(r['cells']==240 for r in result['tables']['per_k'])
    scores_path,spec_path,output=tmp_path/'synthetic_scores.json',tmp_path/'synthetic_spec.json',tmp_path/'report'
    scores_path.write_text(json.dumps(data),encoding='utf-8')
    spec_path.write_text(json.dumps(spec),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['summarize_d92_confirmation','--scores',str(scores_path),'--spec',str(spec_path),'--output',str(output)])
    main()
    report=(output/'report.md').read_text(encoding='utf-8')
    assert 'D92-SFHead-v1' in report and 'D92-SCV' not in report
    assert '1RX' in report and '2412' in report
    assert 'H和新类严格提升' in report


@pytest.mark.parametrize('fault',['missing','duplicate','model_seed','receiver','scenario','k','new_count','support_seed','dg_duplicate'])
def test_sfhead_exact_coverage_rejects_missing_or_substituted_cell_even_same_count(fault):
    data=sfhead_scores()
    candidate=[r for r in data['results'] if r['method']=='D92-SFHead-v1']
    if fault=='missing':
        data['results'].remove(candidate[0])
    elif fault=='duplicate':
        candidate[0].update(candidate[1])
    elif fault=='dg_duplicate':
        dg=[r for r in data['results'] if r['method']=='frozen_dg']
        dg[0].update(dg[1])
    else:
        candidate[0][fault]='wrong' if fault in ('receiver','scenario') else 999
    with pytest.raises(ValueError,match='Incomplete paired matrix|Wrong final coverage'):
        summarize(data,sfhead_spec())


@pytest.mark.parametrize('new_accuracy,passes',[(0.5,False),(0.495,False),(0.51,True)])
def test_sfhead_requires_strict_new_gain_at_every_k_even_when_h_and_guards_pass(new_accuracy,passes):
    data=sfhead_scores()
    for row in data['results']:
        if row['method']=='D92-SFHead-v1' and row['k']==20 and row['new_count']:
            row['old_accuracy']=0.65
            row['new_accuracy']=new_accuracy
            row['harmonic_mean']=2*0.65*new_accuracy/(0.65+new_accuracy)
    result=summarize(data,sfhead_spec())
    assert all(r['h_improved'] and r['old_guard'] and r['new_guard'] for r in result['comparisons'])
    assert result['preregistered_guard_pass'] is passes
    assert result['new_improved_every_k'] is passes


def test_spec_requires_explicit_scenarios_instead_of_guessing():
    spec=sfhead_spec()
    del spec['data']['scenarios']
    with pytest.raises(KeyError,match='scenarios'):
        summarize(sfhead_scores(),spec)


@pytest.mark.parametrize('new_gain',[True,False])
@pytest.mark.parametrize('method,folder,predictor,mode',[
    ('D92-SGJoint-v1','sgjoint','predict_d92_summary_joint.py','d92_sgjoint_registration'),
    ('D92-BranchRidge-v1','branch_ridge','evaluate_d92_branch_ridge.py','d92_branch_ridge_registration'),
])
def test_sgjoint_label_and_strict_new_gain_are_preserved(new_gain,method,folder,predictor,mode):
    data,spec=sfhead_scores(),sfhead_spec()
    spec['confirmation'].update(candidate_method=method,candidate_folder=folder,
        candidate_predictor=predictor,candidate_mode=mode)
    # No optional acceptance override: SGJoint must default to the strict rule.
    del spec['metrics_plan']['acceptance']
    for row in data['results']:
        if row['method']=='D92-SFHead-v1':
            row['method']=method
            if not new_gain and row['k']==20 and row['new_count']:
                row['new_accuracy']=0.5
                row['harmonic_mean']=2*row['old_accuracy']*0.5/(row['old_accuracy']+0.5)
    result=summarize(data,spec)
    assert result['methods']==['D92',method]
    assert all(r['method'] in result['methods'] for r in result['tables']['per_k'])
    assert result['acceptance']['require_new_improvement'] is True
    assert all(r['h_improved'] and r['old_guard'] and r['new_guard'] for r in result['comparisons'])
    assert result['preregistered_guard_pass'] is new_gain
    spec['metrics_plan']['acceptance']=dict(require_new_improvement=False)
    with pytest.raises(ValueError,match=method+'.*strict new-class'):
        summarize(data,spec)
