"""Full synthetic six-matrix tests; no historical or real query results."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import summarize_d92_branch_local_ridge_benchmark as tool
from summarize_d92_confirmation import matrix_definition
from test_summarize_d92_repeated_benchmark import cohorts


def inputs():
    specs, scored = cohorts()
    for i, (spec, data) in enumerate(zip(specs, scored)):
        spec['run_id'] = f'local-{i}'
        spec['execution'] = dict(remote_run_root=f'/runs/local-{i}')
        spec['confirmation'].update(candidate_method=tool.CANDIDATE, candidate_folder='branch_local_ridge',
            candidate_predictor='evaluate_d92_branch_local_ridge.py', candidate_mode='d92_branch_local_ridge_registration',
            capsule=f'/capsule-{i}', branch_ridge_reference_run_id=f'branch-{i}',
            branch_ridge_reference_root=f'/runs/branch-{i}', branch_interaction_reference_run_id=f'interaction-{i}',
            branch_interaction_reference_root=f'/runs/interaction-{i}')
        spec['data']['capsule_id'] = spec['confirmation']['reuse_validated_capsule_id']
        spec['joint_benchmark'] = dict(old_accuracy_scope=tool.OLD_SCOPE)
        for row in spec['rows']:
            seed = row['seeds']['model']
            row.update(row_id=f'model-{seed}', source_root=f'/source/{seed}', reuse_row_root=f'/original-{i}/{seed}',
                       expected_checkpoint_sha256=f'{seed:064x}')
        for row in data['results']:
            if row['method'] == 'D92-SGJoint-v1':
                row['method'] = tool.CANDIDATE
            row.update(row_id=f'model-{row["model_seed"]}', split_id=':'.join(str(row[k]) for k in tool.KEY[1:]),
                class_count=6+row['new_count'], classes=[f'class-{n}' for n in range(6+row['new_count'])],
                query_count=30*(6+row['new_count']))
    branches, branch_scores = deepcopy(specs), deepcopy(scored)
    interactions, interaction_scores = deepcopy(specs), deepcopy(scored)
    for group, results, method, folder, factor in (
        (branches, branch_scores, tool.BRANCH, 'branch_ridge', .8),
        (interactions, interaction_scores, tool.INTERACTION, 'branch_interaction', .9)):
        for i, (spec, data) in enumerate(zip(group, results)):
            run_id = ('branch' if method == tool.BRANCH else 'interaction')+f'-{i}'
            spec['run_id'] = run_id
            spec['execution']['remote_run_root'] = '/runs/'+run_id
            spec['confirmation'].update(candidate_method=method, candidate_folder=folder,
                candidate_predictor=f'evaluate_d92_{folder}.py', candidate_mode=f'd92_{folder}_registration')
            for row in data['results']:
                if row['method'] == tool.CANDIDATE:
                    row['method'] = method
                    for field in tool.METRICS:
                        if row[field] is not None:
                            row[field] *= factor
    return specs, scored, branches, branch_scores, interactions, interaction_scores


def test_complete_triple_baseline_preserves_inputs_true_labels_and_equal_cells(tmp_path):
    args = inputs()
    before = deepcopy(args)
    result = tool.analyze(*args)
    assert args == before
    assert result['status'] == 'VERIFIED_TRIPLE_BASELINE_ANALYSIS'
    assert not result['automatic_promotion'] and result['selected_candidate'] is None
    assert len(result['bindings']) == 2
    for audit in result['baseline_record_audits']:
        assert set(audit) == {'local_vs_branch', 'local_vs_interaction', 'interaction_vs_branch'}
    assert result['baseline_record_audits'][0]['local_vs_branch']['unchanged_d92_records'] == 3600
    assert result['baseline_record_audits'][1]['interaction_vs_branch']['unchanged_dg_records'] == 12
    for name, baseline, expected in [('vs_d92', 'D92', .5),
            ('vs_branch_ridge', tool.BRANCH, .52), ('vs_branch_interaction', tool.INTERACTION, .585)]:
        combined = result['comparisons'][name]['combined']
        assert combined['methods'] == [baseline, tool.CANDIDATE]
        assert combined['rows'] == 9648
        assert combined['acceptance']['require_new_improvement']
        assert combined['acceptance']['old_guard_scope'] == 'all_cells_including_old_only'
        assert all(r['cells'] == 960 for r in combined['tables']['per_k'])
        assert all(r['cells'] == 1200 for r in combined['tables']['per_k_all'])
        assert all(r['cells'] == 240 for r in combined['tables']['per_k_new'])
        assert len(combined['tables']['per_seed_k_delta']) == 16
        assert len(combined['tables']['per_rx_scene_k_delta']) == 48
        pair = {r['method']: r for r in combined['tables']['per_k'] if r['k'] == 1}
        assert pair[tool.CANDIDATE]['harmonic_mean'] == pytest.approx(.65)
        assert pair[baseline]['harmonic_mean'] == pytest.approx(expected)
        assert 'independent model replicates' not in combined['aggregation']
    tool.write(result, tmp_path/'report')
    report = (tmp_path/'report/report.md').read_text(encoding='utf-8')
    assert tool.BRANCH in report and tool.INTERACTION in report
    assert '全部 K×新增类数' in report and '候选绝对性能' in report
    assert '旧类单独任务' in report and '总 support=(6+Nnew)×K' in report
    assert '| 20 | 20 |' in report and '| 1 | 0 |' in report and 'N/A' in report
    assert (tmp_path/'report/vs_branch_interaction/combined_rx4/per_k_new_delta.csv').is_file()
    with pytest.raises(FileExistsError):
        tool.write(result, tmp_path/'report')


@pytest.mark.parametrize('reference', [2, 4])
@pytest.mark.parametrize('fault', ['d92_metric', 'dg_metadata', 'missing_reference', 'class_order',
    'query_count', 'capsule', 'checkpoint', 'seed_role', 'wrong_run', 'wrong_root', 'matrix', 'old_guard'])
def test_reject_any_reference_binding_or_record_change(reference, fault):
    args = inputs()
    specs, data = args[reference:reference+2]
    if fault == 'd92_metric':
        next(r for r in data[0]['results'] if r['method'] == 'D92')['old_accuracy'] += .001
    elif fault == 'dg_metadata':
        next(r for r in data[0]['results'] if r['method'] == 'frozen_dg')['unrecognized_metadata'] = 1
    elif fault == 'missing_reference':
        data[0]['results'].pop()
    elif fault in ('query_count', 'class_order'):
        row = next(r for r in data[0]['results'] if r['method'] not in ('D92', 'frozen_dg'))
        if fault == 'query_count':
            row['query_count'] += 1
        else:
            row['classes'].reverse()
    elif fault == 'capsule':
        specs[0]['confirmation']['capsule'] += '-wrong'
    elif fault == 'checkpoint':
        specs[0]['rows'][0]['expected_checkpoint_sha256'] = 'a'*64
    elif fault == 'seed_role':
        specs[0]['rows'][0]['seeds']['augmentation'] = 9
    elif fault == 'wrong_run':
        specs[0]['run_id'] = 'wrong'
    elif fault == 'wrong_root':
        specs[0]['execution']['remote_run_root'] = '/other/'+specs[0]['run_id']
    elif fault == 'matrix':
        specs[0]['data']['support_seeds'][0] += 1
    else:
        specs[0]['joint_benchmark']['old_accuracy_scope'] = 'joint only'
    with pytest.raises((ValueError, KeyError)):
        tool.analyze(*args)


def test_interaction_branch_preregistration_is_checked_even_if_local_refs_match():
    args = inputs()
    args[4][0]['confirmation']['branch_ridge_reference_root'] = '/wrong/branch-0'
    with pytest.raises(ValueError, match='Reference root'):
        tool.analyze(*args)


@pytest.mark.parametrize('fault', ['duplicate_run', 'missing_cohort', 'incomplete_k', 'overlap_rx',
                                  'old_only_h', 'nan_metric', 'spec_row_mismatch', 'unexposed_claim'])
def test_no_partial_or_malformed_six_matrix_analysis(fault):
    args = inputs()
    specs, data = args[:2]
    if fault == 'duplicate_run':
        specs[1]['run_id'] = specs[0]['run_id']
    elif fault == 'missing_cohort':
        data.pop()
    elif fault == 'incomplete_k':
        specs[0]['data']['k'] = [1, 5, 10]
    elif fault == 'overlap_rx':
        specs[1]['data']['target_receivers'] = [specs[0]['data']['target_receivers'][0]]
    elif fault == 'old_only_h':
        next(r for r in data[0]['results'] if r['method'] == tool.CANDIDATE and r['new_count'] == 0)['harmonic_mean'] = .5
    elif fault == 'nan_metric':
        next(r for r in data[0]['results'] if r['method'] == tool.CANDIDATE)['accuracy'] = float('nan')
    elif fault == 'unexposed_claim':
        specs[0]['permissions']['claim_scope'] = 'Fresh independent confirmation'
    else:
        # Exact shared D92 equality cannot conceal a wrong score->spec row binding.
        for group in (args[1], args[3], args[5]):
            group[0]['results'][0]['row_id'] = 'wrong'
    with pytest.raises((ValueError, KeyError)):
        tool.analyze(*args)


@pytest.mark.parametrize('baseline', [tool.BRANCH, tool.INTERACTION])
def test_alternate_baseline_is_explicit_analysis_only_and_strict_new(baseline):
    spec = inputs()[0][0]
    spec['baseline_method'] = baseline
    with pytest.raises(ValueError, match='analysis-only'):
        matrix_definition(spec)
    spec['analysis_only'] = True
    assert matrix_definition(spec)[0] == (baseline, tool.CANDIDATE)
    spec['baseline_method'] = 'D92-renamed'
    with pytest.raises(ValueError, match='analysis-only'):
        matrix_definition(spec)


def test_old_only_guard_cannot_hide_behind_joint_gains():
    args = inputs()
    for new, reference in zip(args[1], args[5]):
        base = {tuple(r[k] for k in tool.KEY): r for r in reference['results'] if r['method'] == tool.INTERACTION}
        for row in new['results']:
            if row['method'] == tool.CANDIDATE:
                row['old_accuracy'] = base[tuple(row[k] for k in tool.KEY)]['old_accuracy']+.001 if row['new_count'] else 0.
    result = tool.analyze(*args)
    comparisons = result['comparisons']['vs_branch_interaction']['combined']['comparisons']
    assert all(r['delta']['old_accuracy'] > 0 and not r['old_guard'] for r in comparisons)
    assert not result['comparisons']['vs_branch_interaction']['combined']['preregistered_guard_pass']


def write_inputs(tmp_path):
    specs, data, branches, branch_data, interactions, interaction_data = inputs()
    folders = [tmp_path/f'results-{i}' for i in range(6)]
    for folder, spec, scored in zip(folders, specs+branches+interactions, data+branch_data+interaction_data):
        folder.mkdir()
        values = dict(startup=dict(spec=spec, commit='runtime'),
            complete=dict(status='SCORED', commit='runtime', records=len(scored['results'])), scores=scored)
        for name, value in values.items():
            (folder/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
    return folders


@pytest.mark.parametrize('fault', ['running', 'binding', 'count', 'commit', 'partial_matrix', 'cross_binding'])
def test_six_terminal_and_binding_barrier_precedes_any_score_access(tmp_path, monkeypatch, fault):
    folders = write_inputs(tmp_path)
    complete = folders[-1]/'complete.json'
    startup = folders[-1]/'startup.json'
    if fault in ('running', 'count', 'commit'):
        value = tool.read(complete)
        if fault == 'running':
            value['status'] = 'RUNNING'
        elif fault == 'count':
            value['records'] -= 1
        else:
            value['commit'] = 'wrong'
        complete.write_text(json.dumps(value), encoding='utf-8')
    else:
        value = tool.read(startup)
        if fault == 'binding':
            value['spec']['run_id'] = 'wrong'
        elif fault == 'partial_matrix':
            value['spec']['data']['k'] = [1, 5, 10]
        else:
            value['spec']['confirmation']['branch_ridge_reference_root'] = '/wrong/branch-1'
        startup.write_text(json.dumps(value), encoding='utf-8')
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != 'scores.json', 'Score file accessed before full barrier'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises((ValueError, KeyError)):
        tool.prepare(folders[:2], folders[2:4], folders[4:])


def test_prepare_preserves_every_input_and_hashes_all_six_artifact_sets(tmp_path):
    folders = write_inputs(tmp_path)
    before = {str(p): p.read_bytes() for folder in folders for p in folder.iterdir()}
    result = tool.prepare(folders[:2], folders[2:4], folders[4:])
    assert len(result['input_sha256']) == 18
    assert before == {str(p): p.read_bytes() for folder in folders for p in folder.iterdir()}


def test_all_six_coverage_checks_precede_any_metric_aggregation(monkeypatch):
    args = inputs()
    args[-1][-1]['results'].pop()
    def no_aggregation(*args, **kwargs):
        raise AssertionError('Aggregated before all six coverage checks')
    monkeypatch.setattr(tool, 'combine', no_aggregation)
    monkeypatch.setattr(tool, 'summarize', no_aggregation)
    with pytest.raises(ValueError, match='coverage|matrix'):
        tool.analyze(*args)
