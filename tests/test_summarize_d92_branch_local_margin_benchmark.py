"""Complete synthetic repeat matrices only; no real score files or target data."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import summarize_d92_branch_local_margin_benchmark as tool
from summarize_d92_confirmation import matrix_definition
from test_summarize_d92_branch_local_ridge_benchmark import inputs as prior_inputs


def inputs():
    local_specs, local_scores, _, _, interaction_specs, interaction_scores = prior_inputs()
    specs, scores = deepcopy(local_specs), deepcopy(local_scores)
    for i, (spec, data) in enumerate(zip(specs, scores)):
        spec['run_id'] = f'margin-{i}'
        spec['execution']['remote_run_root'] = f'/runs/margin-{i}'
        spec['confirmation'].update(candidate_method=tool.CANDIDATE, candidate_folder='branch_local_margin',
            candidate_predictor='evaluate_d92_branch_local_margin.py', candidate_mode='d92_branch_local_margin_registration',
            branch_local_ridge_reference_run_id=f'local-{i}', branch_local_ridge_reference_root=f'/runs/local-{i}')
        for row in data['results']:
            if row['method'] == tool.LOCAL:
                row['method'] = tool.CANDIDATE
                for metric in tool.METRICS:
                    if row[metric] is not None:
                        row[metric] *= 1.04
    return specs, scores, local_specs, local_scores, interaction_specs, interaction_scores


def test_full_matrix_direct_baseline_descriptive_controls_and_complete_report(tmp_path):
    args = inputs(); before = deepcopy(args)
    result = tool.analyze(*args)
    assert args == before
    assert result['status'] == 'VERIFIED_TRIPLE_BASELINE_ANALYSIS'
    assert result['direct_baseline'] == tool.LOCAL
    assert result['descriptive_baselines'] == [tool.INTERACTION, 'D92']
    assert result['automatic_promotion'] is False and result['selected_candidate'] is None
    assert result['direct_baseline_every_k_screen']['k'] == [1, 5, 10, 20]
    assert result['direct_baseline_every_k_screen']['passed']
    for audit in result['baseline_record_audits']:
        assert set(audit) == {'margin_vs_local', 'margin_vs_interaction', 'local_vs_interaction'}
    assert result['baseline_record_audits'][0]['margin_vs_local']['unchanged_d92_records'] == 3600
    assert result['baseline_record_audits'][1]['local_vs_interaction']['unchanged_dg_records'] == 12
    for name, baseline, expected in [('vs_local_ridge', tool.LOCAL, .65),
            ('vs_branch_interaction', tool.INTERACTION, .585), ('vs_d92', 'D92', .5)]:
        combined = result['comparisons'][name]['combined']
        assert combined['methods'] == [baseline, tool.CANDIDATE]
        assert combined['advancement_gate'] == (baseline == tool.LOCAL)
        assert combined['rows'] == 9648 and combined['acceptance']['require_new_improvement']
        assert combined['acceptance']['old_guard_scope'] == 'all_cells_including_old_only'
        assert all(r['cells'] == 960 for r in combined['tables']['per_k'])
        assert all(r['cells'] == 1200 for r in combined['tables']['per_k_all'])
        assert all(r['cells'] == 240 for r in combined['tables']['per_k_new'])
        assert len(combined['tables']['per_seed_k_delta']) == 16
        assert len(combined['tables']['per_rx_scene_k_delta']) == 48
        pair = {r['method']: r for r in combined['tables']['per_k'] if r['k'] == 1}
        assert pair[tool.CANDIDATE]['harmonic_mean'] == pytest.approx(.676)
        assert pair[baseline]['harmonic_mean'] == pytest.approx(expected)
        assert combined['comparisons'][0]['delta']['harmonic_mean'] == pytest.approx(.676-expected)
        assert 'independent model replicates' not in combined['aggregation']
    tool.write(result, tmp_path/'report')
    report = (tmp_path/'report/report.md').read_text(encoding='utf-8')
    assert '直接基线' in report and '描述性对照' in report
    assert '| 20 | 20 |' in report and '| 1 | 0 |' in report and 'N/A' in report
    assert '旧类单独任务' in report and '总 support=(6+Nnew)×K' in report
    assert (tmp_path/'report/vs_local_ridge/combined_rx4/per_k_new_delta.csv').is_file()
    assert (tmp_path/'report/vs_branch_interaction/rx1/per_rx_scene_k_delta.csv').is_file()
    with pytest.raises(FileExistsError):
        tool.write(result, tmp_path/'report')


@pytest.mark.parametrize('reference', [2, 4])
@pytest.mark.parametrize('fault', ['d92_metric', 'dg_metadata', 'missing_reference', 'class_order',
    'query_count', 'split_id', 'capsule', 'checkpoint', 'seed_role', 'wrong_run', 'wrong_root', 'matrix', 'old_guard'])
def test_reject_reference_binding_or_record_changes(reference, fault):
    args = inputs(); specs, data = args[reference:reference+2]
    if fault == 'd92_metric':
        next(r for r in data[0]['results'] if r['method'] == 'D92')['old_accuracy'] += .001
    elif fault == 'dg_metadata':
        next(r for r in data[0]['results'] if r['method'] == 'frozen_dg')['unexpected_metadata'] = 'changed'
    elif fault == 'missing_reference':
        data[0]['results'].pop()
    elif fault in ('query_count', 'class_order', 'split_id'):
        row = next(r for r in data[0]['results'] if r['method'] not in ('D92', 'frozen_dg'))
        if fault == 'query_count':
            row['query_count'] += 1
        elif fault == 'split_id':
            row['split_id'] += '-different'
        else:
            row['classes'].reverse()
    elif fault == 'capsule':
        specs[0]['confirmation']['capsule'] += '-wrong'
    elif fault == 'checkpoint':
        specs[0]['rows'][0]['expected_checkpoint_sha256'] = 'a'*64
    elif fault == 'seed_role':
        specs[0]['rows'][0]['seeds']['augmentation'] = 17
    elif fault == 'wrong_run':
        specs[0]['run_id'] = 'wrong'
    elif fault == 'wrong_root':
        specs[0]['execution']['remote_run_root'] = '/wrong/'+specs[0]['run_id']
    elif fault == 'matrix':
        specs[0]['data']['support_seeds'][0] += 1
    else:
        specs[0]['joint_benchmark']['old_accuracy_scope'] = 'joint only'
    with pytest.raises((ValueError, KeyError)):
        tool.analyze(*args)


def test_local_interaction_prior_binding_is_required_even_when_candidate_bindings_match():
    args = inputs()
    args[2][0]['confirmation']['branch_interaction_reference_root'] = '/wrong/interaction-0'
    with pytest.raises(ValueError, match='Reference root'):
        tool.analyze(*args)


@pytest.mark.parametrize('fault', ['missing_cohort', 'duplicate_run', 'duplicate_score', 'overlap_rx', 'missing_k',
    'wrong_method', 'old_only_h', 'nan_metric', 'wrong_row', 'unexposed_claim', 'score_checkpoint', 'score_capsule'])
def test_incomplete_or_malformed_input_is_rejected(fault):
    args = inputs(); specs, data = args[:2]
    if fault == 'missing_cohort':
        data.pop()
    elif fault == 'duplicate_run':
        specs[1]['run_id'] = specs[0]['run_id']
    elif fault == 'duplicate_score':
        data[0]['results'][-1] = deepcopy(data[0]['results'][-2])
    elif fault == 'overlap_rx':
        specs[1]['data']['target_receivers'] = [specs[0]['data']['target_receivers'][0]]
    elif fault == 'missing_k':
        specs[0]['data']['k'] = [1, 5, 10]
    elif fault == 'wrong_method':
        specs[0]['confirmation']['candidate_method'] = tool.LOCAL
    elif fault == 'unexposed_claim':
        specs[0]['permissions']['claim_scope'] = 'Fresh independent confirmation'
    else:
        row = next(r for r in data[0]['results'] if r['method'] == tool.CANDIDATE and r['new_count'] == 0)
        if fault == 'old_only_h':
            row['harmonic_mean'] = .5
        elif fault == 'nan_metric':
            row['accuracy'] = float('nan')
        elif fault == 'wrong_row':
            row['row_id'] = 'different-model'
        elif fault == 'score_checkpoint':
            row['checkpoint_sha256'] = 'a'*64
        else:
            row['capsule_id'] = 'wrong'
    with pytest.raises((ValueError, KeyError)):
        tool.analyze(*args)


@pytest.mark.parametrize('baseline', [tool.LOCAL, tool.INTERACTION])
def test_alternate_baseline_requires_explicit_analysis_only(baseline):
    spec = inputs()[0][0]; spec['baseline_method'] = baseline
    with pytest.raises(ValueError, match='analysis-only'):
        matrix_definition(spec)
    spec['analysis_only'] = True
    assert matrix_definition(spec)[0] == (baseline, tool.CANDIDATE)


def change_candidate(args, mutation):
    for data, reference in zip(args[1], args[3]):
        baseline = {tuple(r[k] for k in tool.KEY): r for r in reference['results'] if r['method'] == tool.LOCAL}
        for row in data['results']:
            if row['method'] == tool.CANDIDATE:
                mutation(row, baseline[tuple(row[k] for k in tool.KEY)])


def test_old_only_decline_is_in_direct_guard_not_hidden_by_joint_gain():
    args = inputs()
    change_candidate(args, lambda row, base: row.update(old_accuracy=base['old_accuracy']+.001 if row['new_count'] else 0.))
    result = tool.analyze(*args)
    direct = result['comparisons']['vs_local_ridge']['combined']
    assert all(r['delta']['old_accuracy'] > 0 and not r['old_guard'] for r in direct['comparisons'])
    assert not result['direct_baseline_every_k_screen']['passed']


def test_tolerated_old_loss_passes_guard_but_is_not_strict_joint_gain():
    args = inputs()
    change_candidate(args, lambda row, base: row.update(old_accuracy=base['old_accuracy']-.001))
    result = tool.analyze(*args)
    assert result['direct_baseline_every_k_screen']['passed']
    assert not result['direct_baseline_every_k_screen']['strict_joint_improvement']


def test_every_k_requires_direct_new_gain_even_if_older_controls_win():
    args = inputs()
    def mutate(row, base):
        if row['k'] == 1 and row['new_count']:
            row['new_accuracy'] = base['new_accuracy']
    change_candidate(args, mutate)
    result = tool.analyze(*args)
    assert not result['direct_baseline_every_k_screen']['passed']
    assert result['comparisons']['vs_d92']['combined']['preregistered_guard_pass']
    assert result['comparisons']['vs_branch_interaction']['combined']['preregistered_guard_pass']


def test_negative_single_stratum_is_visible_without_extra_allstrata_gate():
    args = inputs()
    def mutate(row, base):
        if row['new_count']:
            for metric in ('new_accuracy', 'harmonic_mean'):
                row[metric] = base[metric]+(-.01 if row['new_count'] == 2 else .02)
    change_candidate(args, mutate)
    result = tool.analyze(*args)
    assert result['direct_baseline_every_k_screen']['passed']
    rows = result['comparisons']['vs_local_ridge']['combined']['tables']['per_k_new_delta']
    assert all(r['harmonic_mean'] < 0 for r in rows if r['new_count'] == 2)


def test_direct_delta_uses_matching_cells_not_summed_older_deltas():
    args = inputs()
    def mutate(row, base):
        if row['new_count']:
            row['new_accuracy'] = base['new_accuracy']+(.04 if row['receiver'] == '19-1' else -.005)
    change_candidate(args, mutate)
    result = tool.analyze(*args)
    direct = result['comparisons']['vs_local_ridge']['combined']
    assert all(r['delta']['new_accuracy'] == pytest.approx((.04-3*.005)/4) for r in direct['comparisons'])
    assert direct['paired_delta_rule'] == 'mean_of_candidate_minus_reference_on_identical_cell_keys'


def write_inputs(tmp_path):
    args = inputs(); folders = [tmp_path/f'results-{i}' for i in range(6)]
    for folder, spec, data in zip(folders, args[0]+args[2]+args[4], args[1]+args[3]+args[5]):
        folder.mkdir()
        for name, value in dict(startup=dict(spec=spec, commit='runtime'),
                complete=dict(status='SCORED', commit='runtime', records=len(data['results']), run_id=spec['run_id']), scores=data).items():
            (folder/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
    return folders


@pytest.mark.parametrize('fault', ['running', 'wrong_commit', 'wrong_count', 'partial_matrix', 'binding', 'cross_binding'])
def test_six_complete_bound_matrices_required_before_score_file_open(tmp_path, monkeypatch, fault):
    folders = write_inputs(tmp_path)
    terminal = folders[-1]/'complete.json'; startup = folders[2]/'startup.json'
    if fault in ('running', 'wrong_commit', 'wrong_count'):
        value = tool.read(terminal)
        if fault == 'running':
            value['status'] = 'RUNNING'
        elif fault == 'wrong_commit':
            value['commit'] = 'wrong'
        else:
            value['records'] -= 1
        terminal.write_text(json.dumps(value), encoding='utf-8')
    else:
        value = tool.read(startup)
        if fault == 'partial_matrix':
            value['spec']['data']['k'] = [1, 5, 10]
        elif fault == 'binding':
            value['spec']['execution']['remote_run_root'] = '/wrong/local-0'
        else:
            value['spec']['confirmation']['branch_interaction_reference_root'] = '/wrong/interaction-0'
        startup.write_text(json.dumps(value), encoding='utf-8')
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != 'scores.json', 'Score file accessed before full six-input barrier'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises((ValueError, KeyError)):
        tool.prepare(folders[:2], folders[2:4], folders[4:])


@pytest.mark.parametrize('fault', ['last_coverage', 'last_identity', 'paired_split'])
def test_all_identities_checked_before_metric_value_access(monkeypatch, fault):
    args = inputs()
    class GuardedRow(dict):
        def __getitem__(self, key):
            assert key not in tool.METRICS, 'Metric accessed before all six identity checks'
            return super().__getitem__(key)
    for group in (args[1], args[3], args[5]):
        for data in group:
            data['results'] = [GuardedRow(row) for row in data['results']]
    if fault == 'last_coverage':
        args[5][-1]['results'].pop()
    elif fault == 'last_identity':
        args[5][-1]['results'][-1]['row_id'] = 'wrong'
    else:
        next(r for r in args[5][-1]['results'] if r['method'] == tool.INTERACTION)['split_id'] = 'wrong'
    with pytest.raises((ValueError, KeyError)):
        tool.analyze(*args)


def test_prepare_hashes_six_artifact_sets_without_mutating_inputs(tmp_path):
    folders = write_inputs(tmp_path)
    before = {str(p): p.read_bytes() for folder in folders for p in folder.iterdir()}
    result = tool.prepare(folders[:2], folders[2:4], folders[4:])
    assert len(result['input_sha256']) == 18
    assert before == {str(p): p.read_bytes() for folder in folders for p in folder.iterdir()}


def test_all_numeric_validation_precedes_aggregation(monkeypatch):
    args = inputs()
    args[-1][-1]['results'][-1]['accuracy'] = float('nan')
    # Make shared original records equal so the specific finite-value check is reached.
    for group in (args[1], args[3]):
        group[-1]['results'][-1]['accuracy'] = args[-1][-1]['results'][-1]['accuracy']
    def forbidden(*args, **kwargs):
        raise AssertionError('Aggregation before numeric validation')
    monkeypatch.setattr(tool, 'combine', forbidden)
    monkeypatch.setattr(tool, 'summarize', forbidden)
    with pytest.raises(ValueError):
        tool.analyze(*args)
