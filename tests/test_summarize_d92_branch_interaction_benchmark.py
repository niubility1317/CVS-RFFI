"""Full synthetic matrices only; never open actual query results."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import summarize_d92_branch_interaction_benchmark as tool
from summarize_d92_confirmation import matrix_definition
from test_summarize_d92_repeated_benchmark import cohorts


def inputs():
    specs, scored = cohorts()
    for i, (spec, data) in enumerate(zip(specs, scored)):
        spec['confirmation'].update(candidate_method=tool.CANDIDATE, candidate_folder='branch_interaction',
            candidate_predictor='evaluate_d92_branch_interaction.py', candidate_mode='d92_branch_interaction_registration',
            capsule=f'/capsule-{i}', branch_ridge_reference_run_id=f'reference-{i}')
        spec['data']['capsule_id'] = spec['confirmation']['reuse_validated_capsule_id']
        spec['joint_benchmark'] = dict(old_accuracy_scope=tool.OLD_SCOPE)
        for row in spec['rows']:
            seed = row['seeds']['model']
            row.update(row_id=f'model-{seed}', source_root=f'/source/{seed}', reuse_row_root=f'/original-{i}/{seed}',
                       expected_checkpoint_sha256=f'{seed:064x}')
        for row in data['results']:
            if row['method'] == 'D92-SGJoint-v1':
                row['method'] = tool.CANDIDATE
            row.update(row_id=f'model-{row["model_seed"]}',
                split_id=':'.join(str(row[k]) for k in tool.KEY[1:]),
                class_count=6+row['new_count'], classes=[f'class-{n}' for n in range(6+row['new_count'])],
                query_count=30*(6+row['new_count']))
    refs, previous = deepcopy(specs), deepcopy(scored)
    for i, (spec, data) in enumerate(zip(refs, previous)):
        spec['run_id'] = f'reference-{i}'
        spec['confirmation'].update(candidate_method=tool.REFERENCE, candidate_folder='branch_ridge',
            candidate_predictor='evaluate_d92_branch_ridge.py', candidate_mode='d92_branch_ridge_registration')
        for row in data['results']:
            if row['method'] == tool.CANDIDATE:
                row['method'] = tool.REFERENCE
                # Give the actual reference distinct values: aliases would fail.
                for field in tool.METRICS:
                    if row[field] is not None: row[field] *= .9
    return specs, scored, refs, previous


def test_complete_dual_baseline_preserves_labels_inputs_and_equal_cells(tmp_path):
    args = inputs(); before = deepcopy(args)
    result = tool.analyze(*args)
    assert args == before
    assert not result['automatic_promotion']
    assert result['baseline_record_audits'][0]['unchanged_d92_records'] == 3600
    assert result['baseline_record_audits'][1]['unchanged_dg_records'] == 12
    for name, baseline in [('vs_d92','D92'),('vs_branch_ridge',tool.REFERENCE)]:
        combined = result['comparisons'][name]['combined']
        assert combined['methods'] == [baseline, tool.CANDIDATE]
        assert combined['rows'] == 9648
        assert combined['acceptance']['require_new_improvement']
        assert combined['acceptance']['old_guard_scope'] == 'all_cells_including_old_only'
        assert all(row['cells'] == 960 for row in combined['tables']['per_k'])
        assert all(row['cells'] == 1200 for row in combined['tables']['per_k_all'])
        assert all(row['cells'] == 240 for row in combined['tables']['per_k_new'])
        assert len(combined['tables']['per_seed_k_delta']) == 16
        assert len(combined['tables']['per_rx_scene_k_delta']) == 48
        pair = {r['method']:r for r in combined['tables']['per_k'] if r['k'] == 1}
        assert pair[tool.CANDIDATE]['harmonic_mean'] == pytest.approx(.65)
        if name == 'vs_branch_ridge':
            assert pair[baseline]['harmonic_mean'] == pytest.approx(.585)
    tool.write(result, tmp_path/'report')
    report = (tmp_path/'report/report.md').read_text(encoding='utf-8')
    assert tool.REFERENCE in report
    assert '旧类固定为 6 类' in report and '新增 20 类 H' in report
    assert '候选绝对性能' in report and '旧类单独任务' in report
    assert '总 support=(6+Nnew)×K' in report
    with pytest.raises(FileExistsError): tool.write(result, tmp_path/'report')


@pytest.mark.parametrize('fault', ['d92_metric','dg_metadata','missing_reference','query_count','class_order',
                                  'capsule','checkpoint','seed_role','wrong_run','matrix','old_guard'])
def test_reject_any_reference_or_pairing_change(fault):
    args = inputs(); specs, data, refs, old = args
    if fault == 'd92_metric': old[0]['results'][0]['old_accuracy'] += .001
    elif fault == 'dg_metadata':
        next(r for r in old[0]['results'] if r['method'] == 'frozen_dg')['unrecognized_metadata'] = 1
    elif fault == 'missing_reference': old[0]['results'].pop()
    elif fault in ('query_count','class_order'):
        row = next(r for r in data[0]['results'] if r['method'] == tool.CANDIDATE)
        if fault == 'query_count': row['query_count'] += 1
        else: row['classes'].reverse()
    elif fault == 'capsule': refs[0]['confirmation']['capsule'] += '-different'
    elif fault == 'checkpoint': refs[0]['rows'][0]['expected_checkpoint_sha256'] = 'a'*64
    elif fault == 'seed_role': refs[0]['rows'][0]['seeds']['augmentation'] = 9
    elif fault == 'wrong_run': refs[0]['run_id'] = 'wrong'
    elif fault == 'matrix': refs[0]['data']['support_seeds'][0] += 1
    else: specs[0]['joint_benchmark']['old_accuracy_scope'] = 'joint only'
    with pytest.raises((ValueError,KeyError)): tool.analyze(*args)


def test_second_baseline_requires_explicit_analysis_only():
    spec = inputs()[0][0]
    spec['baseline_method'] = tool.REFERENCE
    with pytest.raises(ValueError, match='analysis-only'): matrix_definition(spec)
    spec['analysis_only'] = True
    assert matrix_definition(spec)[0] == (tool.REFERENCE, tool.CANDIDATE)
    spec['baseline_method'] = 'D92-renamed'
    with pytest.raises(ValueError, match='analysis-only'): matrix_definition(spec)


def test_reference_guard_cannot_hide_old_only_drop():
    specs, data, refs, old = inputs()
    for cohort in data:
        for row in cohort['results']:
            if row['method'] == tool.CANDIDATE:
                row['old_accuracy'] = .9 if row['new_count'] else 0.
    result = tool.analyze(specs, data, refs, old)
    rows = result['comparisons']['vs_branch_ridge']['combined']['comparisons']
    assert all(r['delta']['old_accuracy'] > 0 for r in rows)
    # Force a separate explicit fixture where old-only contribution tips the guard.
    for cohort, prior in zip(data, old):
        reference = {tuple(r[k] for k in tool.KEY):r for r in prior['results'] if r['method'] == tool.REFERENCE}
        for row in cohort['results']:
            if row['method'] == tool.CANDIDATE:
                row['old_accuracy'] = reference[tuple(row[k] for k in tool.KEY)]['old_accuracy']+.001 if row['new_count'] else 0.
    result = tool.analyze(specs, data, refs, old)
    assert all(not r['old_guard'] and r['delta']['old_accuracy'] > 0 for r in result['comparisons']['vs_branch_ridge']['combined']['comparisons'])


def write_inputs(tmp_path):
    args = inputs(); specs, data, refs, previous = args
    folders = [tmp_path/f'results-{i}' for i in range(4)]
    for folder, spec, scored in zip(folders, specs+refs, data+previous):
        folder.mkdir()
        values = dict(startup=dict(spec=spec,commit='runtime'),
                      complete=dict(status='SCORED',commit='runtime',records=len(scored['results'])),scores=scored)
        for name, value in values.items():
            (folder/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
    return folders


@pytest.mark.parametrize('fault',['running','binding'])
def test_all_four_terminal_and_binding_checks_precede_scores(tmp_path, monkeypatch, fault):
    folders = write_inputs(tmp_path)
    if fault == 'running':
        (folders[-1]/'complete.json').write_text('{"status":"RUNNING"}',encoding='utf-8')
    else:
        path=folders[-1]/'startup.json'; value=tool.read(path); value['spec']['run_id']='wrong'
        path.write_text(json.dumps(value),encoding='utf-8')
    original = Path.open
    def guarded(path, *a, **kw):
        assert path.name != 'scores.json', 'Scores opened before barrier'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    with pytest.raises(ValueError): tool.prepare(folders[:2],folders[2:])


def test_prepare_does_not_modify_any_input_and_records_hashes(tmp_path):
    folders = write_inputs(tmp_path)
    before = {str(p):p.read_bytes() for d in folders for p in d.iterdir()}
    result = tool.prepare(folders[:2],folders[2:])
    assert len(result['input_sha256']) == 12
    assert before == {str(p):p.read_bytes() for d in folders for p in d.iterdir()}
