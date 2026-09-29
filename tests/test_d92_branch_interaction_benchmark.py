import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_d92_branch_interaction_benchmark as prep
import preflight_d92_branch_interaction as preflight
import run_d92_confirmation as runner
from publish_d92_confirmation import release_tool_paths, REMOTE as PUBLISH_REMOTE
from test_run_d92_confirmation import reused_specification, save


def install_branch_cache(tmp_path, spec):
    source = tmp_path/'frozen-branch-origin'
    producer = runner.read(prep.ROOT/prep.PRODUCER)
    producer_path = tmp_path/'producer.json'
    save(producer_path, producer)
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS, runner.CANDIDATES[8])))
    spec['confirmation'].update(frozen_branch_feature_source_root=str(source),
        frozen_branch_feature_producer_config=str(producer_path),
        branch_ridge_reference_root=str(source), branch_ridge_reference_run_id=source.name)
    manifest = runner.read(Path(spec['confirmation']['capsule'])/'manifest.json')
    manifest['received_count'] = 16
    save(Path(spec['confirmation']['capsule'])/'manifest.json', manifest)
    from export_d92_branch_features import CACHE_SCHEMA, FEATURE_CONTRACT
    for row in spec['rows']:
        cache = source/row['row_id']/'branch_features'
        baseline = source/row['row_id']/'branch_ridge'
        row.update(reuse_branch_features_root=str(cache), reuse_branch_ridge_root=str(baseline))
        common = dict(schema=CACHE_SCHEMA, feature_contract=FEATURE_CONTRACT,
            capsule_id='synthetic', checkpoint_sha256='a'*64, model_seed=row['seeds']['model'],
            truth_read=False, source_data_access=False, adapted_state_inherited=False, encoder_updated=False,
            native_batch_size=1)
        save(cache/'startup.json', dict(common, config=producer, query_fit_access=False))
        save(cache/'features_complete.json', dict(common, status='BRANCH_FEATURES_COMPLETE',
            count=16, query_used_for_fitting=False, native_parameters_unchanged=True, native_buffers_unchanged=True))
        save(cache/'checkpoint_provenance.json', {'synthetic': True})
        (cache/'received_branch_features.npz').write_bytes(b'synthetic-placeholder-not-loaded')
        save(baseline/'predictions_complete.json', dict(status='PREDICTIONS_COMPLETE', capsule_id='synthetic', split_count=2, truth_read=False))
        (baseline/'predictions.jsonl').write_text('UNREAD_SYNTHETIC_BASELINE', encoding='utf-8')
    return producer


def test_prepared_full_matrix_and_frozen_reference_bindings():
    docs = prep.documents(prep.ROOT, 'synthetic-commit')
    for cohort in ('rx3', 'rx1'):
        spec = docs[f'configs/d92_branch_interaction_repeat_{cohort}_20260929.json']
        data = docs[f'configs/d92_branch_interaction_repeat_{cohort}_data_20260929.json']
        old = runner.read(prep.ROOT/f'configs/d92_branch_ridge_repeat_{cohort}_20260929.json')
        conf = spec['confirmation']
        assert data['shots'] == [1, 5, 10, 20] and data['new_counts'] == [0, 2, 5, 10, 20]
        assert data == runner.read(prep.ROOT/f'configs/d92_branch_ridge_repeat_{cohort}_data_20260929.json')
        assert conf['candidate'] == runner.read(prep.ROOT/prep.FROZEN)['algorithm']
        assert conf['candidate_config'] == spec['code']['cwd']+'/'+prep.FROZEN
        assert conf['branch_ridge_reference_run_id'] == old['run_id']
        assert conf['branch_ridge_reference_root'] == old['execution']['remote_run_root']
        assert runner.reuse_branch_cache(spec) and runner.reuse_frozen_rows(spec)
        assert not runner.needs_multiview(conf) and not runner.reuse_multiview_cache(spec)
        assert spec['checkpoint']['runtime_checkpoint_reload'] is False
        for row, previous in zip(spec['rows'], old['rows']):
            assert row['reuse_row_root'] == previous['reuse_row_root']
            assert row['reuse_branch_features_root'] == previous['output_root']+'/branch_features'
            assert row['reuse_branch_ridge_root'] == previous['output_root']+'/branch_ridge'
            assert row['seeds'] == previous['seeds'] and row['gpu'] is None
        assert 'branch_interaction' in conf['candidate_folder']
        assert spec['metrics_plan']['secondary_reference_method'] == 'D92-BranchRidge-v1'
    assert sum(v['confirmation']['predictions_total'] for v in docs.values() if 'confirmation' in v) == 9648


@pytest.mark.parametrize('fault', ['missing_cache', 'wrong_model_path', 'wrong_baseline_path', 'same_output', 'wrong_reference_run', 'wrong_method'])
def test_invalid_cache_reuse_is_rejected_before_new_run(tmp_path, fault):
    path, spec = reused_specification(tmp_path)
    install_branch_cache(tmp_path, spec)
    if fault == 'missing_cache': spec['rows'][0].pop('reuse_branch_features_root')
    if fault == 'wrong_model_path': spec['rows'][0]['reuse_branch_features_root'] = spec['rows'][1]['reuse_branch_features_root']
    if fault == 'wrong_baseline_path': spec['rows'][0]['reuse_branch_ridge_root'] = spec['rows'][1]['reuse_branch_ridge_root']
    if fault == 'same_output': spec['confirmation']['frozen_branch_feature_source_root'] = spec['execution']['remote_run_root']
    if fault == 'wrong_reference_run': spec['confirmation']['branch_ridge_reference_run_id'] = 'different'
    if fault == 'wrong_method': spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS, runner.CANDIDATES[7])))
    save(path, spec)
    with pytest.raises(ValueError): runner.run(path, 'synthetic')
    assert not Path(spec['execution']['remote_run_root']).exists()


@pytest.mark.parametrize('fault', [None, 'config', 'checkpoint', 'source_access', 'incomplete_baseline'])
def test_cache_metadata_never_reads_features_predictions_or_checkpoint(tmp_path, monkeypatch, fault):
    _, spec = reused_specification(tmp_path)
    install_branch_cache(tmp_path, spec)
    row, conf = spec['rows'][0], spec['confirmation']
    cache = Path(row['reuse_branch_features_root'])
    if fault == 'config':
        startup = runner.read(cache/'startup.json'); startup['config']['algorithm']['ridge_coefficient'] = 2
        save(cache/'startup.json', startup)
    if fault in ('checkpoint', 'source_access'):
        marker = runner.read(cache/'features_complete.json')
        marker['checkpoint_sha256' if fault == 'checkpoint' else 'source_data_access'] = 'bad' if fault == 'checkpoint' else True
        save(cache/'features_complete.json', marker)
    if fault == 'incomplete_baseline': save(Path(row['reuse_branch_ridge_root'])/'predictions_complete.json', {'status': 'RUNNING'})
    manifest = runner.read(Path(conf['capsule'])/'manifest.json')
    original = runner.read
    def guarded_read(path):
        assert Path(path).name in ('startup.json', 'features_complete.json', 'producer.json', 'predictions_complete.json')
        return original(path)
    monkeypatch.setattr(runner, 'read', guarded_read)
    if fault:
        with pytest.raises(ValueError): runner.validate_reused_branch_metadata(row, conf, manifest)
    else:
        result = runner.validate_reused_branch_metadata(row, conf, manifest)
        assert result['encoder_executed'] is result['checkpoint_reloaded'] is result['scores_read'] is False


def test_cpu_runner_reuses_both_baselines_and_only_launches_new_candidate(tmp_path, monkeypatch):
    path, spec = reused_specification(tmp_path)
    install_branch_cache(tmp_path, spec)
    save(path, spec)
    origins = [tmp_path/'old-run', tmp_path/'frozen-branch-origin']
    previous = {p: p.read_bytes() for root in origins for p in root.rglob('*') if p.is_file()}
    calls = []
    def invoke(tool, arguments, log, *, gpu=False):
        assert gpu is False
        args = list(map(str, arguments)); calls.append(tool)
        value = lambda key: args[args.index('--'+key)+1]
        if tool == 'evaluate_d92_branch_interaction.py':
            assert Path(value('branch-features')).parent.parent == tmp_path/'frozen-branch-origin'
            assert Path(value('row-root')).parent == tmp_path/'old-run'
            save(Path(value('output'))/'predictions_complete.json', dict(status='PREDICTIONS_COMPLETE', capsule_id='synthetic', split_count=2, truth_read=False))
        elif tool == 'score_d92_confirmation.py':
            assert calls.count('evaluate_d92_branch_interaction.py') == 4
            save(value('output'), dict(status='SCORED', results=[], selection_feedback_forbidden=True))
        else: raise AssertionError('Unexpected encoder/baseline/build invocation: '+tool)
    monkeypatch.setattr(runner, 'invoke', invoke)
    runner.run(path, 'synthetic-commit')
    assert calls.count('evaluate_d92_branch_interaction.py') == 4 and calls[-1] == 'score_d92_confirmation.py'
    assert all(p.read_bytes() == content for p, content in previous.items())
    startup = runner.read(Path(spec['execution']['remote_run_root'])/'startup.json')
    assert startup['worker_policy']['max_feature_workers'] == 0 and startup['worker_policy']['feature_gpu'] is None


def test_release_closure_and_preflight_is_cpu_metadata_only(tmp_path, monkeypatch, capsys):
    _, spec = reused_specification(tmp_path)
    producer = install_branch_cache(tmp_path, spec)
    release_parent = tmp_path/'releases'; release_parent.mkdir()
    spec['code'] = dict(cwd=str(release_parent/'new-release'))
    spec['run_id'] = 'synthetic-interaction-repeat'
    paths = release_tool_paths(spec['confirmation'])
    for name in ('tools/evaluate_d92_branch_interaction.py', 'tools/export_d92_branch_features.py',
                 'tools/export_d92_branch_support_features.py', prep.PRODUCER): assert name in paths
    assert not runner.needs_multiview(spec['confirmation'])
    assert "c.get('multiview',False)" in PUBLISH_REMOTE
    script = preflight.remote_script([spec], producer)
    assert 'nvidia-smi' not in script and 'load_native' not in script
    import os, shutil
    monkeypatch.setattr(os, 'getloadavg', lambda: (0., 0., 0.), raising=False)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=3*1024**3))
    original = Path.read_text
    def read_text(path, *args, **kwargs):
        assert path.name in ('manifest.json', 'features_complete.json', 'startup.json', 'predictions_complete.json')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    exec(compile(script, '<synthetic-preflight>', 'exec'), {})
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'VERIFIED' and result['cpu_only'] is True
    assert result['checkpoint_loaded'] is result['feature_arrays_read'] is result['scores_read'] is False
