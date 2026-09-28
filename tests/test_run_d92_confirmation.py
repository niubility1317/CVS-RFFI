import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_d92_confirmation as runner


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def specification(tmp_path):
    root, data = tmp_path / 'run', tmp_path / 'data'
    data_cfg = tmp_path / 'data.json'
    save(data_cfg, dict(output_root=str(data)))
    spec = dict(execution=dict(remote_run_root=str(root), launch_owner='synthetic-owner'),
        confirmation=dict(capsule=str(data / 'capsule'), truth=str(data / 'score_only' / 'truth.json'),
            data_config=str(data_cfg), candidate_config=str(tmp_path / 'candidate.json'),
            native_code=str(tmp_path / 'native'), source_contract=str(tmp_path / 'source_contract.json')),
        rows=[dict(row_id=f'seed-{seed}', output_root=str(root / f'seed-{seed}'), seeds=dict(model=seed),
            source_root=str(tmp_path / f'source-{seed}'), ground_source=str(tmp_path / f'ground-{seed}'))
            for seed in runner.MODEL_SEEDS])
    path = tmp_path / 'spec.json'
    save(path, spec)
    return path, spec


def install_workers(monkeypatch, spec, fail=None):
    calls, lock = [], threading.Lock()
    root = Path(spec['execution']['remote_run_root'])
    def mock_invoke(tool, arguments, log, *, gpu=False):
        args = list(map(str, arguments))
        def arg(name):
            return args[args.index('--' + name) + 1]
        with lock:
            calls.append((tool, args, gpu))
        if fail and tool == fail[0] and fail[1] in str(log):
            raise RuntimeError('synthetic failure')
        if tool == 'build_d92_confirmation_data.py':
            save(Path(spec['confirmation']['capsule']) / 'manifest.json', dict(protocol_schema='p2_min_v1',
                phase2_data_status='VALIDATED_ONCE', capsule_id='synthetic', split_count=900))
        elif tool == 'cvs_native_artifacts.py':
            assert gpu is True
            save(Path(arg('output')) / 'features_complete.json', dict(status='FROZEN_FEATURES_COMPLETE',
                capsule_id='synthetic', query_used_for_fitting=False))
        elif tool in ('cvs_d92_matched.py', 'predict_d92_support_cv.py'):
            assert gpu is False
            assert len([c for c in calls if c[0] == 'cvs_native_artifacts.py']) == 4
            output = Path(arg('output'))
            if tool == 'predict_d92_support_cv.py':
                assert (output.parent / 'predictions_complete.json').exists()
            save(output / 'predictions_complete.json', dict(status='PREDICTIONS_COMPLETE',
                capsule_id='synthetic', split_count=900, truth_read=False))
        elif tool == 'score_d92_confirmation.py':
            assert all(s['status'] == 'PREDICTIONS_COMPLETE' for s in runner.read(root / 'state.json').values())
            assert len([c for c in calls if c[0] == 'predict_d92_support_cv.py']) == 4
            # Deliberately no truth file exists: orchestration must never open it.
            assert not Path(spec['confirmation']['truth']).exists()
            save(arg('output'), dict(status='SCORED', results=[{}], selection_feedback_forbidden=True))
        else:
            raise AssertionError(tool)
    monkeypatch.setattr(runner, 'invoke', mock_invoke)
    monkeypatch.setattr(runner, 'prepare_ground', lambda row, confirmation: 'synthetic-checkpoint-sha')
    return calls


def test_all_models_freeze_before_independent_score(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    calls = install_workers(monkeypatch, spec)
    runner.run(path, 'synthetic-commit')
    assert calls[0][0] == 'build_d92_confirmation_data.py'
    assert [c[0] for c in calls[1:5]] == ['cvs_native_artifacts.py'] * 4
    assert calls[-1][0] == 'score_d92_confirmation.py'
    root = Path(spec['execution']['remote_run_root'])
    assert runner.read(root / 'complete.json')['status'] == 'SCORED'
    startup = runner.read(root / 'startup.json')
    assert startup['spec'] == spec and startup['commit'] == 'synthetic-commit'
    assert startup['worker_policy']['max_feature_workers'] == 1


@pytest.mark.parametrize('tool', ['cvs_d92_matched.py', 'predict_d92_support_cv.py'])
def test_prediction_failure_preserves_other_rows_without_scoring(tmp_path, monkeypatch, tool):
    path, spec = specification(tmp_path)
    calls = install_workers(monkeypatch, spec, fail=(tool, '2026092701'))
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runner.run(path, 'synthetic-commit')
    assert not any(c[0] == 'score_d92_confirmation.py' for c in calls)
    root = Path(spec['execution']['remote_run_root'])
    state = runner.read(root / 'state.json')
    assert state['seed-2026092701']['status'] == 'TECHNICAL_FAILURE'
    assert all(state[f'seed-{seed}']['status'] == 'PREDICTIONS_COMPLETE' for seed in runner.MODEL_SEEDS[1:])
    assert runner.read(root / 'workflow_state.json')['status'] == 'TECHNICAL_FAILURE'
    assert not (root / 'complete.json').exists()


def test_feature_failure_never_launches_prediction_or_score(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    calls = install_workers(monkeypatch, spec, fail=('cvs_native_artifacts.py', '2026092702'))
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runner.run(path, 'synthetic-commit')
    assert [c[0] for c in calls] == ['build_d92_confirmation_data.py', 'cvs_native_artifacts.py', 'cvs_native_artifacts.py']
    root = Path(spec['execution']['remote_run_root'])
    assert runner.read(root / 'workflow_state.json')['status'] == 'TECHNICAL_FAILURE'
    assert runner.read(root / 'state.json')['seed-2026092702']['status'] == 'TECHNICAL_FAILURE'


def test_existing_run_not_overwritten(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    root = Path(spec['execution']['remote_run_root'])
    root.mkdir()
    save(root / 'state.json', dict(original=True))
    calls = install_workers(monkeypatch, spec)
    with pytest.raises(FileExistsError):
        runner.run(path, 'synthetic-commit')
    assert calls == [] and runner.read(root / 'state.json') == dict(original=True)


def test_workers_have_bounded_cpu_and_gpu_visibility():
    cpu, gpu = runner.worker_environment(), runner.worker_environment(gpu=True)
    assert cpu['CUDA_VISIBLE_DEVICES'] == '' and gpu['CUDA_VISIBLE_DEVICES'] == '0'
    for env in (cpu, gpu):
        assert all(env[name] == '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))


@pytest.mark.parametrize('fault', [None, 'checkpoint_sha', 'source_classes', 'ground_classes'])
def test_ground_copy_bound_to_actual_source_checkpoint(tmp_path, fault):
    sys.path.insert(0, str(runner.RELEASE / 'code'))
    from cvsrffi.phase1_center_lowrank_prototype_bundle import NPZ_NAME
    source, origin, output = tmp_path / 'source', tmp_path / 'original-ground', tmp_path / 'row'
    classes = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
    # The reference source contract has no classes or native_role_comparison.
    contract = dict(role_ids=dict(L_s=['source-id']),
        source_rxs=['source-rx'], source_days=[1], ratios=[0.1], split_seed=42,
        num_classes=6)
    actual_classes = list(reversed(classes)) if fault == 'source_classes' else classes
    save(source / 'source_contract.json', dict(contract, native_role_comparison='EXACT_MATCH', classes=actual_classes))
    save(tmp_path / 'contract.json', contract)
    save(source / 'initialization.json', dict(scratch_only=True, checkpoint_sources=[],
        target_contact=False, source_roles='EXACT_MATCH'))
    save(source / 'completion.json', dict(status='TRAINING_COMPLETE', epochs=200,
        target_evaluated=False))
    save(source / 'resolved_config.json', dict(seed=2026092701, checkpoint_selection='final_only',
        a1_periodic_target_start=0, a1_final_weak_reference=False))
    (source / 'final_ssdg.pth').write_bytes(b'synthetic-checkpoint')
    expected = runner.sha(source / 'final_ssdg.pth')
    save(origin / 'manifest.json', dict(checkpoint_sha256='wrong' if fault == 'checkpoint_sha' else expected,
        target_access=False, source_role='L_s', formal_phase2_eligible=True,
        provenance_status='CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L', member_allowlist=[NPZ_NAME]))
    np.savez(origin / NPZ_NAME, class_registry=list(reversed(classes)) if fault == 'ground_classes' else classes)
    (origin / 'do_not_copy_source_sample_cache.npz').write_bytes(b'cache')
    output.mkdir()
    row = dict(source_root=str(source), ground_source=str(origin), output_root=str(output),
               seeds=dict(model=2026092701))
    confirmation = dict(source_contract=str(tmp_path / 'contract.json'), old_classes=classes)
    if fault is not None:
        with pytest.raises(ValueError, match='class registry mismatch|Ground source/checkpoint'):
            runner.prepare_ground(row, confirmation)
        assert not (output / 'ground').exists()
    else:
        assert runner.prepare_ground(row, confirmation) == expected
        assert {p.name for p in (output / 'ground').iterdir()} == {'manifest.json', NPZ_NAME}
        assert runner.read(output / 'ground_provenance.json')['checkpoint_sha256'] == expected


@pytest.mark.parametrize('fault', [None, 'capsule_id', 'split_count', 'protocol_schema', 'phase2_data_status'])
def test_reuses_validated_capsule_without_builder_or_data_revalidation(tmp_path, monkeypatch, fault):
    path, spec = specification(tmp_path)
    spec['confirmation']['reuse_validated_capsule_id'] = 'synthetic'
    save(path, spec)
    manifest = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
                    capsule_id='synthetic', split_count=900)
    if fault:
        manifest[fault] = 899 if fault == 'split_count' else 'wrong'
    capsule_manifest = Path(spec['confirmation']['capsule']) / 'manifest.json'
    save(capsule_manifest, manifest)
    original = capsule_manifest.read_bytes()
    calls = install_workers(monkeypatch, spec)
    if fault:
        with pytest.raises(ValueError, match='capsule'):
            runner.run(path, 'synthetic-commit')
        assert calls == []
    else:
        runner.run(path, 'synthetic-commit')
        assert not any(c[0] == 'build_d92_confirmation_data.py' for c in calls)
        record = runner.read(Path(spec['execution']['remote_run_root']) / 'data_reuse.json')
        assert record['capsule_id'] == 'synthetic' and record['data_revalidated'] is False
        assert record['data_rebuilt'] is False and record['truth_read'] is False
    assert capsule_manifest.read_bytes() == original
