"""Bounded source-stream orchestration, with subprocesses simulated."""
import importlib
import json
from pathlib import Path

import pytest


def launcher():
    return importlib.import_module('experiments.pamf_phase3_open_world.stream_launch')


@pytest.fixture
def runtime(tmp_path):
    code = dict(base_release=str(tmp_path / 'base'), frozen_sources={name: str(tmp_path / name)
                for name in ('F', 'AF', 'BASE', 'FOLD', 'G')})
    return dict(run_id='stream-pilot', run_root=str(tmp_path / 'stream-pilot'),
        release_root=str(tmp_path), python='python', code_commit='a' * 40, launch_owner='stream-owner',
        allowed_gpus=[2, 3], max_added_workers_per_gpu=2, minimum_free_mb=4096,
        export=dict(code_runtime=code, source_model_root=str(tmp_path / 'source'),
            source_cache=str(tmp_path / 'source-cache'), source_contract=str(tmp_path / 'source-contract.json'),
            model_seed=2026092701, fold=dict(unknown_source_labels=[0, 1], heldout_source_rx=8),
            batch_size=128, branch_export_root=str(tmp_path / 'old-branch-export')))


def markers():
    common = dict(source_only=True, target_reads=0)
    return dict(export=dict(common, status='SOURCE_STREAM_INPUTS_EXPORTED', train_count=3360,
                    calibration_count=14400, discovery_count=1050, future_count=5400,
                    optimizer_updates=0, head_changed=False, parent_epochs=20, parent_updates=540),
                calibrate=dict(common, status='SOURCE_STREAM_CALIBRATED', train_count=3360,
                    calibration_count=14400, class_count=4, U_reads=0, future_reads=0),
                predict=dict(common, status='ALL_STREAM_PREDICTIONS_COMPLETE', truth_read=False,
                    query_fit=False, discovery_count=1050, future_count=5400,
                    state_names=['F', 'D0R0', 'D1R0', 'D0R1', 'D1R1']),
                score=dict(common, status='SCORED_SOURCE_STREAM_COMPLETE',
                    next_step='NO_PROMOTION', eligible_states=[]))


def simulate(monkeypatch, runtime, *, bad_stage=None, mutate=None, exit_stage=None):
    module = launcher()
    root = Path(runtime['run_root'])
    seen, stages = [], markers()
    if mutate:
        mutate(stages)
    active = [0, 0]

    class Child:
        pid = 1234

        def __init__(self, command, **kwargs):
            if command[4].endswith('.stream_export'):
                kind = 'export'
            elif command[4].endswith('.stream_run'):
                kind = command[5]
            elif command[4].endswith('.stream_scoring'):
                kind = 'score'
            else:
                raise AssertionError('Unexpected command: ' + str(command))
            assert len(seen) == ['export', 'calibrate', 'predict', 'score'].index(kind)
            self.kind = kind
            seen.append((kind, command, kwargs))
            active[0] += 1
            active[1] = max(active)
            kwargs['stdout'].write('simulated ' + kind + '\n')
            folder = root / ('calibration' if kind == 'calibrate' else kind)
            folder.mkdir()
            marker = dict(stages[kind])
            if kind == bad_stage:
                marker['status'] = 'INCOMPLETE'
            (folder / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')

        def wait(self):
            active[0] -= 1
            return 9 if self.kind == exit_stage else 0

    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    return seen, active


def test_four_stages_cpu_after_export_and_no_promotion_still_finishes(runtime, monkeypatch):
    module = launcher()
    seen, active = simulate(monkeypatch, runtime)
    gpu_calls = []
    monkeypatch.setattr(module, 'choose_gpu', lambda rt: gpu_calls.append(rt) or 3)
    state = module.dispatch(runtime)
    root = Path(runtime['run_root'])
    assert [row[0] for row in seen] == ['export', 'calibrate', 'predict', 'score']
    assert len(gpu_calls) == 1 and active[1] == 1
    assert [row[2]['env']['CUDA_VISIBLE_DEVICES'] for row in seen] == ['3', '', '', '']
    assert all(row[2]['cwd'] == runtime['release_root'] for row in seen)
    assert seen[0][1][-2:] == ['--device', 'cuda:0']
    assert '--source-inputs' in seen[1][1] and str(root / 'export/source_inputs.npz') in seen[1][1]
    assert str(root / 'export/discovery/public/features.npz') in seen[2][1]
    assert str(root / 'export/evaluation/public/features.npz') in seen[2][1]
    assert str(root / 'configs/export.json') in seen[2][1]
    assert str(root / 'export/discovery/truth/truth.npz') in seen[3][1]
    assert str(root / 'export/evaluation/truth/truth.npz') in seen[3][1]
    config = json.loads((root / 'configs/export.json').read_text())
    assert config['row_id'] == 'source-joint-fold0-s2026092701'
    assert config['branch_export_root'] == runtime['export']['branch_export_root']
    assert state['status'] == 'ANALYZED_SOURCE_ONLY'
    assert state['target_performance_claim'] is False
    assert state['next_step'] == 'NO_PROMOTION'
    assert state['foreign_processes_stopped'] == 0
    assert state['eligible_states'] == []
    assert json.loads((root / 'pipeline_state.json').read_text()) == state
    with pytest.raises(FileExistsError):
        module.dispatch(runtime)
    assert len(seen) == 4


@pytest.mark.parametrize('bad_stage', ['export', 'calibrate', 'predict', 'score'])
def test_stage_marker_failure_preserves_log_and_stops_only_this_pipeline(runtime, monkeypatch, bad_stage):
    module = launcher()
    seen, _ = simulate(monkeypatch, runtime, bad_stage=bad_stage)
    monkeypatch.setattr(module, 'choose_gpu', lambda rt: 2)
    with pytest.raises(ValueError):
        module.dispatch(runtime)
    root = Path(runtime['run_root'])
    state = json.loads((root / 'pipeline_state.json').read_text())
    assert state['status'] == 'STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE'
    assert state['jobs'][bad_stage]['status'] == 'FAILED'
    assert len(seen) == ['export', 'calibrate', 'predict', 'score'].index(bad_stage) + 1
    assert (root / 'logs' / (bad_stage + '.log')).read_text().startswith('simulated')
    assert bad_stage in (root / 'logs/failure.log').read_text()
    assert state['foreign_processes_stopped'] == 0


@pytest.mark.parametrize('bad_value', ['export_count', 'calibration_u', 'predict_truth', 'source_flag',
                                     'score_eligibility', 'control_only_promotion'])
def test_stage_boundary_fields_are_checked(runtime, monkeypatch, bad_value):
    def mutate(values):
        if bad_value == 'export_count': values['export']['train_count'] = 3359
        elif bad_value == 'calibration_u': values['calibrate']['U_reads'] = 1
        elif bad_value == 'predict_truth': values['predict']['truth_read'] = True
        elif bad_value == 'source_flag': values['export']['target_reads'] = False
        elif bad_value == 'score_eligibility': values['score']['eligible_states'] = ['D1R1']
        else:
            values['score']['eligible_states'] = ['D0R1']
            values['score']['next_step'] = 'SOURCE_JOINT_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'
    module = launcher()
    simulate(monkeypatch, runtime, mutate=mutate)
    monkeypatch.setattr(module, 'choose_gpu', lambda rt: 2)
    with pytest.raises(ValueError):
        module.dispatch(runtime)


def test_child_exit_failure_does_not_start_later_stages(runtime, monkeypatch):
    module = launcher()
    seen, _ = simulate(monkeypatch, runtime, exit_stage='calibrate')
    monkeypatch.setattr(module, 'choose_gpu', lambda rt: 2)
    with pytest.raises(RuntimeError, match='exit=9'):
        module.dispatch(runtime)
    assert [item[0] for item in seen] == ['export', 'calibrate']


@pytest.mark.parametrize('fault', ['runtime_target', 'export_target', 'capacity', 'gpu_bool', 'fold'])
def test_runtime_whitelists_and_capacity_are_strict(runtime, fault):
    if fault == 'runtime_target': runtime['target_root'] = 'forbidden'
    elif fault == 'export_target': runtime['export']['target_root'] = 'forbidden'
    elif fault == 'capacity': runtime['max_added_workers_per_gpu'] = 3
    elif fault == 'gpu_bool': runtime['allowed_gpus'] = [True]
    else: runtime['export']['fold']['heldout_source_rx'] = 1
    with pytest.raises(ValueError):
        launcher().validate_runtime(runtime)
    assert not Path(runtime['run_root']).exists()


def test_reused_gpu_choice_uses_available_memory_without_foreign_process_actions(runtime, monkeypatch):
    module = launcher()
    commands = []

    def output(command, **kwargs):
        commands.append(command)
        return '2, 6000\n3, 5000\n'

    monkeypatch.setattr(module.subprocess, 'check_output', output)
    assert module.choose_gpu(runtime) == 2
    assert len(commands) == 1 and commands[0][0] == 'nvidia-smi'
    assert commands[0][1] == '--query-gpu=index,memory.free'
