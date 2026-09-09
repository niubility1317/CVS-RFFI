"""Budget curriculum and exploratory observation contracts, without target data."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/scripts')]
from cvsrffi.a1_budget_schedule import reference_epoch, reference_total, save_budget_snapshot, u_satellite_scenario
from cvsrffi.muse_ssdg import select_adv3b02_u_satellite_scenario
from cvsrffi.schedule import build_stage_state
from SSDG import train_ssdg as train
import run_a1_fast_v2 as runner
import run_a1_r3_budgets as budgets


@pytest.fixture(scope='module')
def matrix():
    return budgets.budget_matrix()


@pytest.mark.parametrize('total,gpu,s2,s3,label,mix,ramp', [
    (120, 2, 10, 41, 78, 66, 24),
    (160, 3, 13, 55, 104, 88, 32),
])
def test_full_budget_curriculum(matrix, total, gpu, s2, s3, label, mix, ramp):
    row = next(r for r in matrix['rows'] if r['id'] == f'R3_B{total}')
    args = train.build_arg_parser().parse_args(runner.v2_command(
        matrix, project_root=Path('/p'), run_root=Path('/r'), row=row)[3:])
    assert row['gpu'] == gpu
    assert args.epochs == args.muse_final_epoch == total
    assert args.label_epochs == label and args.pseudo_epochs == total-label
    assert args.stage1_epochs + 1 == args.muse_s2a_start == s2
    assert args.stage2_epochs + 1 == args.muse_s3a_start == s3
    assert build_stage_state(s2-1, args)['phase'] == 'S1_core'
    assert build_stage_state(s2, args)['phase'] == 'S2_stabilize_aux'
    assert build_stage_state(s3, args)['phase'] == 'S3_refine_aux'
    assert args.mixstyle_late_start == mix and args.mixstyle_late_ramp_epochs == ramp
    assert args.source_episode_structural_warmup_epochs == -1
    assert args.source_episode_structural_start_epoch == -1
    assert args.source_episode_warmup_epochs == round(25*total/200)
    assert args.test_eval_start_epoch == 999999 and args.test_eval_interval == 0
    assert args.from_scratch and not args.baseline_ckpt and not args.teacher_ckpt
    assert args.use_a1_r3 and args.a1_r3_aux_scale == 1 and args.a1_ecrs_cross_rx_weight == 0
    assert not args.a1_runtime_fast and not args.a1_ema_startup_average
    assert args.seed == 392005 and args.batch_size == 128 and args.muse_unlabeled_batch_size == 256
    train._validate_a1_scratch_only(args)
    train._validate_daot_config(args)
    train._muse_config_from_args(args)
    points = {int(x) for x in args.a1_budget_snapshot_epochs.split(',')}
    assert points == {total*i//4 for i in (1, 2, 3, 4)} | set(range(total//2, total+1, 10))
    assert max(map(int, args.daot_diagnostic_epochs.split(','))) == total
    assert reference_epoch(args, 1) == 1 and reference_epoch(args, total) == 200
    for boundary in (40, 90, 160, 180):
        end = boundary*total//200
        assert reference_epoch(args, end) <= boundary < reference_epoch(args, end+1)
    with pytest.raises(ValueError):
        reference_epoch(args, total+1)


def test_legacy_clock_and_snapshot_are_unchanged(tmp_path):
    args = SimpleNamespace(epochs=200, seed=392005)
    assert reference_total(args) == 200
    for epoch in (1, 40, 41, 90, 91, 160, 181, 200):
        assert reference_epoch(args, epoch) == epoch
        assert u_satellite_scenario(args, epoch, 7) == select_adv3b02_u_satellite_scenario(epoch, 7, args.seed)
    assert save_budget_snapshot(args, 200, tmp_path, {}, lambda *_: pytest.fail('legacy write')) is None
    assert not list(tmp_path.iterdir())


def test_snapshot_is_published_only_after_complete_write_and_refuses_overwrite(tmp_path):
    args = SimpleNamespace(a1_r3_budget_mode=True, a1_budget_snapshot_epochs='30')
    final = tmp_path/'epoch_030_ssdg.pth'
    original = {'epoch': 30, 'checkpoint_role': 'original'}
    def save(path, payload):
        assert path.name.endswith('.writing') and not final.exists()
        assert payload['epoch'] == 30
        path.write_bytes(b'complete checkpoint')
    assert save_budget_snapshot(args, 29, tmp_path, original, save) is None
    assert save_budget_snapshot(args, 30, tmp_path, original, save) == str(final)
    assert final.read_bytes() == b'complete checkpoint'
    assert not final.with_suffix('.pth.writing').exists()
    assert original['checkpoint_role'] == 'original'
    with pytest.raises(FileExistsError):
        save_budget_snapshot(args, 30, tmp_path, original, save)
    assert final.read_bytes() == b'complete checkpoint'


def test_failed_snapshot_never_publishes_partial_weights(tmp_path):
    args = SimpleNamespace(a1_r3_budget_mode=True, a1_budget_snapshot_epochs='40')
    def fail(path, payload):
        path.write_bytes(b'partial')
        raise OSError('simulated interrupted write')
    with pytest.raises(OSError):
        save_budget_snapshot(args, 40, tmp_path, {}, fail)
    assert not (tmp_path/'epoch_040_ssdg.pth').exists()
    with pytest.raises(FileExistsError):
        save_budget_snapshot(args, 40, tmp_path, {}, fail)


@pytest.mark.parametrize('coverage', [672000, 671999])
def test_periodic_prediction_precedes_independent_scoring(tmp_path, monkeypatch, coverage):
    row = {'id': 'R3_B120', 'gpu': 2, 'options': {'--epochs': '120', '--a1_budget_evaluation_start_epoch': '60'}}
    folder = tmp_path/row['id']; folder.mkdir()
    logs = tmp_path/'logs'; logs.mkdir()
    checkpoint = folder/'epoch_060_ssdg.pth'; checkpoint.write_bytes(b'fixed')
    state = {}; calls = []
    monkeypatch.setattr(runner, 'gpu_compute_pids', lambda gpu: [123])
    def run(command, **kwargs):
        assert kwargs['check'] and kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2'
        if '--input-package' in command:
            assert not calls and '--truth' not in command and '--mode' in command
            assert Path(command[command.index('--checkpoint')+1]) == checkpoint
            output = Path(command[command.index('--output-root')+1]); output.mkdir(parents=True)
            (output/'predictions.json').write_text('{"fixed": true}', encoding='utf-8')
            calls.append('predict')
        else:
            assert calls == ['predict'] and '--truth' in command
            assert Path(command[command.index('--predictions')+1]).read_text() == '{"fixed": true}'
            Path(command[command.index('--output')+1]).write_text(json.dumps({'record_count': coverage}), encoding='utf-8')
            calls.append('score')
    monkeypatch.setattr(budgets.subprocess, 'run', run)
    def observe():
        return budgets.periodic_target_evaluation(row, project=tmp_path, root=tmp_path, logs=logs, row_state=state, capacity=2)
    assert not observe() and not calls  # no readiness marker yet
    (folder/'source_eval_epoch_060.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(runner, 'gpu_compute_pids', lambda gpu: [123, 456])
    assert not observe() and not calls  # full GPU is deferred
    monkeypatch.setattr(runner, 'gpu_compute_pids', lambda gpu: [123])
    if coverage != 672000:
        with pytest.raises(ValueError, match='coverage'):
            observe()
        assert state['scored_epochs'] == []
    else:
        assert not observe() and state['scored_epochs'] == [60]
        assert not observe()  # completed epoch is never resubmitted
        scope = json.loads((folder/'target_epochs/E060/evaluation_scope.json').read_text())
        assert scope['feeds_training'] is False and 'EXPLORATORY' in scope['scope']
    assert calls == ['predict', 'score'] and checkpoint.read_bytes() == b'fixed'


def test_completion_checks_row_budget_and_preserves_default(tmp_path, monkeypatch):
    checked = []; evaluated = []
    def verify(path, expected_epoch=200):
        checked.append(expected_epoch)
        return {'args': {'from_scratch': True}}
    monkeypatch.setattr(runner, 'verify_checkpoint', verify)
    monkeypatch.setattr(runner, 'evaluate', lambda *a, **kw: evaluated.append(True))
    for total in (120, 160):
        result = runner.complete_row({'final_evaluation': 'exploratory_periodic_target'},
            {'id': 'row', 'options': {'--epochs': str(total)}}, project=tmp_path, root=tmp_path, logs=tmp_path)
        assert result == 'EXPLORATORY_SCORED_PENDING_ANALYSIS'
    assert evaluated == [] and checked == [120, 160]
    assert runner.complete_row({}, {'id': 'old'}, project=tmp_path, root=tmp_path, logs=tmp_path) == 'SCORED_PENDING_ANALYSIS'
    assert checked == [120, 160, 200] and evaluated == [True]


@pytest.mark.parametrize('missing', ['checkpoint', 'marker', None])
def test_exited_training_fails_missing_artifacts_but_waits_for_gpu(tmp_path, monkeypatch, missing):
    """Exercise the dispatcher loop, bounding waits so regressions cannot hang pytest."""
    row = {'id': 'R3_B120', 'gpu': 2, 'options': {
        '--epochs': '120', '--a1_budget_evaluation_start_epoch': '60'}}
    matrix = {'rows': [row], 'max_gpu_processes': 2,
              'final_evaluation': 'exploratory_periodic_target'}
    (tmp_path/'runs').mkdir(); (tmp_path/'logs').mkdir()
    root = tmp_path/'runs/review_exit'
    monkeypatch.setattr(sys, 'argv', ['runner', '--project-root', str(tmp_path), '--run-id', 'review_exit'])
    monkeypatch.setattr(runner, 'required_inputs', lambda *_: [])
    monkeypatch.setattr(runner, 'v2_command', lambda *a, **kw: ['fake-training'])
    monkeypatch.setattr(runner, 'predecessor_complete', lambda *_: True)
    monkeypatch.setattr(runner, 'gpu_compute_pids', lambda *_: [])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: None)
    class Finished:
        pid = 456
        def poll(self):
            return 0
    def launch(*a, **kw):
        folder = root/row['id']
        for epoch in range(60, 121, 10):
            if not (missing == 'checkpoint' and epoch == 60):
                (folder/f'epoch_{epoch:03d}_ssdg.pth').write_bytes(b'fixed')
            if not (missing == 'marker' and epoch == 60):
                (folder/f'source_eval_epoch_{epoch:03d}.json').write_text('{}', encoding='utf-8')
        return Finished()
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    observations = []; sleeps = []; finalizations = []
    def observe(*a, **kw):
        observations.append(True)
        # Full GPU on the first observation; capacity clears on the next poll.
        return len(observations) > 1
    def sleep(seconds):
        sleeps.append(seconds)
        assert missing is None, 'missing completed artifacts must fail without another wait'
        assert len(sleeps) == 1, 'dispatcher did not finish after capacity became available'
        state = json.loads((root/'pipeline_state.json').read_text())
        assert state['rows'][row['id']]['status'] == 'RUNNING'
        assert 'periodic_error' not in state['rows'][row['id']]
    monkeypatch.setattr(runner.time, 'sleep', sleep)
    def complete(*a, **kw):
        finalizations.append(True)
        return 'EXPLORATORY_SCORED_PENDING_ANALYSIS'
    monkeypatch.setattr(runner, 'complete_row', complete)
    runner.main(matrix_factory=lambda: matrix, periodic_callback=observe)
    state = json.loads((root/'pipeline_state.json').read_text())
    if missing:
        assert state['status'] == 'FAILED'
        assert state['rows'][row['id']]['status'] == 'EVAL_FAILED'
        assert 'Training exited without planned snapshots/markers: [60]' in state['rows'][row['id']]['error']
        assert len(observations) == 1 and not sleeps and not finalizations
    else:
        assert state['status'] == 'AWAITING_ARTIFACT_ANALYSIS'
        assert state['rows'][row['id']]['status'] == 'EXPLORATORY_SCORED_PENDING_ANALYSIS'
        assert len(observations) == 2 and sleeps == [30] and finalizations == [True]
