"""Frozen native-noise MAP confirmation selected only from completed source data."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from experiments.pamf_uncertainty_restore.evaluation import predict, score, VIEWS
from experiments.pamf_uncertainty_restore.runtime import load_checkpoint
from experiments.pamf_fcrossrx_readout.frozen import load_reader, REFERENCE_RUN
from experiments.pamf_restoration.data import IQSource, write_json
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_restoration.runner import array_tensor, code_commit, read_json

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261019-phase1-restoration-formal-nativeMAP-F50-s2026092701-r11'
PARENT = '20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'
PARENT_COMMIT = 'b57eca9946b2a912a88870e24826fe3e2f70985d'
KIND = 'native_map_prior'
GPUS = (0, 1, 5, 7)
PATH_ARGS = ('source_cache', 'parent_run', 'reader_folder', 'formal_reference', 'formal_f50', 'target_truth', 'output')


def configuration():
    return dict(schema='native_MAP_formal_v1', run=RUN, candidate=KIND, parent_run=PARENT,
        parent_commit=PARENT_COMMIT, count=168000, views=list(VIEWS), F_run=REFERENCE_RUN, F_seed=2026092701,
        original_default_head=True, new_training_updates=0, head_fit_updates=0, state_updates=0,
        channel_generation_calls=0, source_selection='New source-only R11 selection;R10 primary remains rejected',
        thresholds=dict(mean6_NMSE_vs_R03_max_ratio=1., clean_NMSE_max=.02, lowurban_gain_vs_R03_min_pp=.5,
            clean_drop_vs_R03_max_pp=.2, worst_lowurban_TX_delta_vs_R03_min_pp=-1.),
        source_waveform_selection_F_TX_access=False, source_identity_selection_uses_V_labels=True,
        training_TX_F_access=False, inherited_component_epochs=60, R03_component_epochs=40,
        comparison_scope='Frozen pipeline comparison;not equal-cost/equal-recipe direct-versus-physics ablation',
        user_performance_goal=dict(lowurban_accuracy_min=70., other_five_max_gap_vs_original_clean_pp=5.,
            clean_reference='Original received clean through same frozen F/seed;restoredclean damage separatelyreported'),
        GPU_pool=list(GPUS), max_own_GPU_workers=1, capacity_lock='receiver_residual_capacity4.lock',
        all_owner_cap=2, min_free_MB=12000, feature_reference_cap=512,
        stop_rule='Wrong source/parent/prediction,collision,nonfinite:preserve outputs,stop ownrow,no automaticretry',
        target_feedback=False, truth_last=True)


def qualify_source(native, references):
    for summary, kind in ((native, KIND), (references, 'reference_controls')):
        expected = dict(status='ANALYZED', kind=kind, role='source', count=1800,
            F_run=REFERENCE_RUN, F_seed=2026092701, original_default_head=True,
            all_predictions_before_truth=True, training_TX_labels_read=0, training_F_access=False,
            head_fit_updates=0, target_access=False)
        if any(summary.get(key) != value for key, value in expected.items()) or list(summary['views']) != list(VIEWS):
            raise ValueError('Original complete source-only native/reference scoring required')
    def metric(summary, view, readout):
        value = summary['views'][view][readout]['waveform']['iq_nmse']
        if not math.isfinite(value) or value < 0:
            raise ValueError('Finite source waveform metric required')
        return value
    def records(summary, view, name, dimension):
        rows = [row for row in summary['records'] if row['view'] == view and row['readout'] == name and row['dimension'] == dimension]
        if len({row['value'] for row in rows}) != len(rows) or any(not math.isfinite(row['accuracy']) or not 0 <= row['accuracy'] <= 100 for row in rows):
            raise ValueError('Unique finite source identity groups required')
        return {row['value']: row['accuracy'] for row in rows}
    mean_native = sum(metric(native, view, 'restored') for view in VIEWS[1:])/6
    mean_old = sum(metric(references, view, 'r03_reference') for view in VIEWS[1:])/6
    clean_nmse = metric(native, 'clean', 'restored')
    low_native = records(native, 'practical_low_urban', 'restored', 'overall')['all']
    low_old = records(references, 'practical_low_urban', 'r03_reference', 'overall')['all']
    clean_drop = records(references, 'clean', 'r03_reference', 'overall')['all']-records(native, 'clean', 'restored', 'overall')['all']
    tx_new = records(native, 'practical_low_urban', 'restored', 'TX')
    tx_old = records(references, 'practical_low_urban', 'r03_reference', 'TX')
    if set(tx_new) != set(tx_old) or set(tx_new) != {str(i) for i in range(6)}:
        raise ValueError('All original six TX source groups required')
    worst_delta = min(tx_new[key]-tx_old[key] for key in tx_new)
    waveform_pass = mean_native <= mean_old and clean_nmse <= .02
    identity_pass = low_native-low_old >= .5 and clean_drop <= .2 and worst_delta >= -1.
    qualified = bool(waveform_pass and identity_pass)
    return dict(status='SOURCE_DECISION_FROZEN', selection_stage=RUN, qualified=qualified,
        waveform_A=bool(waveform_pass), source_identity_B=bool(identity_pass), mean_native=mean_native,
        mean_R03=mean_old, native_clean_NMSE=clean_nmse, lowurban_gain_vs_R03_pp=low_native-low_old,
        clean_drop_vs_R03_pp=clean_drop, worst_lowurban_TX_delta_pp=worst_delta,
        target_candidates=[KIND] if qualified else [], previous_R10_primary_changed=False,
        training_TX_F_access=False, waveform_A_TX_F_access=False, identity_B_uses_source_V_labels=True,
        target_reads_before_decision=0, target_feedback=False, thresholds=configuration()['thresholds'])


def parent_contract(parent):
    if parent.name != PARENT or read_json(parent/'launch.json')['commit'] != PARENT_COMMIT:
        raise ValueError('Original R10 frozen native source parent required')
    complete = read_json(parent/'complete.json')
    if complete['status'] != 'ANALYZED' or not complete['source_only'] or complete['target_rows'] or complete['training_TX_F_access']:
        raise ValueError('Source-only completed parent with zero target access required')
    row = parent/KIND
    expected = dict(status='SOURCE_TRAINED', kind=KIND, epochs=20, updates=4940,
        L_exposures=126000, U_exposures=1134000, TX_labels_read=0, F_outputs_read=0, target_reads=0, frozen_context_updates=0)
    trained = read_json(row/'training_complete.json'); done = read_json(row/'complete.json')
    if any(trained.get(key) != value for key, value in expected.items()) or not (row/'final.pt').is_file():
        raise ValueError('Original full-LU zero-TX/F fixed final20/4940 required')
    if (done['status'] != 'ANALYZED' or done['kind'] != KIND or done['role'] != 'source'
            or not done['all_predictions_before_truth'] or done['training_TX_F_access']):
        raise ValueError('Completed native source row required')
    return qualify_source(read_json(row/'summary.json'), read_json(parent/'reference_controls/summary.json'))


def assess_goal(summary, *, expected_kind=KIND):
    if (summary['status'] != 'ANALYZED' or summary['role'] != 'target' or summary['kind'] != expected_kind or summary['count'] != 168000
            or summary['F_run'] != REFERENCE_RUN or summary['F_seed'] != 2026092701 or not summary['original_default_head']
            or not summary['all_predictions_before_truth']):
        raise ValueError('Completed same-F target scoring required for performance assessment')
    rows = [row for row in summary['records'] if row['dimension'] == 'overall' and row['readout'] == 'restored']
    if len(rows) != 7 or {row['view'] for row in rows} != set(VIEWS):
        raise ValueError('All seven individual views required')
    values = {row['view']: row for row in rows}
    original_clean = values['clean']['baseline_accuracy']
    if not all(math.isfinite(value) and 0 <= value <= 100 for value in [original_clean, *[row['accuracy'] for row in rows]]):
        raise ValueError('Finite target accuracy required')
    other = {view: dict(accuracy=values[view]['accuracy'], gap_vs_original_clean_pp=original_clean-values[view]['accuracy'],
        passed=values[view]['accuracy'] >= original_clean-5.) for view in VIEWS[1:] if view != 'practical_low_urban'}
    lowurban_pass = values['practical_low_urban']['accuracy'] >= 70.
    return dict(status='PERFORMANCE_ASSESSED', original_clean_accuracy=original_clean,
        restored_clean_accuracy=values['clean']['accuracy'], clean_drop_pp=original_clean-values['clean']['accuracy'],
        lowurban_accuracy=values['practical_low_urban']['accuracy'], lowurban_pass=lowurban_pass, other_five=other,
        performance_targets_met=bool(lowurban_pass and all(item['passed'] for item in other.values())),
        goal_complete=False, inference_or_selection_use=False, user_thresholds=configuration()['user_performance_goal'])


def prepare_source(args, decision=None):
    decision = parent_contract(args.parent_run) if decision is None else decision
    if not decision['qualified']:
        raise ValueError('Native source waveform and identity requirements failed;no target access')
    if (args.output/'source_decision.json').exists():
        raise FileExistsError('Immutable source decision already exists')
    row = args.output/KIND; row.mkdir()
    with (args.parent_run/KIND/'final.pt').open('rb') as source, (row/'final.pt').open('xb') as destination:
        shutil.copyfileobj(source, destination)
    write_json(args.output/'source_decision.json', decision)
    return decision


@torch.no_grad()
def smoke(args):
    decision = parent_contract(args.parent_run)
    if not decision['qualified']:
        raise ValueError('Native source qualification failed')
    model, payload = load_checkpoint(args.parent_run/KIND/'final.pt', 'cpu')
    if payload['kind'] != KIND or payload['epoch'] != 20 or payload['updates'] != 4940:
        raise ValueError('Original frozen native checkpoint required')
    reader = load_reader(args.reader_folder, args.source_cache, 'cpu')
    x = array_tensor(IQSource(args.source_cache, 'L_s').iq[:2], 'cpu')
    restored = model(x); logits = reader(restored)
    if restored.shape != x.shape or logits.shape != (2, 6) or not torch.isfinite(restored).all() or not torch.isfinite(logits).all():
        raise ValueError('Actual source restorer/originalF50 smoke failed')
    write_json(args.output/'smoke.json', dict(status='PASS', source_samples=2, actual_R10_native_F50=True,
        target_reads=0, TX_labels_read=0, optimizer_updates=0, head_fit_updates=0))
    return decision


def command(args, mode):
    result = [sys.executable, '-u', '-m', 'experiments.pamf_native_formal.runner', mode]
    for key in (*PATH_ARGS, 'expected_commit'):
        result += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    return result


def run(args):
    import fcntl
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        raise ValueError('CPU controller and absent immutable output required')
    args.output.mkdir()
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv,
        commit=args.expected_commit, owner='codex/root/native-MAP-formal', started=time.time()))
    write_json(args.output/'resolved_config.json', configuration())
    try:
        prepare_source(args, smoke(args))
        while True:
            with (args.output.parent/configuration()['capacity_lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = capacity()
                available = [gpu for gpu in GPUS if state[gpu]['free_mb'] >= 12000 and len(state[gpu]['pids']) < 2]
                available.sort(key=lambda gpu: (len(state[gpu]['pids']), -state[gpu]['free_mb'], list(GPUS).index(gpu)))
                if available:
                    gpu = available[0]; cmd = command(args, 'predict')
                    log = args.output/'worker.log'
                    with log.open('x', encoding='utf-8') as stream:
                        process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), start_new_session=True)
                    write_json(args.output/'worker_launch.json', dict(pid=process.pid, GPU=gpu, argv=cmd, cwd=str(ROOT), log=str(log), started=time.time()))
                    deadline = time.monotonic()+45
                    while process.poll() is None and time.monotonic() < deadline:
                        if process.pid in capacity()[gpu]['pids']:
                            break
                        time.sleep(.5)
                    break
            time.sleep(2)
        while process.poll() is None:
            time.sleep(2)
        if process.returncode or not (args.output/'complete.json').is_file():
            raise RuntimeError('Native formal worker failed;outputs preserved;no automatic retry')
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error), traceback=traceback.format_exc(),
            outputs_preserved=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description='Fixed source-selected native MAP and original F50 full-seven-scene confirmation')
    parser.add_argument('mode', choices=('run', 'predict', 'score'))
    for key in PATH_ARGS:
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    for key in PATH_ARGS:
        setattr(args, key, getattr(args, key).resolve())
    args.kind = KIND
    if args.output.name != RUN or code_commit() != args.expected_commit:
        raise ValueError('Pinned native formal run/code differs')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.mode == 'run':
        run(args); return
    if not parent_contract(args.parent_run)['qualified']:
        raise ValueError('Source qualification required')
    if args.mode == 'score':
        score(args, 'target')
        summary = read_json(args.output/'target_confirmation'/KIND/'summary.json')
        write_json(args.output/'goal_assessment.json', assess_goal(summary))
        return
    write_json(args.output/'gpu_ready.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv, physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    predict(args, 'target')
    subprocess.run(command(args, 'score'), cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    write_json(args.output/'complete.json', dict(status='ANALYZED', count=168000, candidate=KIND,
        parent_run=PARENT, training_updates=0, adaptation_updates=0, head_fit_updates=0,
        truth_last=True, source_selected_before_target=True, target_feedback=False, goal_complete=False))


if __name__ == '__main__':
    main()
