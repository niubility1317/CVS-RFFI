import json
import os
import subprocess
import sys
import time
import traceback
import torch
from . import design as d
from .runtime import ROOT, frozen_context, action_metadata, save_checkpoint, load_checkpoint
from .models import UncertaintyRestorer
from .evaluation import VIEWS
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_residual_restore.runtime import load_checkpoint as load_residual
from experiments.pamf_restoration.data import IQSource, write_json, SCENES
from experiments.pamf_restoration.runner import array_tensor, read_json
from experiments.pamf_restoration.dispatcher import capacity, progress


def source_qualification(direct, uncertain, references):
    if any(row['status'] != 'ANALYZED' or row['role'] != 'source' or row['count'] != 1800 or not row['all_predictions_before_truth'] for row in (direct, uncertain, references)):
        raise ValueError('Completed fixed source scoring required')
    def mean(row, name):
        return sum(row['views'][scene][name]['waveform']['iq_nmse'] for scene in SCENES)/6
    def stat(row, scene, name, dimension='overall'):
        return {item['value']: item for item in row['records'] if item['view'] == scene and item['readout'] == name and item['dimension'] == dimension}
    direct_mean, mean_new, r03_mean = mean(direct, 'restored'), mean(uncertain, 'restored'), mean(references, 'r03_reference')
    low_new = stat(uncertain, 'practical_low_urban', 'restored')['all']
    low_old = stat(references, 'practical_low_urban', 'r03_reference')['all']
    clean_new = stat(uncertain, 'clean', 'restored')['all']['accuracy']
    clean_old = stat(references, 'clean', 'r03_reference')['all']['accuracy']
    new_tx = stat(uncertain, 'practical_low_urban', 'restored', 'TX')
    old_tx = stat(references, 'practical_low_urban', 'r03_reference', 'TX')
    if len(new_tx) != 6 or set(new_tx) != set(old_tx):
        raise ValueError('All six source TX groups required')
    worst_delta = min(new_tx[key]['accuracy']-old_tx[key]['accuracy'] for key in new_tx)
    clean_nmse = uncertain['views']['clean']['restored']['waveform']['iq_nmse']
    waveform_pass = mean_new <= .98*direct_mean and mean_new <= r03_mean and clean_nmse <= .02
    identity_pass = low_new['accuracy']-low_old['accuracy'] >= .5 and clean_old-clean_new <= .2 and worst_delta >= -1.
    healthy_control = direct['views']['clean']['restored']['waveform']['iq_nmse'] <= .02
    qualified = waveform_pass and identity_pass and healthy_control
    return dict(status='SOURCE_DECISION_FROZEN', qualified=bool(qualified), waveform_A=bool(waveform_pass),
        source_identity_B=bool(identity_pass), mean_direct=direct_mean, mean_uncertain=mean_new, mean_R03=r03_mean,
        uncertain_clean_NMSE=clean_nmse, lowurban_gain_vs_R03_pp=low_new['accuracy']-low_old['accuracy'],
        clean_drop_vs_R03_pp=clean_old-clean_new, worst_lowurban_TX_delta_pp=worst_delta,
        target_candidates=['direct_control', 'uncertain_map_prior'] if qualified else [],
        training_TX_F_access=False, waveform_A_TX_F_access=False, identity_B_uses_source_V_labels=True,
        target_reads_before_decision=0, target_feedback=False, thresholds=d.configuration()['source_thresholds'])


def source_smoke(args):
    action_metadata(args.parent_action)
    x = array_tensor(IQSource(args.source_cache, 'L_s').iq[:2], 'cpu')
    physics, _ = frozen_context(args, 'native_map_prior', 'cpu')
    coarse, _ = frozen_context(args, 'direct_control', 'cpu')
    reader = load_reader(args.reader_folder, args.source_cache, 'cpu')
    r03, payload = load_residual(args.parent_residual/'residual_deterministic/final.pt', 'residual_deterministic', 'cpu')
    if payload['epoch'] != 20 or payload['updates'] != 4940:
        raise ValueError('Original fixed R03 reference required')
    with torch.no_grad():
        if not torch.allclose(coarse(x), physics.coarse(x), atol=1e-6, rtol=1e-5):
            raise ValueError('Matched direct/physical coarse contexts differ')
        if not torch.isfinite(reader(r03(x))).all():
            raise ValueError('Actual trained R03/F50 source smoke failed')
    for kind in d.NAMES:
        torch.manual_seed(d.MODEL_SEED)
        frozen = coarse if kind == 'direct_control' else physics
        model = UncertaintyRestorer(kind, frozen, .01 if kind == 'uncertain_map_prior' else None).eval()
        with torch.no_grad():
            if not torch.isfinite(model(x)).all():
                raise ValueError('Actual source composite smoke failed')
    write_json(args.output/'smoke.json', dict(status='PASS', source_samples=2, target_reads=0, TX_labels_read=0,
        optimizer_updates=0, head_fit_updates=0, matched_coarse=True, actual_R02_R07_R03_F50_checkpoints=True))


def command(args, mode, kind):
    result = [sys.executable, '-u', '-m', 'experiments.pamf_uncertainty_restore.runner', mode, '--kind', kind]
    for key in ('source_cache', 'parent_coarse', 'parent_action', 'parent_residual', 'reader_folder', 'formal_reference', 'formal_f50', 'target_truth', 'output', 'expected_commit'):
        result.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
    return result


def run_queue(args, pending, mode, launches):
    import fcntl
    pending = list(pending); active = {}; failed = []; completed = []
    while pending or active:
        for name, (process, record) in list(active.items()):
            if process.poll() is None:
                continue
            required = args.output/'calibration.json' if mode == 'calibrate' else (
                args.output/'target_confirmation'/name/'complete.json' if mode == 'target' else args.output/name/'complete.json')
            if process.returncode or not required.is_file():
                failed.append(dict(kind=name, exit_code=process.returncode))
            else:
                completed.append(name)
            del active[name]
        if pending and not failed and len(active) < 3:
            with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = capacity(); reserved = {record['GPU'] for _, record in active.values()}
                cards = [gpu for gpu in d.GPUS if gpu not in reserved and state[gpu]['free_mb'] >= 12000 and len(state[gpu]['pids']) < 2]
                cards.sort(key=lambda gpu: (len(state[gpu]['pids']), -state[gpu]['free_mb'], list(d.GPUS).index(gpu)))
                if cards:
                    gpu = cards[0]; name = pending.pop(0); cmd = command(args, mode, name)
                    log = args.output/'logs'/(mode+'-'+name+'.log')
                    with log.open('x', encoding='utf-8') as stream:
                        process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), start_new_session=True)
                    record = dict(kind=name, mode=mode, pid=process.pid, GPU=gpu, cwd=str(ROOT), argv=cmd, log=str(log), started=time.time())
                    active[name] = (process, record); launches.append(record)
                    progress(args.output/'workers.json', dict(rows=launches))
                    deadline = time.monotonic()+45
                    while process.poll() is None and time.monotonic() < deadline:
                        if process.pid in capacity()[gpu]['pids']:
                            break
                        time.sleep(.5)
        progress(args.output/'queue_state.json', dict(mode=mode, active=[record for _, record in active.values()],
            pending=pending, completed=completed, failed=failed, time=time.time()))
        if failed and not active:
            break
        if pending or active:
            time.sleep(2)
    if failed:
        raise RuntimeError('Technical row failure;healthy rows preserved,unstarted rows held: '+str(failed))


def run(args):
    if args.output.exists():
        raise FileExistsError('Immutable uncertainty output exists;reconcile')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU controller required')
    args.output.mkdir(); (args.output/'logs').mkdir()
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv, commit=args.expected_commit,
        owner='codex/root/uncertainty-restoration', started=time.time()))
    write_json(args.output/'resolved_config.json', d.configuration())
    try:
        source_smoke(args)
        launches = []
        run_queue(args, ['calibration'], 'calibrate', launches)
        run_queue(args, ['reference_controls'], 'reference', launches)
        run_queue(args, d.NAMES, 'train', launches)
        decision = source_qualification(read_json(args.output/'direct_control/summary.json'),
            read_json(args.output/'uncertain_map_prior/summary.json'), read_json(args.output/'reference_controls/summary.json'))
        write_json(args.output/'source_decision.json', decision)
        run_queue(args, decision['target_candidates'], 'target', launches)
        write_json(args.output/'complete.json', dict(status='ANALYZED', source_qualified=decision['qualified'],
            source_rows=['reference_controls', *d.NAMES], target_rows=decision['target_candidates'], training_TX_F_access=False,
            source_only=not decision['qualified'], target_feedback=False, goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
            traceback=traceback.format_exc(), outputs_preserved=True, automatic_retry=False))
        raise
