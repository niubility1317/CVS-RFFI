"""Continue the resource-interrupted affine mixture run from its fixed E22 boundary."""
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import mixture_affine_trial as original
from . import multiview_trial as trial
from .cost_probe import write
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r02'
MODULE = 'experiments.pamf_phase3_restoration.mixture_affine_resume_trial'
ARM = original.ARM
PARENT_RUN = original.RUN
PARENT_COMMIT = 'b231eb29aae1ff0d85049960dba02a64b404b4a6'
PARENT_EPOCH = 22
PARENT_CONFIG = original.configuration(ARM)
CAP = 5


def configuration(arm):
    if arm != ARM: raise ValueError('Only original affine arm allowed')
    cfg = dict(PARENT_CONFIG)
    cfg.update(run_id=RUN, technical_recovery_parent=PARENT_RUN,
               technical_recovery_commit=PARENT_COMMIT, technical_recovery_epoch=PARENT_EPOCH,
               resource_policy='GPU6; at most three existing compute PIDs at admission; monitor total cap5.',
               scientific_change='Original affine mixture objective and source schedule, technical E22 continuation only.')
    return cfg


def factory(arm):
    configuration(arm)
    return original.factory(arm)


def parent_training(root):
    parent = root / 'runs' / PARENT_RUN
    stop = read(parent / 'resource_stop.json')
    terminal = read(parent / ('terminal-' + ARM + '-train.json'))
    dispatch = read(parent / 'resource_controller.json')
    if (stop['stage'], stop['reason'], stop['cap'], stop['only_own_child_signaled'],
        stop['other_owner_processes_signaled'], terminal['pid'], terminal['exit_code'],
        terminal['scientific_code_commit']) != ('train', 'TOTAL_COMPUTE_PID_CAP_EXCEEDED', 3,
            True, False, stop['pid'], -15, PARENT_COMMIT):
        raise ValueError('Original resource-protected stop required')
    if any(parent.glob('submitted-' + ARM + '-cross*.json')):
        raise ValueError('Parent must not have started target predictions')
    for pid in (stop['pid'], dispatch['pid'], read(parent / 'launch.json')['pid']):
        path = Path('/proc') / str(pid) / 'cmdline'
        if path.exists():
            command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if (original.MODULE in command or 'mixture_affine_trial.py' in command
                    or 'mixture_affine_capacity_dispatch' in command):
                raise ValueError('Parent owner still live; do not duplicate')
    folder = parent / 'rows' / ARM / 'training'
    smoke = read(folder / 'smoke.json')
    if smoke['status'] != 'PASS' or any(smoke[k] for k in ('TX_labels_read', 'V_reads', 'OE_reads', 'target_inputs')):
        raise ValueError('Original source-only smoke required')
    if not (folder / 'recovery' / f'E{PARENT_EPOCH:02d}.pt').is_file() or (folder / 'complete.json').exists():
        raise ValueError('Original incomplete E22 technical checkpoint required')
    return folder


def train(args, out):
    parent = parent_training(args.root)
    recovery = dict(run=PARENT_RUN, training=parent, epoch=PARENT_EPOCH,
                    commit=PARENT_COMMIT, config=PARENT_CONFIG)
    training.train(args, out, build=factory, config_for=configuration,
        bind_prior=original.parent.bind_prior, training_readout=original.source_readout,
        smoke_gradient_check=original.parent.gradient_smoke, technical_resume=recovery,
        additional_prior_keys=('mixture_within', 'mixture_offsets', 'mixture_weights'))


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(6,), maximum_processes=4)
            if item['free_memory_MB'] >= 12288]


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA-empty controller required')
    parent_training(args.root)
    out = args.root / 'runs' / RUN
    out.mkdir(); (out / 'logs').mkdir(); row = out / 'rows' / ARM
    row.mkdir(parents=True); (row / 'prediction').mkdir()
    write(out / 'launch.json', dict(pid=os.getpid(), command=sys.argv, CWD=os.getcwd(), code_commit=args.commit))
    for stage in ('train', 'cross_rx', 'cross_both', 'score'):
        admission = None
        if stage != 'score':
            while not admission:
                choices = admitted_gpus()
                if choices: admission = choices[0]
                else:
                    print('WAITING_FOR_ADMITTED_GPU_WITH_HEADROOM', stage, flush=True)
                    time.sleep(30)
        command = trial.command(args, 'score' if stage == 'score' else ('train' if stage == 'train' else 'predict'), ARM,
                                stage if stage in ('cross_rx', 'cross_both') else None)
        with (out / 'logs' / (ARM + '-' + stage + '.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=Path.cwd(), env=dict(os.environ,
                CUDA_VISIBLE_DEVICES='' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(out / ('submitted-' + ARM + '-' + stage + '.json'),
              dict(pid=child.pid, command=command, CWD=os.getcwd(), admission=admission))
        failures = 0; reason = None; observed = None
        while child.poll() is None:
            if admission is not None:
                try:
                    observed = resource.compute_pids(admission['GPU'], admission['CUDA_ID']); failures = 0
                    if child.pid in observed and len(observed) > CAP: reason = 'TOTAL_COMPUTE_PID_CAP_EXCEEDED'
                except (RuntimeError, ValueError) as error:
                    failures += 1
                    print('GPU_COUNT_OBSERVATION_FAILED', str(error), flush=True)
                    if failures >= 3: reason = 'GPU_COUNT_UNVERIFIABLE'
                if reason:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    write(out / 'resource_stop.json', dict(stage=stage, pid=child.pid, GPU=admission['GPU'],
                        reason=reason, observed_compute_PIDs=sorted(observed) if observed is not None else None,
                        cap=CAP, only_own_child_signaled=True, other_owner_processes_signaled=False))
                    break
            time.sleep(2)
        code = child.wait()
        write(out / ('terminal-' + ARM + '-' + stage + '.json'),
              dict(pid=child.pid, GPU=None if admission is None else admission['GPU'], exit_code=code))
        if code or reason: raise RuntimeError('Preserved stopped or failed stage: ' + stage + ' ' + str(reason))
    if read(row / 'results.json')['status'] != 'ANALYZED':
        raise ValueError('Missing complete original Main result')
    write(out / 'complete.json', dict(status='ARTIFACTS_COMPLETE', all_rows_trained30_and_main_seven_scored=True,
                                     default_DURA_changed=False, goal_complete=False))



def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train, trial.controller = configuration, factory, train, controller


if __name__ == '__main__':
    install()
    trial.main()
