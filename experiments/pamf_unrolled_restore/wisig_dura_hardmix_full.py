"""Full-channel-only WiSig Main evaluation of the existing hard-mix weights."""

import argparse
import os
from pathlib import Path, PurePosixPath
import signal
from time import monotonic, sleep

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main import publish as transport
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_repl_view_shuffle.runner import activate
from . import wisig_dura_hardmix as original
from .wisig_dura_hardmix_data import load
from .wisig_dura_hardmix_gpu_handoff import cuda_device, process


RUN = '20261126-phase1-wisig-dura-hardmix-full-eval-r01'
CONTINUATIONS = ('balanced', 'hard')


def owned(pid, release, expected):
    row = process(pid)
    if row is None or (row['uid'], row['cwd'], row['argv']) != (os.getuid(), release, expected):
        raise ValueError('Expected the exact run-owned process before changing evaluation route')
    return row


def children(pid):
    path = Path(f'/proc/{pid}/task/{pid}/children')
    return tuple(int(child) for child in path.read_text().split() if process(int(child)))


def stop_for_handoff(pid):
    os.kill(pid, signal.SIGSTOP)
    end = monotonic() + 3
    while process(pid)['state'] != 'T':
        if monotonic() > end:
            raise RuntimeError('Run-owned scheduler did not suspend')
        sleep(.02)


def transfer_evaluation(rt):
    root, release = map(Path, (rt['training_root'], rt['training_release']))
    scratch_marker = root / 'models/scratch/complete.json'
    previous = read(root / 'gpu_memory_handoff.json')
    controller_pid = previous['controller_pid']
    helper = str(release.parent / f'{original.RUN}-gpu-memory-7c7b83278.py')
    base = [rt['python'], '-u', '-m', original.__spec__.name]
    runtime = str(release / 'runtime.json')
    if process(controller_pid) is None and scratch_marker.is_file():
        return None
    owned(controller_pid, release, [rt['python'], '-u', helper, runtime])
    paused = []
    try:
        paused.append(controller_pid)
        stop_for_handoff(controller_pid)
        wrapper_pid, = children(controller_pid)
        owned(wrapper_pid, release, [*base, 'arm', runtime, 'scratch'])
        paused.append(wrapper_pid)
        stop_for_handoff(wrapper_pid)
        fit_pid, = children(wrapper_pid)
        fit_cmd = [*base, 'fit', str(root / 'source_runtime.json'), 'scratch']
        owned(fit_pid, release, fit_cmd)
        gpu = cuda_device(fit_pid)
        if any((root / 'logs' / f'predict-scratch-{split}.log').exists() for split in original.CASES):
            raise ValueError('Scratch prediction has already started')
        # Stop both evaluation schedulers. The source fit is a distinct live process.
        for pid in paused:
            os.kill(pid, signal.SIGTERM)
        write(root / 'state.json', dict(status='EVALUATION_ROUTE_SUPERSEDED', route='full',
              evaluation_output=rt['run_root'], scratch_fit_continues=True), exclusive=False)
        proof = dict(pid=fit_pid, gpu=gpu, cmd=fit_cmd)
        write(Path(rt['run_root']) / 'scratch_training_adopted.json', proof)
        return proof
    finally:
        for pid in paused:
            if process(pid) is not None:
                os.kill(pid, signal.SIGCONT)


def source_smoke(rt):
    import torch

    original.setup_torch()
    activate(rt)
    training = Path(rt['training_root'])
    iq = tensor(load(training / 'source/01/hard.npy')[:4], 'cuda:0')
    _, heads = original.evaluation.readers(rt)
    for arm in CONTINUATIONS:
        engine, _ = original.frozen_arm(training / arm, arm)
        with torch.no_grad():
            restored = engine(iq)[0]
            original.evaluation.classify(heads, restored)
    print('FULL_EVAL_NO_QUERY_SMOKE_PASS', flush=True)


def run_matrix(rt, arms):
    jobs = [('predict', f'{arm}:{split}') for arm in arms for split in original.CASES]
    schedule(rt, jobs, original.launch)


def dispatch(rt):
    root = Path(rt['run_root'])
    training = Path(rt['training_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    state = root / 'state.json'
    try:
        fit = transfer_evaluation(rt)
        if fit:
            first = dict(rt, allowed_gpus=[gpu for gpu in rt['allowed_gpus'] if gpu != fit['gpu']])
        else:
            first = rt
        write(state, dict(status='SOURCE_SMOKE', route='full'))
        schedule(first, [('check', 'full')], original.launch)
        write(state, dict(status='FULL_CONTINUATION_PREDICTING'), exclusive=False)
        run_matrix(first, CONTINUATIONS)
        original.complete(original.launch(rt, 'score', 'continuations'), 'full continuation scoring')
        write(state, dict(status='CONTINUATIONS_SCORED_WAITING_SCRATCH'), exclusive=False)
        marker = training / 'models/scratch/complete.json'
        while not marker.is_file():
            active = process(fit['pid']) if fit else None
            if not active or active['argv'] != fit['cmd']:
                if marker.is_file():
                    break
                raise RuntimeError('Original scratch fit ended without a frozen checkpoint')
            sleep(20)
        write(state, dict(status='FULL_SCRATCH_PREDICTING'), exclusive=False)
        run_matrix(rt, ('scratch',))
        original.complete(original.launch(rt, 'score', 'complete'), 'complete full scoring')
        write(state, dict(status='ANALYZED', route='full'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    paths = dict(training_root=str(runs / original.RUN),
                 training_release=str(releases / f'{original.RUN}-release'),
                 parent_root=str(runs / '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'),
                 baseline_root=str(runs / original.evaluation.BASELINE),
                 recognizer_run=str(runs / original.evaluation.F_RUN),
                 decision_replay=str(runs / '20261125-phase1-wisig-dura-force-main-m2-r01/predictions'))
    rt.update(paths, run_id=RUN, runner=__spec__.name, code_commit=transport.release_commit(repo),
              run_root=str(runs / RUN), release_root=str(releases / f'{RUN}-release'),
              external_releases=list(paths.values()), allowed_gpus=list(range(8)), max_workers=2,
              max_compute_processes=None)
    transport.release(repo, output, rt, ssh_config, archive_extras=(
        'tools/wisig_paired_main_uncertainty.py', 'tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='operation', required=True)
    release = sub.add_parser('publish')
    for item in ('repo', 'output', 'context', 'ssh-config'):
        release.add_argument(f'--{item}', type=Path, required=True)
    for operation in ('dispatch', 'check', 'predict', 'score'):
        command = sub.add_parser(operation)
        command.add_argument('runtime', type=Path)
        command.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.operation == 'publish':
        return publish(args.repo, args.output, args.context, args.ssh_config)
    rt = read(args.runtime)
    if args.operation == 'dispatch':
        dispatch(rt)
    elif args.operation == 'check':
        source_smoke(rt)
    elif args.operation == 'predict':
        original.predict(rt, args.name, route='full')
    elif args.name == 'continuations':
        original.score(rt, route='full', arms=CONTINUATIONS, output='continuation_scores.json')
    elif args.name == 'complete':
        original.score(rt, route='full')
    else:
        raise ValueError('Expected the registered full scoring scope')


if __name__ == '__main__':
    main()
