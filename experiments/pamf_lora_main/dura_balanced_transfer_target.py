"""Wait for the registered balanced-transfer LoRa fit and run its blind target pilot."""

import argparse
from json import JSONDecodeError
from pathlib import Path, PurePosixPath
import sys
from time import sleep

if __name__ == '__main__' and __spec__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_balanced_transfer_source as training, weak_guard_target as replay
from experiments.pamf_lora_main.dura_dual_aperture_evaluate import PER_RECEIVER, ROUTES, SEED, SEEDS


RUN = '20261125-phase1-lora-dura-balanced-transfer-pilot-s2026092702-r01'
ORIGIN = training.RUN
ORIGIN_CODE = '35530da3ffe3d5d1634a30810dc600d7c809e08b'
MODE = training.MODE
MODES = ('direct', MODE)
EVENTS = 'LORA_BALANCED_TRANSFER_PILOT'


def completion():
    return replay.completion(origin=ORIGIN, origin_code=ORIGIN_CODE, modes=MODES,
                             seeds=SEEDS, per_receiver=PER_RECEIVER)


def expected_files():
    return replay.expected_files(MODES, seeds=SEEDS)


def source_release(rt):
    project = Path(rt['project'])
    release = project/'releases'/f'{ORIGIN}-release'
    parent = read(release/'runtime.json')
    if (parent['run_id'], parent['code_commit'], Path(parent['run_root']), Path(parent['source_root'])) != (
            ORIGIN, ORIGIN_CODE, project/'runs'/ORIGIN, Path(rt['source_root'])):
        raise ValueError('Expected the registered balanced-transfer training and source IQ contract')
    return parent, release


def training_state(parent):
    try:
        return read(parent/'state.json')['status']
    except (FileNotFoundError, JSONDecodeError):
        return None


def await_source(rt, root, state):
    parent, release = source_release(rt)
    source_run = Path(parent['run_root'])
    write(state, dict(status='WAITING_FOR_FROZEN_TRAINING'))
    process = Path(f"/proc/{rt['training_dispatcher']}")
    while True:
        status = training_state(source_run)
        if status == 'SOURCE_SCORED':
            break
        if status == 'FAILED':
            raise RuntimeError('The registered balanced-transfer source run failed')
        try:
            owned = ((process/'cwd').resolve(strict=True) == release
                     and b'dura_balanced_transfer_source' in (process/'cmdline').read_bytes())
        except FileNotFoundError:
            owned = False
        if not owned and training_state(source_run) != 'SOURCE_SCORED':
            raise RuntimeError('Source dispatcher exited without a closed source matrix')
        sleep(20)
    preserve_source_reference(source_run, root)


def preserve_source_reference(source_run, root):
    from experiments.pamf_lora_main.local_wiener_source import read_frozen

    matrices = {route: read_frozen(source_run, route, training.MODES) for route in ROUTES}
    if matrices['full'][0] != matrices['residual'][0]:
        raise ValueError('Source predictions must cover the same physical packets')
    pairs = {}
    candidate, comparator = (training.MODES.index(name) for name in (MODE, 'expanded_half'))
    for route in ROUTES:
        scores = read(source_run/f'{route}-source-scores.json')
        if (scores['status'] != 'SOURCE_SCORED' or scores['route'] != route
                or scores['target_reads'] or not scores['truth_after_predictions']
                or scores['modes'] != list(training.MODES) or scores['n'] != len(matrices[route][0])):
            raise ValueError('Expected independent same-matrix source scoring')
        six = scores['six']
        clean = scores['metrics']['clean']['correct']
        pairs[route] = dict(six_net=six[candidate]-six[comparator], clean_net=clean[candidate]-clean[comparator])
        write(root/f'{route}-predictions.json', read(source_run/f'{route}-predictions.json'))
        write(root/f'{route}-source-scores.json', scores)
    write(root/'source-promotion.json', dict(reference_only=True, pairs=pairs, target_reads=0))


def bind(rt, device):
    from experiments.pamf_lora_main.runner import activate

    parent, _ = source_release(rt)
    if training_state(Path(parent['run_root'])) != 'SOURCE_SCORED':
        raise ValueError('The registered training and source matrix must close before target inference')
    activate(rt)
    return training.bind_reader(parent, device)


def publish(repo, output, context, ssh_config, training_receipt):
    from experiments.pamf_lora_main.publish import release, release_commit

    cfg, receipt = read(context), read(training_receipt)
    project = PurePosixPath(cfg['project'])
    if (PurePosixPath(receipt['run_root']), receipt['code_commit'], receipt['archive_transfer']) != (
            project/'runs'/ORIGIN, ORIGIN_CODE, 'VERIFIED'):
        raise ValueError('Expected the verified balanced-transfer source receipt')
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo), training_dispatcher=receipt['pid'],
              run_root=str(project/'runs'/RUN), release_root=str(project/'releases'/f'{RUN}-release'),
              source_root=str(project/'releases'/f'{replay.DATA}-release'/'data'/'source'),
              external_releases=[str(project/'releases'/f'{ORIGIN}-release')])
    release(repo, output, rt, ssh_config)


def main():
    from functools import partial

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    publisher = actions.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config', 'training-receipt'):
        publisher.add_argument('--'+flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate'):
        entry = actions.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('row')
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
        return
    rt = read(args['runtime'])
    if rt['run_id'] != RUN:
        raise ValueError('Expected the immutable balanced-transfer pilot root')
    if action == 'dispatch':
        replay.dispatch(rt, args['runtime'], entrypoint=Path(__file__).resolve(), modes=MODES, complete=completion,
                        events=EVENTS, seeds=SEEDS, before_smoke=partial(await_source, rt))
    else:
        replay.run_worker(rt, action, args['row'], bind=bind, modes=MODES, events=EVENTS, seeds=SEEDS,
                          per_receiver=PER_RECEIVER, time_restoration=True)


if __name__ == '__main__':
    main()
