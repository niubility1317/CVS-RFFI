"""One-owner N607 release for source-only training and paired screening."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import sleep

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import LORA_ACTION, LORA_GAIN, WISIG_CACHE, WISIG_PACKET, WISIG_PARENT


RUN = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
ROWS = ('wisig', 'wisig_real_gain')


def free_gpus():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    values = [tuple(int(x.strip()) for x in line.split(',')) for line in lines]
    return [gpu for gpu, free in sorted(values, key=lambda x: -x[1]) if free >= 8192]


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    reusable_lora = bool(rt.get('lora_frozen_root')) and 'lora' not in ROWS
    all_roles = (*ROWS, 'lora') if reusable_lora else ROWS
    for row in all_roles:
        (root / row).mkdir()
    state = {'status': 'STARTING', 'workers': {}, 'screens': {}, 'failures': {}}

    def save():
        write(root / 'state.json', state, exclusive=False)

    def start(module, action, row, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        args = [sys.executable, '-u', '-m', module]
        if action == 'fit':
            args += [str(runtime), row, str(root / row)]
        else:
            args += [action, str(runtime), row]
        with (root / f'{row}-{action}.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print('FUSION_WORKER', action, row, proc.pid, gpu, flush=True)
        return proc

    try:
        gpus = free_gpus()
        if len(gpus) < len(all_roles):
            raise RuntimeError('Too few distinct GPUs have 8 GiB free at preflight')
        assigned = dict(zip(all_roles, gpus[:len(all_roles)], strict=True))
        jobs = {}
        for row in ROWS:
            try:
                jobs[row] = start('experiments.pamf_lora_main.dura_real_gain_fit', 'fit', row, assigned[row])
            except OSError as error:
                state['failures'][row] = f'fit start failed: {error}'
        state.update(status='TRAINING', gpu=assigned, workers={k: v.pid for k, v in jobs.items()})
        save()
        pending = {dataset: needs for dataset, needs in
                   (('lora', {'lora'}), ('wisig', {'wisig', 'wisig_real_gain'}))
                   if needs.issubset(ROWS)}
        predictions = {}
        if reusable_lora:
            try:
                predictions['lora'] = start('experiments.pamf_lora_main.dura_real_gain_screen',
                                            'predict', 'lora', assigned['lora'])
            except OSError as error:
                state['failures']['lora_screen'] = f'prediction start failed: {error}'
        while jobs or pending or predictions:
            for row, proc in list(jobs.items()):
                code = proc.poll()
                if code is None:
                    continue
                del jobs[row]
                if code:
                    state['failures'][row] = f'fit failed with exit code {code}'
            for dataset, needs in list(pending.items()):
                if needs.intersection(jobs):
                    continue
                del pending[dataset]
                if needs.intersection(state['failures']):
                    state['failures'][dataset + '_screen'] = 'Required fit failed; no screen launched'
                    continue
                try:
                    predictions[dataset] = start('experiments.pamf_lora_main.dura_real_gain_screen',
                                                 'predict', dataset, assigned[dataset])
                except OSError as error:
                    state['failures'][dataset + '_screen'] = f'prediction start failed: {error}'
            for dataset, proc in list(predictions.items()):
                code = proc.poll()
                if code is None:
                    continue
                del predictions[dataset]
                if code:
                    state['failures'][dataset + '_screen'] = f'prediction failed with exit code {code}'
                    continue
                try:
                    scorer = start('experiments.pamf_lora_main.dura_real_gain_screen', 'score', dataset, None)
                    if scorer.wait():
                        state['failures'][dataset + '_score'] = 'Independent source scorer failed'
                    else:
                        state['screens'][dataset] = read(root / dataset / 'screen-score.json')['promote']
                except (OSError, ValueError, KeyError) as error:
                    state['failures'][dataset + '_score'] = f'Independent scorer error: {error}'
            state.update(status='RUNNING', workers={k: v.pid for k, v in jobs.items()},
                         predictions={k: v.pid for k, v in predictions.items()})
            save()
            if jobs or pending or predictions:
                sleep(3)
        state.update(status='PARTIAL' if state['failures'] else 'SOURCE_SCREEN_SCORED', workers={}, predictions={})
        save()
    except Exception as error:
        state.update(status='FAILED', error=str(error))
        save()
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit

    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              lora_frozen_root=str(runs / '20261128-phase1-dura-realgain-crossdata-s2026112845-r01'),
              external_releases=[str(runs / n) for n in (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
                                                          WISIG_PACKET, WISIG_PARENT,
                                                          '20261128-phase1-dura-realgain-crossdata-s2026112845-r01')]
              + [str(releases / '20261125-phase1-wisig-native-fir-source-m4-r01-release')])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    run = commands.add_parser('dispatch')
    run.add_argument('runtime', type=Path)
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    (dispatch if action == 'dispatch' else publish)(**args)
