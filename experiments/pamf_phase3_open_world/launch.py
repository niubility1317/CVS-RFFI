"""One immutable Phase3 owner; never stops or excludes another user's work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

STAGES = ('smoke', 'calibrate', 'build', 'predict', 'score')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def admissible(free_mb, owned_workers, minimum_free_mb=4096):
    return owned_workers < 2 and free_mb >= minimum_free_mb


def validate_runtime(rt):
    required = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner',
                'allowed_gpus', 'max_added_workers_per_gpu', 'model', 'data'}
    if not required <= set(rt) or rt['max_added_workers_per_gpu'] != 2:
        raise ValueError('Complete runtime and exactly two added worker slots/card required')
    root = Path(rt['run_root'])
    if not root.is_absolute() or root.name != rt['run_id'] or not rt['launch_owner']:
        raise ValueError('Absolute run-owned immutable output root required')
    if not rt['allowed_gpus'] or any(type(g) is not int or g < 0 for g in rt['allowed_gpus']):
        raise ValueError('Explicit physical GPU indices required')
    if len(rt['code_commit']) != 40:
        raise ValueError('Actual code Git commit required')


def choose_gpu(rt):
    while True:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        available = []
        for line in result.splitlines():
            parts = line.split(',')
            gpu, free = int(parts[0].strip()), int(parts[1].strip())
            # This dispatcher is sequential: it adds exactly one GPU worker.
            # Existing CUDA PIDs and their account names do not exclude a card.
            if gpu in rt['allowed_gpus'] and admissible(free, 0, rt.get('minimum_free_mb', 4096)):
                available.append((free, gpu))
        if available:
            return max(available)[1]
        print('WAITING_FOR_MEMORY no processes stopped', flush=True)
        time.sleep(20)


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'configs').mkdir()
    state = dict(status='PREPARING', run_id=rt['run_id'], code_commit=rt['code_commit'],
                 owner=rt['launch_owner'], pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
                 started_at=time.time(), jobs={}, max_added_workers_per_gpu=2,
                 foreign_processes_stopped=0)

    def record():
        write(root / 'pipeline_state.json', state, exclusive=False)

    record()
    model = dict(rt['model'], run_id=rt['run_id'], code_commit=rt['code_commit'])
    source_cache = model.pop('source_cache')
    retention = model.pop('retention', .99)
    for kind in ('smoke', 'calibrate', 'predict'):
        cfg = dict(model, row_id=rt['run_id'] + '-' + kind,
                   output_root=str(root / ('calibration' if kind == 'calibrate' else kind)))
        if kind == 'predict':
            cfg.update(public_root=str(root / 'data/evaluation/public'), calibration_root=str(root / 'calibration'))
        else:
            cfg.update(source_cache=source_cache, retention=retention)
        write(root / 'configs' / (kind + '.json'), cfg)
    commands = {}
    for kind in ('smoke', 'calibrate', 'predict'):
        commands[kind] = [rt['python'], '-B', '-u', '-m', 'experiments.pamf_phase3_open_world.runner',
                          kind, '--config', str(root / 'configs' / (kind + '.json')), '--device', 'cuda:0']
    data = rt['data']
    commands['build'] = [rt['python'], '-B', '-u', '-m', 'experiments.pamf_phase3_open_world.data',
        '--source-root', data['source_root'], '--output-root', str(root / 'data'),
        '--known-classes', *data['known_classes'], '--unknown-classes', *data['unknown_classes'],
        '--receivers', *map(str, data['receivers']), '--seed', str(data['evaluation_seed'])]
    commands['score'] = [rt['python'], '-B', '-u', '-m', 'experiments.pamf_phase3_open_world.scoring',
        '--predictions', str(root / 'predict/predictions.npz'),
        '--truth', str(root / 'data/evaluation/truth/truth.npz'), '--output', str(root / 'score'),
        '--known-class-count', str(len(data['known_classes']))]
    try:
        for kind in STAGES:
            gpu = choose_gpu(rt) if kind in ('smoke', 'calibrate', 'predict') else None
            env = dict(os.environ, PYTHONPATH=rt['release_root'],
                       CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            log_path = root / 'logs' / (kind + '.log')
            with log_path.open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(commands[kind], cwd=rt['release_root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                state['status'] = 'RUNNING'
                state['jobs'][kind] = dict(status='RUNNING', pid=child.pid, gpu=gpu,
                    cwd=rt['release_root'], argv=commands[kind], log=str(log_path), started_at=time.time())
                record()
                status = child.wait()
            if status:
                raise RuntimeError(kind + ' failed; exit=' + str(status) + '; see ' + str(log_path))
            if kind == 'build':
                marker = read(root / 'data/evaluation/public/manifest.json')
                if marker.get('data_status') != 'VALIDATED_ONCE':
                    raise ValueError('Builder did not produce valid fixed observations')
                if marker.get('count') != data['expected_query_count']:
                    raise ValueError('Fixed pilot query coverage differs from preregistration')
            else:
                folder = 'calibration' if kind == 'calibrate' else kind
                marker = read(root / folder / 'complete.json')
                expected = dict(smoke='PASS', calibrate='SOURCE_CALIBRATION_FROZEN',
                                predict='ALL_PREDICTIONS_COMPLETE', score='SCORED_COMPLETE')[kind]
                if marker.get('status') != expected:
                    raise ValueError(kind + ' completion artifact differs: ' + repr(marker.get('status')))
            state['jobs'][kind].update(status='COMPLETE', ended_at=time.time())
            record()
        state.update(status='ANALYZED', ended_at=time.time(), scientific_success='SEE_SCORE_GOALS')
        record()
        print('ANALYZED', rt['run_id'], 'scientific results in score/scores.json', flush=True)
    except Exception as exc:
        state.update(status='STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE', failure=str(exc),
                     ended_at=time.time(), performance_claim='NO_COMPLETE_PERFORMANCE_RESULT')
        record()
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    dispatch(read(args.runtime))


if __name__ == '__main__':
    main()
