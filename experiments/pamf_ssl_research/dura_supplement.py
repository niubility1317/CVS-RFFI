"""Frozen PSM/RXPSM DURA readouts, isolated from all training owners."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read, write

RUN = '20261023-phase1-PAMF-SSL-DURA-eval-m6-r01'
SOURCE = '20261023-phase1-PAMF-physical-softmatch-m6-r02'
SEEDS = (2026092701, 2026092702, 2026092703)
GPU_BY_SEED = dict(zip(SEEDS, (5, 6, 7)))


def prepare(context, project, release, commit):
    root = project + '/runs/' + RUN
    blobs, rows = {}, []
    for seed in SEEDS:
        original = context[str(seed)]['F21']
        for method in ('PSM', 'RXPSM'):
            row_id = method + '-s' + str(seed)
            source_root = project + '/runs/' + SOURCE + '/' + row_id
            prediction = deepcopy(original['predict'])
            for key in ('original_arm', 'strict_supervised_system'):
                prediction.pop(key, None)
            prediction.update(run_id=RUN, row_id=row_id, source_run_id=SOURCE,
                source_row_id=row_id, source_output=source_root + '/source',
                raw_predictions_from=source_root + '/predictions',
                source_physics_diagnostics=False,
                output_root=root + '/' + row_id + '/predictions', code_commit=commit)
            from experiments.cric_dura_comparison.evaluation import reject_truth
            reject_truth(prediction)
            score = deepcopy(original['score'])
            score.update(run_id=RUN, row_id=row_id,
                prediction_output=prediction['output_root'],
                output_root=root + '/' + row_id + '/scores', code_commit=commit)
            paths = {k: release + '/configs/' + row_id + '-' + k + '.json'
                     for k in ('predict', 'score')}
            rows.append(dict(row_id=row_id, method=method, model_seed=seed,
                gpu=GPU_BY_SEED[seed], prediction_config=paths['predict'],
                score_config=paths['score']))
            blobs['configs/' + row_id + '-predict.json'] = prediction
            blobs['configs/' + row_id + '-score.json'] = score
    runtime = dict(run_id=RUN, project=project, release_root=release, run_root=root,
        runtime_file=release + '/runtime.json', code_commit=commit, rows=rows,
        launch_owner='codex-side-pamf-ssl-dura-20261023', allowed_gpus=[5, 6, 7],
        max_own_per_gpu=1, max_compute_per_gpu=2, minimum_free_MiB=12000,
        python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',
        frozen_sources=context[str(SEEDS[0])]['F21']['predict']['frozen_sources'],
        baseline_scores={str(seed): {arm: context[str(seed)][arm]['score']['output_root']
                                   for arm in ('F20', 'F21')} for seed in SEEDS},
        source_run_id=SOURCE, new_F_training=0, new_G_training=0,
        stop_rule='Smoke failure or two identical technical failures stop dispatch; preserve all foreign/healthy jobs and artifacts; no performance-based changes.')
    validate(runtime, blobs)
    blobs['runtime.json'] = runtime
    return runtime, blobs


def validate(runtime, blobs):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    rows = runtime['rows']
    expected = {(method, seed) for method in ('PSM', 'RXPSM') for seed in SEEDS}
    if len(rows) != 6 or {(r['method'], r['model_seed']) for r in rows} != expected:
        raise ValueError('Exactly the six frozen SSL source models required')
    for row in rows:
        stem = 'configs/' + row['row_id']
        c, s = blobs[stem + '-predict.json'], blobs[stem + '-score.json']
        reject_truth(c)
        if (row['gpu'] != GPU_BY_SEED[row['model_seed']] or c['source_run_id'] != SOURCE
                or c['source_row_id'] != row['row_id'] or c['source_output'] !=
                runtime['project'] + '/runs/' + SOURCE + '/' + row['row_id'] + '/source'
                or list(c['readouts']) != ['raw', 'DURA-LU']
                or c['readouts']['DURA-LU']['kind'] != 'original_F0_LU'
                or c['readouts']['DURA-LU']['roles'] != 'LU'
                or c['readouts']['DURA-LU']['checkpoint'] != runtime['project'] +
                '/runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'
                or c['raw_predictions_from'] != runtime['project'] + '/runs/' + SOURCE + '/' + row['row_id'] + '/predictions'
                or s['prediction_output'] != c['output_root']
                or s['readouts'] != list(c['readouts'])
                or not c['output_root'].startswith(runtime['run_root'] + '/')
                or not s['output_root'].startswith(runtime['run_root'] + '/')):
            raise ValueError('Frozen source/readout/output binding differs')


def publish(args):
    from . import publish as existing
    # Select only this release's dispatcher; the existing training owners and
    # their immutable release files are never touched.
    original = existing.REMOTE
    try:
        existing.REMOTE = original.replace('experiments.pamf_ssl_research.control',
                                          'experiments.pamf_ssl_research.dura_supplement')
        existing.publish(args.repo, args.context, args.output, args.ssh_config,
                         prepare_run=prepare, run_id=RUN)
    finally:
        existing.REMOTE = original


def smoke(runtime):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(runtime)
    numerical_policy()
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.cric_dura_comparison.frozen_inference import load_F, load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    c = read(runtime['rows'][0]['prediction_config'])
    contract = read(c['source_contract'])
    reader, _ = load_F(c['source_output'], contract, 'cuda:0')
    model = load_G(c['readouts']['DURA-LU'], contract, 'cuda:0')
    x = tensor(np.load(Path(c['source_cache']) / 'L_s_x.npy', mmap_mode='r')[:2].copy(), 'cuda:0')
    with torch.inference_mode():
        direct = reader.classify_features(reader.features(x))
        restored, _, _, logits = InferenceEngine(model, reader, reuse=True, graph=True, native=False)(x)
        if (direct.shape != (2, 6) or logits.shape != (2, 6)
                or not torch.isfinite(direct).all() or not torch.isfinite(logits).all()
                or not torch.allclose(restored, model(x), atol=1e-5, rtol=1e-5)):
            raise ValueError('Frozen real-checkpoint DURA/default-head source smoke failed')
    write(Path(runtime['run_root']) / 'smoke.json', dict(status='PASS',
        F_source=c['source_output'], G=c['readouts']['DURA-LU'], source_samples=2,
        target_reads=0, training_updates=0, finite_default_head_logits=True,
        graph_native_IQ_match=True, immediate_continuation=True))
    print('SSL_DURA_SOURCE_SMOKE_PASS', flush=True)


def run(runtime):
    from experiments.cric_dura_comparison.source_queue import capacity
    root = Path(runtime['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    logs = root / 'logs'
    logs.mkdir()
    state = dict(status='WAITING_FOR_SMOKE_CAPACITY', pid=os.getpid(), cwd=os.getcwd(),
        argv=sys.argv, code_commit=runtime['code_commit'], owner=runtime['launch_owner'],
        observed_unix=time.time(), rows={r['row_id']: dict(status='QUEUED') for r in runtime['rows']})
    active, failures = {}, Counter()
    smoke_passed = False

    def launch(key, action, config, gpu):
        module = ('experiments.pamf_ssl_research.dura_supplement' if action == 'smoke'
                  else 'experiments.cric_dura_comparison.evaluation')
        command = [runtime['python'], '-u', '-m', module, action, '--config', config]
        path = logs / (key + '-' + action + '.log')
        stream = path.open('x')
        child = subprocess.Popen(command, cwd=runtime['release_root'],
            env=dict(os.environ, PYTHONPATH=runtime['release_root'],
                     CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                     OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        active[key] = dict(child=child, stream=stream, path=path, action=action, gpu=gpu)
        record = dict(status='SMOKE_RUNNING' if action == 'smoke' else
                      ('PREDICTING' if action == 'predict' else 'SCORING'),
                      pid=child.pid, gpu=gpu, config=config, log=str(path))
        if key == 'source_smoke':
            state['smoke'] = record
        else:
            state['rows'][key] = record

    while True:
        for key, item in list(active.items()):
            child = item['child']
            if child.poll() is None:
                continue
            item['stream'].close()
            action = item['action']
            completion = (root / 'smoke.json' if action == 'smoke' else root / key /
                          ('predictions' if action == 'predict' else 'scores') / 'complete.json')
            expected = {'smoke': 'PASS', 'predict': 'ALL_PREDICTIONS_COMPLETE', 'score': 'SCORED_COMPLETE'}[action]
            good = child.returncode == 0 and completion.is_file() and read(completion).get('status') == expected
            record = state['smoke'] if action == 'smoke' else state['rows'][key]
            record.update(status=('PASS' if action == 'smoke' else
                                 ('PREDICTIONS_COMPLETE' if action == 'predict' else 'SCORED')) if good else 'FAILED',
                          returncode=child.returncode)
            if good and action == 'smoke':
                smoke_passed = True
            if not good:
                tail = item['path'].read_text(errors='replace').splitlines()
                reason = tail[-1] if tail else 'missing completion'
                failures[reason] += 1
                record['failure'] = reason
                if action == 'smoke':
                    state['smoke_failed'] = True
            del active[key]
        systemic = state.get('smoke_failed', False) or any(v >= 2 for v in failures.values())
        if not smoke_passed and not active and not systemic:
            for gpu in runtime['allowed_gpus']:
                apps, free = capacity(gpu)
                if apps < runtime['max_compute_per_gpu'] and free >= runtime['minimum_free_MiB']:
                    launch('source_smoke', 'smoke', runtime['runtime_file'], gpu)
                    break
        if smoke_passed and not systemic:
            for row in runtime['rows']:
                key = row['row_id']
                if state['rows'][key]['status'] != 'QUEUED':
                    continue
                gpu = row['gpu']
                if any(v['gpu'] == gpu for v in active.values()):
                    continue
                apps, free = capacity(gpu)
                if apps < runtime['max_compute_per_gpu'] and free >= runtime['minimum_free_MiB']:
                    launch(key, 'predict', row['prediction_config'], gpu)
            if not any(v['action'] == 'score' for v in active.values()):
                for row in runtime['rows']:
                    key = row['row_id']
                    if state['rows'][key]['status'] == 'PREDICTIONS_COMPLETE':
                        launch(key, 'score', row['score_config'], None)
                        break
        pending = any(v['status'] in ('QUEUED', 'PREDICTIONS_COMPLETE') for v in state['rows'].values())
        state.update(observed_unix=time.time(), counts=dict(Counter(v['status'] for v in state['rows'].values())),
            failures=dict(failures), systemic_stop_new_launches=systemic,
            status='RUNNING' if smoke_passed else state['status'])
        write(root / 'state.json', state, exclusive=False)
        if not active and (systemic or (smoke_passed and not pending)):
            break
        time.sleep(10)
    state.update(status='ALL_SCORED' if all(v['status'] == 'SCORED' for v in state['rows'].values())
                 else 'PARTIAL_TECHNICAL_FAILURE', observed_unix=time.time())
    write(root / 'state.json', state, exclusive=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', choices=('smoke', 'control', 'publish'), default='control')
    parser.add_argument('--config')
    for key in ('repo', 'context', 'output', 'ssh-config'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.action == 'publish':
        if any(getattr(args, k) is None for k in ('repo', 'context', 'output', 'ssh_config')):
            parser.error('publish requires repo, context, output and ssh-config')
        publish(args)
    else:
        if not args.config:
            parser.error('smoke/control requires config')
        (smoke if args.action == 'smoke' else run)(read(args.config))
