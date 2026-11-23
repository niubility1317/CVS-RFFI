"""Paired Main split WiSig reference with the frozen three-view F and R69."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

from experiments.cric_dura_comparison.evaluation import handle, reject_truth
from experiments.cric_dura_comparison.io import read, tensor, write
from .plan import RUN as F_RUN, TASKS, VIEWS


RUN = '20261123-phase1-wisig-r69-paired-main-m6-r01'
R69_RUN = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
R69_TRAIN_COMMIT = 'b2a2ce937a64654e17e84c2aa817b9da4257a48f'
SEEDS = (2026092701, 2026092702, 2026092703)
COUNTS = dict(cross_rx=126000, cross_both=42000)
ROUTES = ('residual', 'full')
READOUTS = ['raw', 'R69']
BATCH = 128


def historical(rt, kind, seed):
    row = f'main_split-repl-shuffled_time-3view-s{seed}'
    path = Path(rt['recognizer_run']) / 'configs' / f'{kind}__{row}.json'
    cfg = read(path)
    if (cfg['run_id'], cfg['row_id'], cfg['model_seed']) != (F_RUN, row, seed):
        raise ValueError('Unexpected historical Main split model binding')
    return cfg


def build_full(rt):
    """Build full observations with the registered residual clean, scene and latent IDs."""
    from dataclasses import replace

    parent = Path(rt['residual_reference'])
    manifest = read(parent / 'manifest.json')
    if (manifest['status'], manifest['channel'], manifest['count']) != (
            'VALIDATED_ONCE', 'residual/post_sync/noeq', sum(COUNTS.values())):
        raise ValueError('Expected original registered 168,000-packet residual reference')
    if manifest['scenes'] != list(VIEWS[1:]) or manifest['truth_read'] is not False:
        raise ValueError('Original six-scene receipt differs')
    with np.load(parent / 'index.npz', allow_pickle=False) as index:
        all_ids = index['ids']
    if all_ids.shape != (sum(COUNTS.values()),) or len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError('Original physical ID coverage differs')
    position = {ident: i for i, ident in enumerate(all_ids)}
    original = historical(rt, 'predict', SEEDS[0])
    clean = np.load(manifest['clean_ref'], mmap_mode='r', allow_pickle=False)
    if clean.shape != (len(all_ids), 2, 256) or clean.dtype != np.float32:
        raise ValueError('Original registered clean geometry differs')
    full = Path(rt['run_root']) / 'full_views'
    full.mkdir()
    for task in TASKS:
        residual_root = original['tests'][task]['views_root']
        view, ids, locations = handle(residual_root, original['classes'])
        if len(ids) != COUNTS[task]:
            raise ValueError('Main split target population count differs')
        where = np.asarray([position[ident] for ident in ids], dtype=np.int64)
        residual_clean = np.load(view['clean_ref'], mmap_mode='r', allow_pickle=False)
        for part in range(0, len(ids), 2048):
            piece = slice(part, part + 2048)
            old = piece if locations is None else locations[piece]
            if not np.array_equal(clean[where[piece]], residual_clean[old]):
                raise ValueError('Full and residual clean IQ differ for the same physical IDs')
        wrapper = full / task
        wrapper.mkdir()
        np.savez(wrapper / 'index.npz', ids=ids, positions=where)
        output_manifest = dict(view, clean_ref=manifest['clean_ref'], observations_root=str(full),
                               channel='full/post_sync/noeq', matched_residual_root=residual_root,
                               latent_namespace=manifest['namespace'], receiver_session=manifest['session'])
        write(wrapper / 'pending.json', output_manifest)
    with np.load(full / TASKS[0] / 'index.npz', allow_pickle=False) as first:
        with np.load(full / TASKS[1] / 'index.npz', allow_pickle=False) as other:
            if not np.array_equal(np.sort(np.concatenate((first['positions'], other['positions']))),
                                  np.arange(len(all_ids))):
                raise ValueError('Target partitions must form a disjoint complete physical union')
    sys.path.insert(0, str(Path(rt['channel_release']) / 'code'))
    from leo_practical import Config
    configs = {item['scene']: item['config'] for item in manifest['evidence']}
    with ProcessPoolExecutor(max_workers=rt['view_workers'],
                             initializer=init_channel,
                             initargs=(manifest['clean_ref'], all_ids, manifest)) as pool:
        for scene in VIEWS[1:]:
            cfg = replace(Config(**configs[scene]), processing_route='full')
            if cfg.mode != 'post_sync' or cfg.equalization_enabled:
                raise ValueError('Full must change only the registered route')
            out = np.lib.format.open_memmap(full / (scene + '.npy'), mode='w+',
                                            dtype=np.float32, shape=clean.shape)
            jobs = ((cfg, start, min(start + 768, len(all_ids)))
                    for start in range(0, len(all_ids), 768))
            for start, received in pool.map(channel_chunk, jobs):
                out[start:start + len(received)] = received
            out.flush()
            del out
            print('FULL_RECEPTION_COMPLETE', scene, len(all_ids), flush=True)
    for task in TASKS:
        pending = full / task / 'pending.json'
        pending.rename(pending.with_name('manifest.json'))


def init_channel(clean, ids, manifest):
    global _CHANNEL_CLEAN, _CHANNEL_IDS, _CHANNEL_MANIFEST
    _CHANNEL_CLEAN = np.load(clean, mmap_mode='r', allow_pickle=False)
    _CHANNEL_IDS, _CHANNEL_MANIFEST = ids, manifest


def channel_chunk(job):
    from leo_practical import apply_leo_practical_channel_batch

    cfg, start, stop = job
    m = _CHANNEL_MANIFEST
    x = np.asarray(_CHANNEL_CLEAN[start:stop], dtype=np.float64)
    data, _, _ = apply_leo_practical_channel_batch(
        x, cfg, seed=m['seed'], sample_ids=_CHANNEL_IDS[start:stop],
        session_ids=[m['session']] * len(x), realization_namespace=m['namespace'],
        receiver_seed=m['receiver_seed'], return_meta=False)
    if data.shape != x.shape or not np.isfinite(data).all():
        raise ValueError('Nonfinite or incomplete full received batch')
    return start, data


def model_pair(rt, original, device):
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .source import load_reader

    contract = read(original['source_contract'])
    model, payload = load_checkpoint(rt['r69_checkpoint'], device)
    if (payload['arm'], payload['epoch'], payload['updates'], payload['code_commit']) != (
            'stage2', 20, 4940, R69_TRAIN_COMMIT) or sum(p.numel() for p in model.parameters()) != 266951:
        raise ValueError('Unexpected original R69 source-final weights')
    reader, source = load_reader(original['source_output'], contract, device)
    if (source['run_id'], source['row_id'], source['model_seed']) != (
            original['run_id'], original['row_id'], original['model_seed']):
        raise ValueError('Different frozen reader')
    return reader, InferenceEngine(model.requires_grad_(False), reader,
                                  reuse=True, graph=True, native=False)


def smoke(rt):
    import torch

    original = historical(rt, 'predict', SEEDS[0])
    reader, restorer = model_pair(rt, original, 'cuda:0')
    source = Path(original['source_contract']).parent / 'L_s_x.npy'
    sample = tensor(np.load(source, mmap_mode='r', allow_pickle=False)[:2], 'cuda:0')
    with torch.inference_mode():
        raw = reader.classify_features(reader.features(sample))
        restored = restorer(sample)[3]
    if any(x.shape != (len(sample), 6) or not torch.isfinite(x).all() for x in (raw, restored)):
        raise ValueError('Real checkpoint no-query smoke failed')
    print('WI_SIG_R69_SOURCE_SMOKE_PASS', flush=True)


def view_root(rt, task, route, residual):
    return residual if route == 'residual' else str(Path(rt['run_root']) / 'full_views' / task)


def predict(rt, route, seed):
    import torch

    original = historical(rt, 'predict', seed)
    reject_truth(original)
    if list(original['tests']) != list(TASKS):
        raise ValueError('Expected both truth-blind Main split views')
    reader, restorer = model_pair(rt, original, 'cuda:0')
    row = Path(rt['run_root']) / route / str(seed) / 'predictions'
    row.mkdir(parents=True, exist_ok=False)
    completion = {}
    for task in TASKS:
        view_path = view_root(rt, task, route, original['tests'][task]['views_root'])
        m, ids, pos = handle(view_path, original['classes'])
        if len(ids) != COUNTS[task] or m['channel'] != f'{route}/post_sync/noeq':
            raise ValueError('Wrong fixed Main split route or target population')
        values = {}
        for view in VIEWS:
            path = m['clean_ref'] if view == 'clean' else Path(m['observations_root']) / (view + '.npy')
            data = np.load(path, mmap_mode='r', allow_pickle=False)
            if (data.ndim != 3 or data.shape[1:] != (2, 256)
                    or (pos is None and len(data) != len(ids))
                    or (pos is not None and (pos.min() < 0 or pos.max() >= len(data)))):
                raise ValueError('Invalid route array coverage')
            result = {name: [] for name in READOUTS}
            with torch.inference_mode():
                for start in range(0, len(ids), BATCH):
                    stop = start + BATCH
                    selected = slice(start, stop) if pos is None else pos[start:stop]
                    x = tensor(np.ascontiguousarray(data[selected], dtype=np.float32), 'cuda:0')
                    raw = reader.classify_features(reader.features(x))
                    restored = restorer(x)[3]
                    for name, logits in zip(READOUTS, (raw, restored), strict=True):
                        if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
                            raise ValueError('Invalid frozen default head')
                        result[name].extend(logits.argmax(1).cpu().tolist())
            for name in READOUTS:
                values[name + '__' + view] = np.asarray(result[name], dtype=np.int64)
            print('MAIN_BASELINE_VIEW_COMPLETE', route, seed, task, view, len(ids), flush=True)
        folder = row / task
        folder.mkdir()
        np.savez(folder / 'predictions.npz', ids=ids, **values)
        completion[task] = dict(count=len(ids), views=list(VIEWS), readouts=READOUTS)
    write(row / 'complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', run_id=RUN,
          row_id=f'{route}-s{seed}', tests=completion, views=list(VIEWS),
          readouts=READOUTS, truth_read=False, query_fit=False))


def score_config(rt, route, seed):
    c = historical(rt, 'score', seed)
    row = Path(rt['run_root']) / route / str(seed)
    tests = {task: dict(item, views_root=view_root(rt, task, route, item['views_root']))
             for task, item in c['tests'].items()}
    return dict(c, run_id=RUN, row_id=f'{route}-s{seed}', tests=tests,
                readouts=READOUTS, prediction_output=str(row / 'predictions'),
                output_root=str(row / 'scores'))


def score_all(rt):
    from experiments.cric_dura_comparison import evaluation as existing

    # All six rows must close before the independent CPU scorer can open truth.
    existing.tasks = lambda _: list(TASKS)
    configs = [(route, seed, score_config(rt, route, seed)) for route in ROUTES for seed in SEEDS]
    for _, _, c in configs:
        existing.preflight(c)
    for _, _, c in configs:
        existing.score(c)
    summarize(rt, configs)


def summarize(rt, configs):
    rows = []
    for route, seed, c in configs:
        score = read(Path(c['output_root']) / 'scores.json')
        if score['truth_last'] is not True or score['target_feedback_forbidden'] is not True:
            raise ValueError('Expected untouched truth-last scores')
        raw = {(r['test'], r['view'], r['readout']): r for r in score['results']
               if r['dimension'] == 'overall'}
        paired = {(p['test'], p['view']): p for p in score['paired_raw']
                  if p['dimension'] == 'overall' and p['readout'] == 'R69'}
        for split in (*TASKS, 'combined'):
            tasks = TASKS if split == 'combined' else (split,)
            expected = sum(COUNTS[task] for task in tasks)
            views = {}
            for view in VIEWS:
                correct = {}
                for readout in READOUTS:
                    parts = [raw[task, view, readout] for task in tasks]
                    if sum(part['count'] for part in parts) != expected:
                        raise ValueError('Unclosed target population in scores')
                    correct[readout] = sum(int(np.trace(part['confusion'])) for part in parts)
                repairs = sum(paired[task, view]['repairs'] for task in tasks)
                harms = sum(paired[task, view]['damage'] for task in tasks)
                if correct['R69'] - correct['raw'] != repairs - harms:
                    raise ValueError('Paired corrections disagree with correct counts')
                views[view] = dict(correct=correct, repairs=repairs, harms=harms)
            mean = {name: sum(views[view]['correct'][name] for view in VIEWS[1:]) /
                    (len(VIEWS[1:]) * expected) for name in READOUTS}
            rows.append(dict(route=route, seed=seed, split=split, count_per_view=expected,
                             six_practical_accuracy=mean, by_view=views))
    write(Path(rt['run_root']) / 'summary.json', dict(status='ANALYZED',
          frozen_reader_run=F_RUN, r69_run=R69_RUN, truth_last=True, rows=rows))


def launch(rt, kind, name, gpu=None):
    root = Path(rt['run_root'])
    log = root / 'logs' / f'{kind}-{name}.log'
    cmd = [rt['python'], '-u', '-m', __spec__.name, kind,
           str(Path(rt['release_root']) / 'runtime.json'), name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('x', encoding='utf-8') as stream:
        worker = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT)
    print('WISIG_BASELINE_STARTED', kind, name, worker.pid, gpu, flush=True)
    return worker


def dispatch(rt):
    from experiments.pamf_lora_main.packet_runner import schedule

    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    (root / 'logs').mkdir()
    state = root / 'state.json'
    try:
        write(state, dict(status='SMOKE'))
        schedule(rt, [('smoke', 'source')], launch)
        write(state, dict(status='BASELINE_PREDICTION'), exclusive=False)
        full = launch(rt, 'build_full', 'views')
        schedule(rt, [('predict', f'residual-{seed}') for seed in SEEDS], launch)
        if full.wait():
            raise RuntimeError('Full received view construction failed')
        schedule(rt, [('predict', f'full-{seed}') for seed in SEEDS], launch)
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        if launch(rt, 'score_all', 'cpu').wait():
            raise RuntimeError('Independent CPU scorer failed')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, local_context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit

    context = read(local_context)
    project = context['project']
    run_root = f'{project}/runs/{RUN}'
    release_root = f'{project}/releases/{RUN}-release'
    rt = {key: context[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, run_root=run_root, release_root=release_root, runner=__spec__.name,
              code_commit=release_commit(repo), allowed_gpus=[3, 6, 2, 7, 0, 1, 4, 5],
              max_workers=3, max_compute_processes=4, view_workers=16,
              recognizer_run=f'{project}/runs/{F_RUN}',
              residual_reference=f'{project}/runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views',
              channel_release=f'{project}/releases/cvs_residual_sixscene_20261002_r01',
              r69_checkpoint=f'{project}/runs/{R69_RUN}/stage2/native_map_prior/final.pt')
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    publication = commands.add_parser('publish')
    for key in ('repo', 'output', 'local-context', 'ssh-config'):
        publication.add_argument('--' + key, required=True, type=Path)
    for kind in ('dispatch', 'smoke', 'build_full', 'predict', 'score_all'):
        entry = commands.add_parser(kind)
        entry.add_argument('runtime', type=Path)
        entry.add_argument('name', nargs='?')
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    else:
        rt = read(args['runtime'])
        if action == 'predict' or action == 'smoke':
            from .runner import activate
            activate(rt)
        if action == 'predict':
            route, seed = args['name'].split('-')
            seed = int(seed)
            if route not in ROUTES or seed not in SEEDS:
                raise ValueError('Unregistered route/seed')
            predict(rt, route, seed)
        else:
            globals()[action](rt)
