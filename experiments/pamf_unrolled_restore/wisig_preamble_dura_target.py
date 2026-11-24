"""Truth-blind WiSig Main split replay of the single frozen preamble DURA."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.evaluation import handle, reject_truth
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_repl_view_shuffle.plan import RUN as F_RUN, TASKS, VIEWS
from experiments.pamf_repl_view_shuffle.runner import activate
from experiments.pamf_repl_view_shuffle.source import load_reader
from . import wisig_anchored_window_source as frozen
from . import wisig_short_training_source as data
from .wisig_preamble_dura_source import RUN as ORIGIN, build, gate_features
from .wisig_uncertain_prior import as_complex, as_iq
from .wisig_uncertain_source import source_iq


RUN = '20261124-phase1-wisig-preamble-dura-main-m4-r01'
ORIGIN_CODE = '53ad5fae18df08c340e2464fb0ace4edf33386a4'
BASELINE = '20261123-phase1-wisig-r69-paired-main-m6-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
ROUTES = ('full', 'residual')
COUNTS = {'cross_rx': 126000, 'cross_both': 42000}
MODES = ('raw', 'R69', 'preamble_gated')
BATCH = 128
ROWS = {f'{route}-{split}': (route, split) for route in ROUTES for split in TASKS}


def historical(root, kind, seed):
    row = f'main_split-repl-shuffled_time-3view-s{seed}'
    original = read(root / 'configs' / f'{kind}__{row}.json')
    if (original['run_id'], original['row_id'], original['model_seed']) != (F_RUN, row, seed):
        raise ValueError('Expected the registered Main split Tri-SIG-TIME reader')
    if set(original['tests']) != set(TASKS):
        raise ValueError('Expected the two registered Main split tasks')
    if kind == 'predict':
        reject_truth(original)
    return original


def view_root(rt, original, route, split):
    if route == 'full':
        return Path(rt['baseline_root']) / 'full_views' / split
    return Path(original['tests'][split]['views_root'])


def frozen_candidate(rt):
    project = Path(rt['project'])
    original = read(project / 'releases' / f'{ORIGIN}-release' / 'runtime.json')
    if original['run_id'] != ORIGIN or original['code_commit'] != ORIGIN_CODE:
        raise ValueError('Expected the registered WiSig source fit execution')
    fit = Path(original['run_root'])
    record = fit / 'preamble-complete.json'
    if not record.is_file():
        raise ValueError('The fixed terminal WiSig DURA weights and gate must be frozen')
    (model, gate), = frozen.load_models({'run_root': str(fit)}, arms=('preamble',), build=build)
    return model, gate


def readers(rt):
    original = {seed: historical(Path(rt['recognizer_run']), 'predict', seed) for seed in SEEDS}
    models = {}
    for seed, cfg in original.items():
        model, evidence = load_reader(cfg['source_output'], read(cfg['source_contract']), 'cuda:0')
        if (evidence['run_id'], evidence['row_id'], evidence['model_seed']) != (F_RUN, cfg['row_id'], seed):
            raise ValueError('Frozen source head or WiSig seed changed')
        models[seed] = model.eval().requires_grad_(False)
    return original, models


def single_packet_median(run):
    durations = []
    for index in range(20):
        torch.cuda.synchronize()
        start = perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= 5:
            durations.append((perf_counter()-start)*1000)
    return float(np.median(durations))


@torch.no_grad()
def smoke(rt):
    model, gate = frozen_candidate(rt)
    _, heads = readers(rt)
    restore = data.original_r69(Path(rt['parent_root']))
    source, _, calibration = source_iq(Path(rt['source_root']))
    quiet = tensor(source[calibration[:12]], 'cuda:0')
    clean = as_complex(quiet)
    observed = clean + .75*torch.randn(clean.shape, device=clean.device, dtype=clean.dtype,
                                       generator=torch.Generator(device=clean.device).manual_seed(776121))
    bypass, active = frozen.guarded(model, gate, clean, gate_features=gate_features)
    if active.any() or not torch.equal(bypass, clean):
        raise ValueError('Matched source clean must take exact bypass')
    result, triggered = frozen.guarded(model, gate, observed, gate_features=gate_features)
    baseline, = restore(quiet)
    for head in heads.values():
        for signal in (baseline, quiet, as_iq(result)):
            frozen.classify(head, as_complex(signal))
    selected = int(triggered.to(torch.int32).argmax())
    stressed = observed[selected:selected+1]
    latency = {}
    for label, sample in (('source_clean', clean[:1]), ('source_perturbed', stressed)):
        batch = as_iq(sample)
        latency[label] = dict(R69_ms=single_packet_median(lambda: restore(batch)),
                              candidate_ms=single_packet_median(lambda: frozen.guarded(
                                  model, gate, sample, gate_features=gate_features)))
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', seeds=list(SEEDS),
          source_u=len(quiet), modified_in_probe=int(triggered.sum()),
          timed_perturbed_active=bool(triggered[selected]), recovery_b1_median_ms=latency,
          device=torch.cuda.get_device_name(), target_reads=0))


@torch.no_grad()
def predict(rt, route, split):
    model, gate = frozen_candidate(rt)
    original, heads = readers(rt)
    restore = data.original_r69(Path(rt['parent_root']))
    base = original[SEEDS[0]]
    path = view_root(rt, base, route, split)
    if any(view_root(rt, cfg, route, split) != path or cfg['classes'] != base['classes'] for cfg in original.values()):
        raise ValueError('Frozen readers disagree on the target view or class map')
    manifest, ids, positions = handle(path, base['classes'])
    if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
        raise ValueError('Wrong Main split target population or received view')
    outputs, activation = {}, {}
    for view in VIEWS:
        file = Path(manifest['clean_ref']) if view == 'clean' else Path(manifest['observations_root']) / f'{view}.npy'
        observed = np.load(file, mmap_mode='r', allow_pickle=False)
        count_ok = len(observed) == len(ids) if positions is None else positions.min() >= 0 and positions.max() < len(observed)
        if observed.shape[1:] != (2, 256) or observed.dtype != np.float32 or not count_ok:
            raise ValueError('Received IQ array differs from the registered view')
        result = {key: np.empty((len(SEEDS), len(ids)), dtype=np.uint8) for key in MODES}
        activated = 0
        for start in range(0, len(ids), BATCH):
            stop = min(start+BATCH, len(ids))
            selection = np.s_[start:stop]
            iq = tensor(observed[selection if positions is None else positions[selection]], 'cuda:0')
            received = as_complex(iq)
            baseline, = restore(iq)
            amended, mask = frozen.guarded(model, gate, received, gate_features=gate_features)
            active = int(mask.sum())
            activated += active
            cues = (received, as_complex(baseline), amended)
            for i, head in enumerate(heads.values()):
                decisions = frozen.classify(head, cues[0]).argmax(1)
                result['raw'][i, selection] = decisions.cpu().tolist()
                result['R69'][i, selection] = frozen.classify(head, cues[1]).argmax(1).cpu().tolist()
                if active:
                    amended_classes = frozen.classify(head, cues[2]).argmax(1)
                    decisions = torch.where(mask, amended_classes, decisions)
                result[MODES[-1]][i, selection] = decisions.cpu().tolist()
        outputs.update({f'{key}__{view}': value for key, value in result.items()})
        activation[view] = activated
        print('WISIG_DURA_TARGET_VIEW', route, split, view, len(ids), activated, flush=True)
    row = Path(rt['run_root']) / route / split
    row.mkdir(parents=True, exist_ok=False)
    with (row/'decisions.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **outputs)
    write(row/'complete.json', dict(status='PREDICTIONS_FROZEN', route=route, split=split, source_run=ORIGIN,
          count=len(ids), seeds=list(SEEDS), modes=list(MODES), views=list(VIEWS), active=activation, truth_read=False))


def launch(rt, kind, name, gpu=None):
    base = Path(rt['run_root'])
    args = [sys.executable, '-u', '-m', rt['runner'], kind, str(Path(rt['release_root'])/'runtime.json'), name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (base/'logs'/f'{kind}-{name}.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    print('WISIG_DURA_TARGET_STARTED', kind, name, child.pid, gpu, flush=True)
    return child


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    (root/'logs').mkdir()
    state = root/'state.json'
    try:
        fields = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits']
        inventory = (tuple(map(int, line.split(','))) for line in subprocess.check_output(fields, text=True).splitlines())
        priority = [index for _, _, index in sorted((load, -free, index) for index, load, free in inventory)]
        rt = dict(rt, allowed_gpus=priority)
        write(state, dict(status='SOURCE_SMOKE'))
        schedule(rt, [('smoke', 'source')], launch)
        write(state, dict(status='PREDICTING'), exclusive=False)
        schedule(rt, [('predict', name) for name in ROWS], launch)
        write(state, dict(status='PREDICTIONS_FROZEN', truth_read=False), exclusive=False)
        if launch(rt, 'score', 'cpu').wait():
            raise RuntimeError('Independent WiSig Main split scorer failed')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    base = PurePosixPath(rt['project'])
    runs = base/'runs'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(runs/RUN), release_root=str(base/'releases'/f'{RUN}-release'),
              source_root=str(runs/data.CACHE_RUN/'cache'), parent_root=str(runs/data.R17_RUN),
              baseline_root=str(runs/BASELINE), recognizer_run=str(runs/F_RUN),
              max_workers=4, max_compute_processes=None)
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',
                                                       'tools/wisig_paired_main_uncertainty.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+key, type=Path, required=True)
    for key in ('dispatch', 'smoke', 'predict', 'score'):
        child = commands.add_parser(key)
        child.add_argument('runtime', type=Path)
        child.add_argument('name', nargs='?')
    ns = vars(parser.parse_args())
    action = ns.pop('action')
    if action == 'publish':
        publish(**ns)
    else:
        rt = read(ns['runtime'])
        if rt['run_id'] != RUN:
            raise ValueError('Expected the immutable WiSig main comparison run')
        if action == 'dispatch':
            dispatch(rt)
        elif action == 'score':
            from .wisig_preamble_dura_score import score
            score(rt)
        else:
            activate(rt)
            if action == 'smoke':
                smoke(rt)
            else:
                predict(rt, *ROWS[ns['name']])
