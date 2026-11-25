"""Train original LoRa PAMF W24 with an added labeled-source satellite CE."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from math import cos, pi
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, METHOD, SEED, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify, load_restorers,
    native_full_view, predict, receipt, recover, score,
)


RUN = '20261125-phase1-lora-pamf-w24-sat-aug-s2026092702-r01'
MODES = ('direct', 'complex_response_half_center')
EPOCHS = 112
AUG_SEED = 2026112548
AUG_WEIGHT = .5
PRACTICAL = tuple(view for view in VIEWS if view != 'clean')
ELIGIBLE = (100, 104, 108, 112)


def scene_slots(count, draw):
    rng = np.random.default_rng([AUG_SEED, draw])
    return rng.permutation(np.arange(count) % len(PRACTICAL))


def eligible_checkpoint(epoch, current, best):
    return epoch in ELIGIBLE and (best is None
           or (current['correct'], -current['nll']) > (best['correct'], -best['nll']))


def receive_source(namespace, item):
    physical_id, rx, view, packet = item
    return native_full_view(packet, view, rx, physical_id, namespace)


def augmentation_cache(source, contract, receivers, root, workers):
    from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, clean_iq
    count = contract['counts']['L_s']
    raw = native(source, 'L_s', count)
    target = root / 'L_s_satellite.npy'
    output = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32,
                                      shape=(2, count, 8, 2, 432))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for draw in (0, 1):
            assigned = scene_slots(count, draw)
            namespace = f'w24_sat_aug_source_{AUG_SEED}_{draw}'
            sequence = ((i, SOURCE[int(receivers[i])], PRACTICAL[scene], raw[i])
                        for i, scene in enumerate(assigned))
            futures = pool.map(partial(receive_source, namespace), sequence, chunksize=12)
            for start in range(0, count, 32):
                size = min(32, count - start)
                packets = np.stack([next(futures) for _ in range(size)])
                output[draw, start:start + size] = clean_iq(packets, 384, ALL_CHIRPS, native_edges=True,
                                                          edge_points=24)
            print('W24_SAT_SOURCE_CACHE', draw, count, flush=True)
    output.flush()
    write(root / 'satellite_source.json', dict(status='SOURCE_READY', role='L_s', count=count,
          draws=2, modes=list(PRACTICAL), seed=AUG_SEED, source_protocol=contract['protocol'],
          target_reads=0))
    return output


def fit(rt):
    from experiments.pamf_lora_main import tri_clean95 as original

    activate(rt)
    root, src = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(src)
    row = original.NATIVE_WIDE_ROWS[METHOD]
    warmup = row['warmup']
    if (warmup, original.STEPS, row['edge_points']) != (96, 88, 24):
        raise ValueError('W24 original training contract differs')
    device = 'cuda:0'
    yn, rn = original.labels(src, contract, 'L_s')
    clean = Path(rt['clean_cache'])
    _, expected = original.cache_layout(384, row['chirps'], contract['protocol'],
                                        native_edges=True, edge_points=24)
    if read(clean / 'complete.json') != expected:
        raise ValueError('Original W24 clean source cache differs')
    satellite_cache = augmentation_cache(src, contract, rn, root, rt['cpu_workers'])
    def load(role):
        return tensor(np.load(clean / f'{role}.npy', mmap_mode='r', allow_pickle=False), device)

    out = root / 'F'
    out.mkdir()
    x, u, v = (load(role) for role in ('L_s', 'U_s', 'V'))
    y, rx = tensor(yn, device), tensor(rn, device)
    ur = tensor(original.labels(src, contract, 'U_s'), device)
    vy, vr = (tensor(k, device) for k in original.labels(src, contract, 'V'))
    satellite = (tensor(satellite_cache[0], device), tensor(satellite_cache[1], device))
    original.seed_all(SEED)
    f = original.clean_model(dict(scratch=True), root, contract, row, device)
    original.seed_all(SEED)
    with torch.no_grad():
        if any(not torch.isfinite(f(wave[:2])[0]).all() for wave in (x, satellite[0])):
            raise FloatingPointError('Real W24 clean and synthetic source smoke failed')
    # Reconstruct after smoke so the original initialization, buffers and RNG match.
    original.seed_all(SEED)
    f = original.clean_model(dict(scratch=True), root, contract, row, device)
    original.seed_all(SEED)
    write(root / 'fit-smoke.json', dict(status='PASS', clean_packets=2, augmented_packets=2, target_reads=0))

    steps_per_epoch = original.STEPS
    warm_steps = warmup * steps_per_epoch
    updates = EPOCHS * steps_per_epoch
    def lr_factor(step):
        if step < warm_steps:
            return .25 + .375 * (1 + cos(pi * step / warm_steps))
        progress = (step - warm_steps) / (updates - warm_steps)
        return 1 / 30 + (.25 - 1 / 30) * .5 * (1 + cos(pi * progress))

    optimizer = torch.optim.AdamW(f.parameters(), lr=row.get('lr', 3e-4), weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    rng = np.random.default_rng(721)
    ugen = torch.Generator(device=device).manual_seed(SEED)
    settings = dict(mechanism=original.terms(row), ordinary_std=row.get('ordinary_std', .015),
                    clean_anchor_weight=row.get('clean_anchor_weight', 0.),
                    rtl_weight=row.get('rtl_weight', 1.), pair_margin_weight=row.get('pair_margin_weight', 0.),
                    rtl_threshold=row.get('rtl_repl_threshold', .95))
    labeled_cfg = {k: settings[k] for k in ('rtl_weight', 'pair_margin_weight')}
    initial = original.clean_validation(f, v, vy, vr)
    write(out / 'initial.json', initial)
    best = None
    step = 0
    with (out / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, EPOCHS + 1):
            use_u = epoch > warmup
            order = torch.randperm(len(u), device=device, generator=ugen)[:steps_per_epoch * 16]
            u_batches = order.reshape(steps_per_epoch, 16) if use_u else (None,) * steps_per_epoch
            source_sat = satellite[(epoch - 1) % 2]
            totals = dict(clean_l=0., unlabeled=0., satellite_ce=0., accepted=0)
            for ids, pos in zip(original.batches(yn, rn, rng), u_batches, strict=True):
                i = tensor(ids, device)
                step += 1
                f.train()
                optimizer.zero_grad(set_to_none=True)
                if use_u:
                    clean_loss, lab, unlab, accepted = original._labeled_and_unlabeled(
                        f, x[i], y[i], rx[i], u[pos], ur[pos], step, True, **settings)
                else:
                    clean_loss = original.clean_source_ce(f(x[i])[0], y[i], rx[i], **labeled_cfg)
                    lab, unlab, accepted = clean_loss.detach(), 0., 0
                ce_sat = original.clean_source_ce(f(source_sat[i])[0], y[i], rx[i], **labeled_cfg)
                loss = clean_loss + AUG_WEIGHT * ce_sat
                loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                totals['clean_l'] += float(lab)
                totals['unlabeled'] += float(unlab)
                totals['satellite_ce'] += float(ce_sat.detach())
                totals['accepted'] += int(accepted)
            row_log = dict(epoch=epoch, step=step, satellite_batches=steps_per_epoch,
                           lr=optimizer.param_groups[0]['lr'], **totals)
            if epoch in (32, 48, 64, 80, 88, 96, *ELIGIBLE):
                metrics = original.clean_validation(f, v, vy, vr)
                row_log['V'] = metrics
                if eligible_checkpoint(epoch, metrics, best):
                    best = metrics | {'epoch': epoch}
                    torch.save(f.state_dict(), out / 'best.pt')
            log.write(json.dumps(row_log, allow_nan=False) + '\n')
            log.flush()
            print('W24_SAT_FIT', epoch, step, totals['satellite_ce'] / steps_per_epoch,
                  row_log.get('V', {}).get('accuracy'), flush=True)
    if step != updates or best is None:
        raise ValueError('Incomplete W24 satellite enhancement training budget')
    write(out / 'complete.json', dict(status='SOURCE_FROZEN', method=RUN, contract=contract,
          w24_recipe_commit=W24_COMMIT, seed=SEED, initialization='random_seed', epochs=EPOCHS,
          steps=step, frozen=best, clean_only=False, satellite_batches=step, satellite_source_role='L_s',
          satellite_weight=AUG_WEIGHT, target_reads=0, U_truth_reads=0))


def render(packets, models, device):
    raw, _, restored = recover(packets, models, device)
    return raw, restored


def evaluate(rt):
    from experiments.pamf_lora_main.tri_clean95 import NATIVE_WIDE_ROWS, clean_model

    activate(rt)
    root, device = Path(rt['run_root']), 'cuda:0'
    contract = source_contract(Path(rt['source_root']))
    artifact = read(root / 'F' / 'complete.json')
    if (artifact['status'] != 'SOURCE_FROZEN' or artifact['method'] != RUN or artifact['contract'] != contract
            or artifact['clean_only'] or artifact['satellite_batches'] != 9856
            or artifact['frozen']['epoch'] not in ELIGIBLE or artifact['target_reads']):
        raise ValueError('Frozen augmented W24 checkpoint does not match the registered training recipe')
    f = clean_model(dict(scratch=True), root, contract, NATIVE_WIDE_ROWS[METHOD], device)
    f.load_state_dict(torch.load(root / 'F' / 'best.pt', map_location=device, weights_only=True), strict=True)
    f.eval().requires_grad_(False)
    g = load_restorers(contract, root.parent, device)
    raw = native(Path(rt['source_root']), 'U_s', contract['counts']['U_s'])[:4]
    for wave in render(raw, g, device):
        classify(f, wave, device)
    write(root / 'smoke.json', dict(status='PASS', classifier_run=RUN, source_U_packets=4, target_reads=0))
    predict(rt, (f, *g), device, render=render, modes=MODES, classifier_run=RUN)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Original W24 source implementation required')
    source_runtime = root / 'source-runtime.json'
    write(source_runtime, {k: rt[k] for k in ('run_root', 'source_root', 'frozen_sources', 'clean_cache', 'cpu_workers')})
    write(root / 'state.json', dict(status='WAIT_GPU'))
    deadline = time.monotonic() + 3600
    while not (cards := available(rt['allowed_gpus'], max_compute_processes=rt['max_compute_processes'])):
        if time.monotonic() >= deadline:
            write(root / 'state.json', dict(status='FAILED', error='GPU wait expired'), exclusive=False)
            return
        time.sleep(30)
    gpu = next(i for i in rt['allowed_gpus'] if i in cards)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(reference))
    script = str(Path(__file__).resolve())
    for phase, config in (('fit', source_runtime), ('evaluate', runtime)):
        with (root / (phase + '.log')).open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([rt['python'], '-u', script, phase, str(config)], cwd=reference,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status='RUNNING', phase=phase, pid=proc.pid, gpu=gpu), exclusive=False)
            if proc.wait():
                write(root / 'state.json', dict(status='FAILED', error=phase, gpu=gpu), exclusive=False)
                return
    if read(root / 'predictions' / 'complete.json') != receipt(MODES, classifier_run=RUN):
        raise ValueError('Frozen W24 satellite matrix differs')
    write(root / 'state.json', dict(status='PREDICTIONS_FROZEN', gpu=gpu), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    for command in ('dispatch', 'fit', 'evaluate'):
        commands.add_parser(command).add_argument('runtime', type=Path)
    publish = commands.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    scoring = commands.add_parser('score')
    for key in ('predictions', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        context = read(args.context)
        rt = {k: context[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        native_data = project / 'releases' / f'{DATA_RUN}-release' / 'data'
        w24_root = project / 'runs' / W24_RUN
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN), source_root=str(native_data / 'source'),
                  native_target_root=str(native_data / 'target'), clean_cache=str(w24_root / 'clean_cache' /
                                                                                  '384-all8-nativeedges24'),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  allowed_gpus=[2, 3, 6, 5, 7, 0, 1, 4], max_compute_processes=8, cpu_workers=12)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'score':
        score(args.predictions, args.dataset, args.output, modes=MODES, classifier_run=RUN)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'fit':
        fit(read(args.runtime))
    else:
        evaluate(read(args.runtime))


if __name__ == '__main__':
    main()
