"""WiSig R17 three-direction real gain trained on source low-urban observations."""

import argparse
from pathlib import Path, PurePosixPath
import os
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, as_complex, wisig_models, wisig_real_prior
from experiments.pamf_lora_main.dura_real_gain_fusion import RealSpectralGain
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import load_gain, packet_scale, paired


RUN = '20261128-phase1-wisig-r17-amplitude-gain-s2026112905-r01'
URBAN_PARENT = '20261128-phase1-wisig-r17-urban-source-s2026112893-r01'
DOSE_SCREEN = '20261128-phase1-wisig-r17-gain-dose-s2026112901-r01'
PREVIOUS_SCREEN = '20261128-phase1-wisig-r17-full-source-s2026112889-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
SEED, STEPS, BATCH, RX = 2026112905, 2000, 24, 6
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'wisig_r17', 'real_gain', 'prior_fusion', 'urban_fusion', 'fusion')


def spectral_fusion(model, observed, prior):
    return prior + model(observed.unflatten(-1, (8, 32)), prior.unflatten(-1, (8, 32))).flatten(-2)


def setup(rt):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    project = Path(rt['project'])
    source, ids, r17 = wisig_models(project, rt['frozen_sources'])
    prior = wisig_real_prior(project)
    control = load_gain(project / 'runs' / PARENT, 'wisig', real=True)
    old_fusion = load_gain(project / 'runs' / PARENT, 'wisig')

    def real_gain(y):
        observed, estimate, scale = prior(y)
        return ((estimate + control(observed, estimate)) * scale).flatten(-2)

    return source, ids, r17, real_gain, old_fusion


def select_fresh_rows(ids, rx, day, previous_ids):
    result = []
    for value in (1, 2, 3):
        eligible = np.flatnonzero((rx == RX) & (day == value) & ~np.isin(ids, previous_ids))
        if len(eligible) < 100:
            raise ValueError('Insufficient source V packets outside previous screens')
        result.extend(np.random.default_rng(SEED + value).choice(eligible, 100, replace=False).tolist())
    return np.asarray(sorted(result), dtype=np.int64)


def urban_training_indices(unlabeled, quantile, scenes):
    if quantile.shape != scenes.shape:
        raise ValueError('Source observation metadata shapes differ')
    return np.intersect1d(unlabeled, np.flatnonzero((quantile >= .15) & (scenes == 5)))


def fit(rt):
    source, ids, r17, real_gain, _ = setup(rt)
    cache = Path(rt['project']) / 'runs/20261019-phase1-reception-restoration-selfsup-manysig-m6-r02/cache'
    observed_file = cache / 'epochs/01/full.npy'
    observed = np.load(observed_file, mmap_mode='r', allow_pickle=False)
    quantile = np.load(cache / 'epochs/01/quantile.npy', allow_pickle=False)
    scenes = np.load(cache / 'epochs/01/scene.npy', allow_pickle=False)
    if observed.shape != source.shape or quantile.shape != (len(source),):
        raise ValueError('Registered source full-channel observations do not match clean source order')
    active = urban_training_indices(ids, quantile, scenes)
    if not len(active):
        raise ValueError('No permitted U full-channel observations')
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = R17AmplitudeGain(32).to(DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        indices = rng.choice(active, BATCH, replace=False)
        y = as_complex(observed[indices])
        scale = packet_scale(y)
        y = y / scale
        clean = as_complex(source[indices]) / scale
        with torch.no_grad():
            p, action = r17(y)
            pc, actionc = r17(clean)
            r, rc = real_gain(y), real_gain(clean)
        output = model(y, p, r, action)
        preserved = model(clean, pc, rc, actionc)
        if step == 1:
            torch.testing.assert_close(output, p, rtol=0, atol=0)
        error = output - clean
        power = torch.fft.fft(clean, norm='ortho').abs().square()
        spectral = (torch.fft.fft(error, norm='ortho').abs().square() / (.03 + power)).log1p().mean()
        keep = (preserved - pc).abs().square().mean()
        retention = (output - p).abs().square().mean()
        loss = error.abs().square().mean() + .3 * spectral + 1.5 * keep + .1 * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = model.net[-1].weight.grad.flatten(1).norm(dim=1)
        if step == 1 and (not torch.isfinite(gradient).all() or not bool((gradient > 0).all())):
            raise ValueError('All three gain directions require finite nonzero source gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only loss')
        optimizer.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_AMPLITUDE_FIT', step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset='wisig', seed=SEED, updates=STEPS, batch=BATCH,
                 U_exposures=STEPS * BATCH, U_pool=len(active), F_reads=0, TX_reads=0, V_reads=0,
                 target_reads=0, parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    root = Path(rt['run_root'])
    with (root / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(root / 'fit.json', proof)


@torch.no_grad()
def predict(rt):
    import importlib
    project, root = Path(rt['project']), Path(rt['run_root'])
    source, source_ids, r17, real_gain, old_fusion = setup(rt)
    from leo_practical.channel import normalize_input
    from experiments.pamf_reception_restore.data import shared_pair
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    reference = project / 'releases/20261125-phase1-wisig-native-fir-source-m4-r01-release'
    config = read(reference / 'runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    checkpoint = torch.load(root / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', SEED, STEPS) or any(proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected frozen source-only candidate')
    model = R17AmplitudeGain(32).to(DEVICE).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'], strict=True)
    parent_root = project / 'runs' / URBAN_PARENT
    parent = torch.load(parent_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    parent_proof = read(parent_root / 'fit.json')
    if parent['proof'] != parent_proof or parent_proof['status'] != 'SOURCE_FROZEN' or any(
            parent_proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads')):
        raise ValueError('Expected source-only urban real-gain control')
    urban = RealSpectralGain(32).to(DEVICE).eval().requires_grad_(False)
    urban.load_state_dict(parent['weights'], strict=True)

    def classify(z):
        return np.asarray(original.classify(reader, z).argmax(1).cpu().tolist(), dtype=np.int16)

    def decisions(batch):
        y = as_complex(batch)
        scale = packet_scale(y)
        n = y / scale
        p, action = r17(n)
        r = real_gain(n)
        return np.stack([classify(z) for z in (y, r17(y)[0], r * scale,
                         old_fusion(n, p, action) * scale, spectral_fusion(urban, n, p) * scale,
                         model(n, p, r, action) * scale)])

    if decisions(source[source_ids[:2]]).shape != (len(MODES), 2):
        raise ValueError('No-query frozen checkpoint smoke failed')
    folder = project / 'data_caches/phase1_manysig_reference_20261019/source_cache/V'
    all_ids = np.load(folder / 'physical_ids.npy', mmap_mode='r', allow_pickle=False)
    rx = np.load(folder / 'rx_i.npy', mmap_mode='r', allow_pickle=False)
    day = np.load(folder / 'day_i.npy', mmap_mode='r', allow_pickle=False)
    raw = np.load(folder / 'iq.npy', mmap_mode='r', allow_pickle=False)
    old_ids = np.load(project / 'runs/20261019-phase1-reception-restoration-selfsup-manysig-m6-r02/cache/validation_full/physical_ids.npy', allow_pickle=False)
    # This source-V packet was exposed during the prelaunch channel smoke.
    with np.load(project / 'runs' / PREVIOUS_SCREEN / 'screen.npz', allow_pickle=False) as previous:
        old_ids = np.concatenate((old_ids, previous['ids'].copy(), np.asarray(['5fa0fd382eb8af0fbc4cc08b'])))
    for prior_run in (URBAN_PARENT, DOSE_SCREEN):
        with np.load(project / 'runs' / prior_run / 'screen.npz', allow_pickle=False) as previous:
            old_ids = np.concatenate((old_ids, previous['ids'].copy()))
    selected = select_fresh_rows(all_ids, rx, day, old_ids)
    clean = []
    for value in raw[selected]:
        z = normalize_input(value[0] + 1j * value[1])[0]
        clean.append(np.stack((z.real, z.imag)).astype(np.float32))
    clean = np.stack(clean)
    received = []
    for i, index in enumerate(selected):
        _, full, _ = shared_pair(clean[i], 'practical_low_urban', all_ids[index],
            f'rx{int(rx[index])}:day{int(day[index])}:eq1', 2026101901, 0, 2026101902,
            'restoration/validation/draw0')
        received.append(full)
    outputs = {}
    for view, values in (('clean', clean), ('practical_low_urban', np.stack(received))):
        outputs[view] = np.concatenate([decisions(values[i:i+48]) for i in range(0, len(selected), 48)], axis=1)
        print('WISIG_AMPLITUDE_SCREEN', view, len(selected), flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=all_ids[selected], **outputs)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=len(selected), source_RX=RX, excluded_previous_panel=True, source_truth_reads=0, target_reads=0))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root, project = Path(rt['run_root']), Path(rt['project'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), 300, 0):
        raise ValueError('Both views must freeze before source truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        predictions = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != 300 or any(value.shape != (len(MODES), 300) for value in predictions.values()):
        raise ValueError('Incomplete source predictions')
    folder = project / 'data_caches/phase1_manysig_reference_20261019/source_cache/V'
    all_ids = np.load(folder / 'physical_ids.npy', allow_pickle=False)
    all_labels = np.load(folder / 'labels.npy', allow_pickle=False)
    lookup = dict(zip(all_ids.tolist(), all_labels.tolist(), strict=True))
    labels = np.asarray([lookup[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(len(ids))
    hard, clean = predictions[VIEWS[-1]], predictions['clean']
    pairs = {name: paired(hard[-1], hard[i], labels, permutation, 2) for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((clean[-1] == labels).astype(np.int16) - (clean[i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(v['passes'] for v in pairs.values()) and min(clean_net.values()) >= -2
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=len(ids),
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in predictions.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote,
          missing_controls=['DURA v1 comparison deferred to formal evaluation'], formal_acceptance_complete=False,
          exposure='Physical IDs exclude our previous 2700 source V screens and one smoke packet; source V historically exposed.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        available = free_gpus()
        if not available:
            raise RuntimeError('No GPU with 8 GiB available')
        gpu = next(value for value in (7, 2, 3, 6, 1, 4, 0, 5) if value in available)
        for action in ('fit', 'predict', 'score'):
            device = None if action == 'score' else gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            args = [sys.executable, '-u', '-m', __spec__.name, action, str(runtime)]
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            print('WISIG_AMPLITUDE_WORKER', action, proc.pid, device, flush=True)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')
        write(state, dict(status='SOURCE_SCREEN_SCORED', gpu=gpu, promote=read(root / 'screen-score.json')['promote']),
              exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / PARENT), str(runs / PREVIOUS_SCREEN), str(runs / URBAN_PARENT), str(runs / DOSE_SCREEN),
                 str(releases / '20261125-phase1-wisig-native-fir-source-m4-r01-release')])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit', 'predict', 'score'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    elif action == 'dispatch':
        dispatch(**args)
    else:
        {'fit': fit, 'predict': predict, 'score': score}[action](read(args['runtime']))
