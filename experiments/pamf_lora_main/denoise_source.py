"""Fit LoRa DURA's additive-noise stage before any channel inversion."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.denoise import Denoise, noise_loss

from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, real_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .joint_unroll_source import coordinates, fit_arrays, source_control, source_validation
from .native_detail import native_pair
from .packet_physics import chirp_windows, residual_view
from .packet_runner import schedule
from .physical_inverse_source import PER_RX as INVERSE_PER_RX, select_panel
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-noise-supervised-source-m2-r01'
KEY = 2026112617
FIT_PER_RX, HELD_PER_RX, EVAL_PER_RX = 650, 80, 180
BATCH, STEPS = 12, 1500
ARMS = ('local', 'packet')
MODES = ('direct', 'reference', 'local', 'packet', 'packet_half', 'known_noise')
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


def risk(model, arrays, indices, arm, *, identity=False):
    input_wave, q, epsilon, clean, clean_q = (x[indices] for x in arrays)
    if identity:
        input_wave[-2:] = clean[-2:, None]
        q[-2:] = clean_q[-2:, None]
        epsilon[-2:] = 0.
    y, q, epsilon = (x.flatten(0, 1) for x in (input_wave, q, epsilon))
    return noise_loss(model(y, q, packet=arm == 'packet'), epsilon, y)


def record(route, contract):
    return dict(status='SOURCE_FROZEN', route=route, contract=contract, seed=KEY, steps=STEPS,
                fit_per_rx=FIT_PER_RX, held_per_rx=HELD_PER_RX, arms=list(ARMS), **ISOLATION)


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(KEY)
    count, hold = (5, 3) if smoke else (FIT_PER_RX, HELD_PER_RX)
    tracker = PacketRestorer('consensus').to(device).eval()
    arrays, held, contract = fit_arrays(rt, route, tracker, count, hold)
    models = {arm: Denoise().to(device) for arm in ARMS}
    models['packet'].load_state_dict(models['local'].state_dict())
    optimizer = torch.optim.AdamW([p for model in models.values() for p in model.parameters()],
                                  lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(KEY)
    steps = 1 if smoke else STEPS
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-5)
    for step in range(1, steps + 1):
        selected = torch.randint(len(arrays[0]), (BATCH,), device=device, generator=generator)
        values = [risk(model, arrays, selected, arm, identity=True) for arm, model in models.items()]
        values = torch.stack([torch.stack(pair) for pair in values])
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite LoRa noise reconstruction objective')
        optimizer.zero_grad(set_to_none=True)
        values[:, 0].sum().backward()
        for model in models.values():
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and norm <= 0:
                raise ValueError('Expected a source waveform gradient in both noise models')
        optimizer.step()
        lr.step()
        if step == 1 or step % 50 == 0:
            print('DENOISE_SOURCE_FIT', route, step, *values.detach().flatten().tolist(), flush=True)
    diagnostics = {}
    with torch.no_grad():
        for arm, model in models.items():
            model.eval()
            parts = [torch.stack(risk(model, held, i, arm)) * len(i)
                     for i in torch.arange(len(held[0]), device=device).split(BATCH)]
            diagnostics[arm] = (torch.stack(parts).sum(0) / len(held[0])).tolist()
    root = Path(rt['run_root'])
    proof = record(route, contract)
    report = {k: v for k, v in proof.items() if k != 'contract'}
    report.update(held_waveform_error=diagnostics)
    if smoke:
        write(root / f'smoke-{route}.json', dict(report, status='PASS', steps=steps,
                                                fit_per_rx=count, held_per_rx=hold))
    else:
        torch.save(dict(proof=proof, models={k: model.state_dict() for k, model in models.items()}),
                   root / f'{route}-model.pt')
        write(root / f'{route}-complete.json', report)


def panel(rx):
    excluded = np.union1d(source_validation(rx), select_panel(rx, INVERSE_PER_RX))
    rng = np.random.default_rng(KEY + 1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             EVAL_PER_RX, replace=False)) for name in SOURCE])


def replay(route, scene, row):
    iq, rx, identity = row
    if scene == 'clean':
        wave = chirp_windows(iq)
        return wave, np.zeros_like(wave)
    if route == 'full':
        wave, _, noise = native_pair((iq, rx, identity, scene, 'source-V', None), sidebands=False)
        return wave, noise
    wave, signal, noise = residual_view(iq, scene, rx, identity, 'source-V', components=True)
    if not np.allclose(wave[:, 0] + 1j * wave[:, 1], signal + noise, atol=8e-7, rtol=2e-6):
        raise ValueError('Source V residual noise trace does not reproduce the observation')
    return wave, np.stack((noise.real, noise.imag), axis=1).astype(np.float32)


def normalized(wave, original):
    dims = (-3, -2, -1)
    ratio = original.square().sum(dims, keepdim=True) / wave.square().sum(dims, keepdim=True).clamp_min(1e-7)
    return wave * ratio.sqrt()


def readouts(iq, known_noise, baseline, tracker, models, packet_arms=(False, True)):
    reference, active = baseline(iq)
    outputs = [iq.clone() for _ in MODES[2:]]
    if active.any():
        y = iq[active]
        working, phase, q, rms = coordinates(tracker, y)
        guesses = [real_iq(model(working, q, packet=context)*phase*rms)
                   for model, context in zip(models, packet_arms, strict=True)]
        for arm, noise in zip(outputs, (*guesses, .5*guesses[1], known_noise[active]), strict=True):
            arm[active] = normalized(y-noise, y)
    return torch.stack((iq, reference, *outputs)), int(active.sum())


def summarize(root, route, source, chosen, ids, predictions, metrics, modes=MODES, *,
              predictions_written=False, comparisons=()):
    if not predictions_written:
        write(root / f'{route}-predictions.json', dict(ids=ids[chosen].tolist(),
              rows={k: v.tolist() for k, v in predictions.items()}))
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        if not np.array_equal(metadata['ids'][chosen], ids[chosen]):
            raise ValueError('Unexpected LoRa source truth order')
        truth = metadata['y'][chosen]
    hits = {k: v == truth for k, v in predictions.items()}
    for view, correct in hits.items():
        base = correct[1]
        metrics[view].update(correct=correct.sum(1).tolist(),
            repairs=(correct & ~base).sum(1).tolist(), harms=(~correct & base).sum(1).tolist(),
            by_rx=correct.reshape(len(modes), len(SOURCE), EVAL_PER_RX).sum(-1).tolist())
    six = np.stack([hits[k] for k in VIEWS[1:]]).sum(0)
    rng = np.random.default_rng(KEY+3)
    draws = np.concatenate([rng.integers(i*EVAL_PER_RX, (i+1)*EVAL_PER_RX, (1600, EVAL_PER_RX))
                            for i in range(len(SOURCE))], axis=-1)
    ci = {k: [np.quantile(delta[draws].sum(1), (.025, .975)).tolist() for delta in h-h[1]]
          for k, h in (('six', six), ('low', hits[VIEWS[-1]].astype(int)))}
    paired = {}
    for arm, base in comparisons:
        a, b = modes.index(arm), modes.index(base)
        paired[f'{arm}_vs_{base}'] = dict(
            six_net_by_rx=(six[a]-six[b]).reshape(len(SOURCE), EVAL_PER_RX).sum(1).tolist(),
            views={view: dict(repairs=int((hit[a] & ~hit[b]).sum()), harms=int((~hit[a] & hit[b]).sum()))
                   for view, hit in hits.items()},
            ci={k: np.quantile((h[a]-h[b])[draws].sum(1), (.025, .975)).tolist()
                for k, h in (('six', six), ('low', hits[VIEWS[-1]].astype(int)))})
    write(root / f'{route}-source-scores.json', dict(status='SOURCE_SCORED', route=route,
          n=len(chosen), modes=list(modes), metrics=metrics, six=six.sum(1).tolist(),
          paired_ci_vs_reference=ci, **({'paired_comparisons': paired} if paired else {}),
          target_reads=0, truth_after_predictions=True))


@torch.no_grad()
def score_models(root, route, binding, models, *, selection=panel, modes=MODES,
                 packet_arms=(False, True), device='cuda:0'):
    source, cache, contract, baseline, tracker = binding
    f = recognizer(device).eval().requires_grad_(False)
    f_path = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(f_path, map_location=device, weights_only=True), strict=True)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = selection(rx)
    rows = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    official = np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False) if cache is not None else None
    metrics, predictions = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for j, view in enumerate(VIEWS):
            y, epsilon = (np.stack(values) for values in zip(*pool.map(partial(replay, route, view),
                                                                     rows, chunksize=8), strict=True))
            if official is not None and not np.array_equal(y, official[j, chosen]):
                raise ValueError('Source V waveform replays differ from the registered full route')
            parts, n_active = [], 0
            for wave, noise in zip(tensor(y, device).split(24), tensor(epsilon, device).split(24), strict=True):
                result, count = readouts(wave, noise, baseline, tracker, models, packet_arms)
                logits = f(result.flatten(0, 1))[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite frozen Tri-set')
                parts.append(logits.argmax(1).reshape(len(modes), len(wave)).cpu().tolist())
                n_active += count
            predictions[view] = np.concatenate(parts, axis=1)
            metrics[view] = dict(active=n_active)
            print('DENOISE_SOURCE_PREDICTED', route, view, n_active, flush=True)
    summarize(root, route, source, chosen, ids, predictions, metrics, modes)


def score(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    binding = source_control(rt, route, device)
    proof = torch.load(root / f'{route}-model.pt', map_location=device, weights_only=True)
    if proof['proof'] != record(route, binding[2]) or set(proof['models']) != set(ARMS):
        raise ValueError('Wrong frozen noise model or source route')
    models = [Denoise().to(device).eval().requires_grad_(False) for _ in ARMS]
    for model, arm in zip(models, ARMS, strict=True):
        model.load_state_dict(proof['models'][arm], strict=True)
    score_models(root, route, binding, models, device=device)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        for action in ('smoke', 'fit', 'score'):
            write(state, dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, route) for route in ROUTES], partial(start, module=rt['runner']))
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as e:
        write(state, dict(status='FAILED', error=str(e)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    sub = read(parent_submit)
    if sub['archive_transfer'] != 'VERIFIED' or PurePosixPath(sub['run_root']) != parent:
        raise ValueError('Matching residual source release required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        p.add_argument('--' + option, type=Path, required=True)
    for cmd in ('dispatch', 'smoke', 'fit', 'score'):
        p = commands.add_parser(cmd)
        p.add_argument('runtime', type=Path)
        if cmd != 'dispatch':
            p.add_argument('route', choices=ROUTES)
    opts = vars(parser.parse_args())
    action = opts.pop('action')
    if action == 'publish':
        publish(**opts)
    else:
        rt = read(opts['runtime'])
        if action == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if action == 'score':
                score(rt, opts['route'])
            else:
                fit(rt, opts['route'], smoke=action == 'smoke')
