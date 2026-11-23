"""Evaluate jointly updated DURA waveforms and receiver action on two LoRa routes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.joint import JointUnroll, joint_loss

from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_restore_draw import DRAW, RUN as DRAW_RUN, load_draw
from .full_supervised import construct as recognizer, handles
from .native_detail import native_pair
from .native_detail_source import bind as full_control, evaluation_panel as native_panel
from .packet_physics import chirp_windows, residual_view
from .packet_runner import schedule
from .publish import release, release_commit
from .r17_anchored_source import validation_indices as anchored_panel
from .refine_runner import start
from .residual_dura import L_COUNT, RUN as RESIDUAL_RUN, SEED, ResidualView
from .residual_dura_calibration import source_model, source_state
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .residual_dura_stage_blend import source_readouts
from .runner import activate
from .spectral_unroll import ROUTES, observed


RUN = '20261123-phase1-lora-dura-joint-unroll-source-m2-r01'
KEY = 2026112581
STEPS = 1400
FIT_PER_RX = 400
HELD_PER_RX = 60
EVAL_PER_RX = 180
BATCH = 10
ARMS = ('single', 'iterative')
MODES = ('direct', 'reference', 'single', 'iterative', 'iterative_half')
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


def fit_replay(route, observation):
    iq, rx, identity, slot, alternate = observation
    scene = VIEWS[1 + (int(slot) + (route == 'residual' and alternate)) % 6]
    if route == 'full':
        y, _, noise = native_pair((iq, rx, identity, scene, 'source-U', DRAW if alternate else None),
                                 sidebands=False)
    else:
        y, signal, epsilon = residual_view(iq, scene, rx, identity, 'source-U', components=True,
                                            realization=1 if alternate else None)
        if not np.allclose(y[:, 0] + 1j * y[:, 1], signal + epsilon, atol=8e-7, rtol=2e-6):
            raise ValueError('Residual source trace does not explain the registered observation')
        noise = np.stack((epsilon.real, epsilon.imag), axis=1).astype(np.float32)
    return y, noise


@torch.no_grad()
def coordinates(tracker, received):
    framed, rotation, coherence, *_ = tracker.track(received)
    rms = framed.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return framed / rms, rotation, coherence.unsqueeze(-1), rms


@torch.no_grad()
def fit_arrays(rt, route, tracker, count, hold):
    parent = Path(rt['parent_root'])
    source, cache, contract = handles(dict(rt, parent_root=str(parent.parent / DATA_RUN)))
    iq, rx, identities, slots = source_role(source, contract, 'U_s')
    clean = np.load(cache / 'U_s.npy', mmap_mode='r', allow_pickle=False)[0]
    if route == 'full':
        draws = (np.load(cache / 'U_s_practical.npy', mmap_mode='r', allow_pickle=False),
                 load_draw(parent.parent / DRAW_RUN, 'U_s'))
    else:
        _, pairs, matched, _ = source_state(rt)
        stream = ResidualView(pairs)
        if matched != contract or not stream.ready():
            raise ValueError('The residual source replay cache does not match')
        draws = stream.received[:, L_COUNT:]
        if not np.array_equal(stream.clean[L_COUNT:], clean):
            raise ValueError('The two source routes have different paired clean IQ')
    rng = np.random.default_rng(KEY)
    selected = [rng.permutation(np.flatnonzero(rx == group))[:count + hold] for group in SOURCE]
    fitting = np.concatenate([group[:count] for group in selected])
    holdout = np.concatenate([group[count:] for group in selected])
    if len(fitting) != len(SOURCE)*count or len(holdout) != len(SOURCE)*hold:
        raise ValueError('Insufficient receiver-balanced U replay cohort')
    result = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for cohort in (fitting, holdout):
            truth, _, truth_q, _ = coordinates(tracker, tensor(clean[cohort], 'cuda:0'))
            pair = []
            for alternate, expected in enumerate(draws):
                jobs = ((iq[i], rx[i], identities[i], slots[i], bool(alternate)) for i in cohort)
                rows = list(pool.map(partial(fit_replay, route), jobs, chunksize=8))
                y, noise = (np.stack(values) for values in zip(*rows, strict=True))
                if not np.array_equal(y, expected[cohort]) or not np.isfinite(noise).all():
                    raise ValueError('Source trace must replay the exact cached receiver windows')
                obs, rotation, coherence, rms = coordinates(tracker, tensor(y, 'cuda:0'))
                error = complex_iq(tensor(noise, 'cuda:0')) * rotation.conj() / rms
                pair.append((obs, coherence, error))
                print('JOINT_REPLAY_READY', route, len(cohort), alternate, flush=True)
            result.append((*[torch.stack(parts, dim=1) for parts in zip(*pair, strict=True)], truth, truth_q))
    return (*result, contract)


def batch_loss(model, arrays, indices, arm, *, identity=False):
    y, coherence, epsilon, truth, truth_q = (x[indices] for x in arrays)
    clean = truth.unsqueeze(1).expand_as(y).flatten(0, 1)
    if identity:
        # Same packets and identity exposures for both arms.
        y[-2:] = truth[-2:, None]
        coherence[-2:] = truth_q[-2:, None]
        epsilon[-2:] = 0.
    y, coherence, epsilon = (x.flatten(0, 1) for x in (y, coherence, epsilon))
    return joint_loss(model(y, coherence, iterative=arm == 'iterative'), clean, epsilon, y)


def record(route, contract):
    return dict(status='SOURCE_FROZEN', route=route, contract=contract, seed=KEY, steps=STEPS,
                fit_per_rx=FIT_PER_RX, held_per_rx=HELD_PER_RX, arms=list(ARMS), **ISOLATION)


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(KEY)
    count, hold = (5, 3) if smoke else (FIT_PER_RX, HELD_PER_RX)
    tracker = PacketRestorer('consensus').to(device).eval()
    inputs, held, contract = fit_arrays(rt, route, tracker, count, hold)
    models = {arm: JointUnroll().to(device) for arm in ARMS}
    models['iterative'].load_state_dict(models['single'].state_dict())
    optimizer = torch.optim.AdamW([p for model in models.values() for p in model.parameters()],
                                  lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(KEY)
    steps = 1 if smoke else STEPS
    for step in range(1, steps + 1):
        selected = torch.randint(len(inputs[0]), (BATCH,), device=device, generator=generator)
        values = [batch_loss(model, inputs, selected, arm, identity=True) for arm, model in models.items()]
        losses = torch.stack([v[0] for v in values])
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Source unroll losses are nonfinite')
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        for model in models.values():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and any(block.out.weight.grad.norm() <= 0 for block in model.steps):
                raise ValueError('Missing physical refinement gradient in a stage')
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print('JOINT_UNROLL_FIT', route, step, *losses.detach().tolist(),
                  *(float(v[1].detach()) for v in values), flush=True)
    diagnostics = {}
    with torch.no_grad():
        for arm, model in models.items():
            model.eval()
            parts = [torch.stack(batch_loss(model, held, i, arm)) * len(i)
                     for i in torch.arange(len(held[0]), device=device).split(BATCH)]
            diagnostics[arm] = (torch.stack(parts).sum(0) / len(held[0])).tolist()
    root = Path(rt['run_root'])
    proof = record(route, contract)
    report = {k: v for k, v in proof.items() if k != 'contract'}
    report.update(held_error=diagnostics)
    if smoke:
        write(root / f'smoke-{route}.json', dict(report, status='PASS', steps=steps,
                                                fit_per_rx=count, held_per_rx=hold))
    else:
        torch.save(dict(proof=proof, models={k: model.state_dict() for k, model in models.items()}),
                   root / f'{route}-model.pt')
        write(root / f'{route}-complete.json', report)


def source_validation(rx):
    excluded = np.union1d(native_panel(rx), anchored_panel(rx))
    rng = np.random.default_rng(KEY + 7)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             EVAL_PER_RX, replace=False)) for name in SOURCE])


def source_control(rt, route, device):
    if route == 'full':
        source, cache, contract, control, tracker = full_control(rt, device)
        return source, cache, contract, control, tracker
    parent, _, contract, paired = source_state(rt)
    r69 = source_model(parent, 'r69', paired, device).eval().requires_grad_(False)
    gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    projection = ObservedShape().to(device).eval()

    def control(iq):
        waves, _, _ = source_readouts(iq, r69, gate, projection)
        return waves[1], gate(iq)

    return Path(rt['source_root']), None, contract, control, tracker


@torch.no_grad()
def readouts(iq, baseline, tracker, models):
    reference, active = baseline(iq)
    candidates = [iq.clone() for _ in MODES[2:]]
    if active.any():
        subset = iq[active]
        measured, rotation, coherence, rms = coordinates(tracker, subset)
        for model, iterative, output in zip(models, (False, True), candidates[:2], strict=True):
            predicted = model(measured, coherence, iterative=iterative)[-1][0]
            delta = (torch.fft.ifft(predicted, norm='ortho') - measured) * rotation * rms
            output[active] = subset + real_iq(delta)
        candidates[2][active] = subset + .5 * (candidates[1][active] - subset)
    return torch.stack((iq, reference, *candidates)), int(active.sum())


def score_predictions(root, source, chosen, ids, predictions, diagnostics):
    write(root, dict(ids=ids[chosen].tolist(), rows={k: v.tolist() for k, v in predictions.items()}))
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as labels:
        if not np.array_equal(labels['ids'][chosen], ids[chosen]):
            raise ValueError('Frozen source prediction IDs do not match the source truth row')
        truth = labels['y'][chosen]
    hits = {name: output == truth for name, output in predictions.items()}
    for name, value in hits.items():
        ref = value[1]
        diagnostics[name].update(correct=value.sum(-1).tolist(),
                                 repairs=(value & ~ref).sum(-1).tolist(), harms=(~value & ref).sum(-1).tolist(),
                                 by_rx=value.reshape(len(MODES), len(SOURCE), EVAL_PER_RX).sum(-1).tolist())
    six = np.stack([hits[name] for name in VIEWS[1:]]).sum(0)
    rng = np.random.default_rng(KEY + 8)
    draws = np.concatenate([rng.integers(i * EVAL_PER_RX, (i+1) * EVAL_PER_RX, (1600, EVAL_PER_RX))
                            for i in range(len(SOURCE))], axis=1)
    intervals = {name: [np.quantile(row[draws].sum(1), (.025, .975)).tolist() for row in values - values[1]]
                 for name, values in (('six', six), ('low', hits[VIEWS[-1]].astype(int)))}
    return six.sum(1).tolist(), intervals


@torch.no_grad()
def score(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    source, cache, contract, baseline, tracker = source_control(rt, route, device)
    saved = torch.load(root / f'{route}-model.pt', map_location=device, weights_only=True)
    if saved['proof'] != record(route, contract) or set(saved['models']) != set(ARMS):
        raise ValueError('Wrong frozen source joint recovery model or data contract')
    models = [JointUnroll().to(device).eval().requires_grad_(False) for _ in ARMS]
    for arm, model in zip(ARMS, models, strict=True):
        model.load_state_dict(saved['models'][arm], strict=True)
    f = recognizer(device).eval().requires_grad_(False)
    weights = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = source_validation(rx)
    official = np.load(cache / 'V.npy', mmap_mode='r', allow_pickle=False) if route == 'full' else None
    physical = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    predictions, diagnostics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, scene in enumerate(VIEWS):
            if route == 'full':
                values = official[index, chosen]
            elif index == 0:
                values = np.stack(list(pool.map(chirp_windows, raw[chosen], chunksize=12)))
            else:
                values = np.stack(list(pool.map(partial(observed, scene, route), physical, chunksize=12)))
            parts, n_active = [], 0
            for batch in tensor(values, device).split(18):
                wave, active = readouts(batch, baseline, tracker, models)
                logits = f(wave.flatten(0, 1))[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite frozen Tri-set predictions')
                parts.append(logits.argmax(1).reshape(len(MODES), len(batch)).cpu().tolist())
                n_active += active
            predictions[scene] = np.concatenate(parts, axis=1)
            diagnostics[scene] = dict(active=n_active)
            print('JOINT_SOURCE_PREDICTED', route, scene, n_active, flush=True)
    six, intervals = score_predictions(root / f'{route}-source-predictions.json', source, chosen, ids,
                                       predictions, diagnostics)
    write(root / f'{route}-source-scores.json', dict(status='SOURCE_SCORED', route=route, n=len(chosen),
          modes=list(MODES), metrics=diagnostics, six=six, paired_ci_vs_reference=intervals,
          target_reads=0, truth_after_predictions=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    launch = partial(start, module=rt['runner'])
    try:
        for phase in ('smoke', 'fit', 'score'):
            write(state, dict(status=phase.upper()), exclusive=False)
            schedule(rt, [(phase, route) for route in ROUTES], launch)
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    config = read(context)
    rt = {k: config[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Use the matched residual and native source release')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5],
              max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for command in ('dispatch', 'smoke', 'fit', 'score'):
        tool = commands.add_parser(command)
        tool.add_argument('runtime', type=Path)
        if command != 'dispatch':
            tool.add_argument('route', choices=ROUTES)
    options = vars(parser.parse_args())
    command = options.pop('command')
    if command == 'publish':
        publish(**options)
    else:
        runtime = read(options['runtime'])
        if command == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            if command == 'score':
                score(runtime, options['route'])
            else:
                fit(runtime, options['route'], smoke=command == 'smoke')
