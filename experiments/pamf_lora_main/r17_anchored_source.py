"""Fit and score route-matched, fixed-dose direction corrections at N607."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .data import SOURCE, VIEWS
from .dura_repair import no_tone
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_lineage import OBSERVED_SHAPE_RUN, TRACE_RUN, frozen_g, load_utility
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .packet_physics import chirp_windows
from .packet_runner import schedule
from .publish import release, release_commit
from .r17_anchored import AnchoredDirection, direction_loss, trajectory
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .residual_dura_bandfit import band_masks, correction, fft, power
from .residual_dura_calibration import source_model, source_state
from .residual_dura_donormix import sixth_panel
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .runner import activate
from .spectral_unroll import EVAL_PER_RX as PREVIOUS_PER_RX, ROUTES, bind, observed


RUN = '20261123-phase1-lora-dura-r17-anchored-source-m2-r01'
STEPS = 700
BATCH = 64
EVAL_PER_RX = 200
MODES = ('direct', 'r69', 'r17', 'near_only', 'mid_only', 'both')


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(SEED)
    reference, contract, stream = bind(rt, route, device, pairs=True)
    reference.eval().requires_grad_(False)
    if not stream.ready():
        raise ValueError('Complete same-route L/U pairs required')
    projector = ObservedShape().to(device).eval()
    model = AnchoredDirection().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    order = np.random.default_rng(SEED + 11).permutation(len(stream))
    steps = 1 if smoke else STEPS
    for step, indices in enumerate(order[:steps * BATCH].reshape(steps, BATCH), 1):
        batch = stream.batch(indices, 1 + (step - 1) // (STEPS // 2))
        y, clean = (tensor(batch[key], device) for key in ('received', 'clean'))
        features, coordinates = trajectory(y, reference, projector)
        paired_shape = no_tone(projector.coordinates(clean)[0])
        desired = fft(paired_shape - no_tone(coordinates[0]))
        optimizer.zero_grad(set_to_none=True)
        proposal = model(features)
        loss = direction_loss(proposal, features[:, -1], desired).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite direction loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1 and (grad <= 0 or any(p.grad is not None for p in reference.parameters())):
            raise ValueError('Expected gradients in the direction head only')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            base_loss = direction_loss(features[:, -1], features[:, -1], desired).mean()
            print('ANCHORED_DIRECTION_STEP', route, step, float(loss.detach()), float(base_loss), float(grad), flush=True)
    root = Path(rt['run_root'])
    proof = dict(status='PASS' if smoke else 'SOURCE_FROZEN', route=route, seed=SEED, steps=steps,
                 F_reads=0, V_reads=0, TX_reads=0, target_reads=0)
    if smoke:
        write(root / f'smoke-{route}.json', proof)
    else:
        folder = root / route
        folder.mkdir(exist_ok=False)
        torch.save(dict(proof=proof, contract=contract, model=model.state_dict()), folder / 'final.pt')
        write(folder / 'complete.json', proof)


def load_model(root, route, contract, device):
    folder = root / route
    proof = dict(status='SOURCE_FROZEN', route=route, seed=SEED, steps=STEPS,
                 F_reads=0, V_reads=0, TX_reads=0, target_reads=0)
    saved = torch.load(folder / 'final.pt', map_location=device, weights_only=True)
    if saved['proof'] != proof or read(folder / 'complete.json') != proof or saved['contract'] != contract:
        raise ValueError('Frozen head contract or route differs')
    model = AnchoredDirection().to(device)
    model.load_state_dict(saved['model'], strict=True)
    return model.eval().requires_grad_(False)


def validation_indices(rx):
    panel = sixth_panel(rx).reshape(len(SOURCE), -1)
    all_slots = np.arange(panel.shape[1])
    previous = np.random.default_rng(SEED + 2).choice(all_slots, PREVIOUS_PER_RX, replace=False)
    chosen = np.random.default_rng(SEED + 13).choice(np.setdiff1d(all_slots, previous), EVAL_PER_RX, replace=False)
    return panel[:, np.sort(chosen)].flatten()


def proposals(iq, r17, r69, model, projector, gate):
    active = gate(iq)
    out = [iq, *(iq.clone() for _ in MODES[1:])]
    if not active.any():
        return torch.stack(out), 0, 0.
    y = iq[active].flatten(0, 1)
    spectra, coordinates = trajectory(y, r17, projector)
    anchor = spectra[:, -1]
    candidate = model(spectra)
    bands = band_masks(y.device)
    selected = [torch.where(band, candidate, anchor) for band in bands]
    # Match each band, with a separate check before time conversion.
    drift = max(float((power(candidate[:, band]).sqrt() - power(anchor[:, band]).sqrt()).abs().max())
                for band in bands)
    steps = [correction(projector, r69(y), coordinates, no_tone(coordinates[0])),
             *(torch.fft.ifft(x, norm='ortho') for x in (anchor, *selected, candidate))]
    for output, step in zip(out[1:], steps, strict=True):
        wave = step * coordinates[1] * coordinates[2]
        output[active] = (y + torch.stack((wave.real, wave.imag), dim=1)).reshape_as(iq[active])
    return torch.stack(out), int(active.sum()), drift


@torch.no_grad()
def score(rt, route):
    device = 'cuda:0'
    parent, _, contract, paired = source_state(rt)
    r17, train_contract, _ = bind(rt, route, device)
    root = Path(rt['run_root'])
    model = load_model(root, route, train_contract, device)
    if route == 'residual':
        r69 = source_model(parent, 'r69', paired, device)
        gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    else:
        r69 = frozen_g(parent.parent / OBSERVED_SHAPE_RUN, contract, device)
        full_gate = load_utility(parent.parent / TRACE_RUN, contract, SEED, device)
        gate = lambda iq: full_gate.impairment(iq).gt(0)
    projector = ObservedShape().to(device).eval()
    f = recognizer(device).eval().requires_grad_(False)
    ckpt = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    source = Path(rt['source_root'])
    iq, rx, ids, _ = source_role(source, contract, 'V')
    chosen = validation_indices(rx)
    rows = list(zip(iq[chosen], rx[chosen], ids[chosen], strict=True))
    metrics, predictions = {}, {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for view in VIEWS:
            inputs = (np.stack([chirp_windows(row[0]) for row in rows]) if view == 'clean' else
                      np.stack(list(pool.map(partial(observed, view, route), rows, chunksize=15))))
            parts, count, maximum = [], 0, 0.
            for y in tensor(inputs, device).split(16):
                wave, enabled, drift = proposals(y, r17, r69, model, projector, gate)
                logits = f(wave.flatten(0, 1))[0]
                if drift > 2e-6 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen readout or unmatched per-band dose')
                parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
                count += enabled
                maximum = max(maximum, drift)
            predictions[view] = np.concatenate(parts, axis=1)
            metrics[view] = dict(active=count, max_spectral_norm_difference=maximum)
            print('ANCHORED_SOURCE_PREDICTED', route, view, count, flush=True)
    path = root / f'{route}-source-predictions.json'
    write(path, {name: values.tolist() for name, values in predictions.items()})
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][chosen]
    hits = {name: value == truth for name, value in predictions.items()}
    for view, hit in hits.items():
        metrics[view].update(correct=hit.sum(1).tolist(),
                             by_rx=hit.reshape(len(MODES), len(SOURCE), EVAL_PER_RX).sum(-1).tolist())
    six = np.stack([hits[name] for name in VIEWS[1:]]).sum(0)
    low = hits['practical_low_urban'].astype(int)
    rng = np.random.default_rng(SEED + 19)
    draws = np.concatenate([rng.integers(i * EVAL_PER_RX, (i + 1) * EVAL_PER_RX, (1600, EVAL_PER_RX))
                            for i in range(len(SOURCE))], axis=1)
    intervals = {name: [np.quantile(delta[draws].sum(1), (.025, .975)).tolist() for delta in v - v[2]]
                 for name, v in (('six', six), ('low_urban', low))}
    write(root / f'{route}-source-scores.json', dict(status='SOURCE_SCORED', route=route, n=len(chosen),
          modes=list(MODES), metrics=metrics, six=six.sum(1).tolist(), paired_ci_vs_r17=intervals,
          target_reads=0, truth_after_predictions=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    launch = partial(start, module=rt['runner'])
    try:
        for action in ('smoke', 'fit', 'score'):
            write(state, dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, route) for route in ROUTES], launch)
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Matching residual parent release required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2,
              max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'fit', 'score'):
        command = commands.add_parser(kind)
        command.add_argument('runtime', type=Path)
        if kind != 'dispatch':
            command.add_argument('route', choices=ROUTES)
    options = vars(parser.parse_args())
    kind = options.pop('kind')
    if kind == 'publish':
        publish(**options)
    else:
        rt = read(options['runtime'])
        if kind == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if kind == 'score':
                score(rt, options['route'])
            else:
                fit(rt, options['route'], smoke=kind == 'smoke')
