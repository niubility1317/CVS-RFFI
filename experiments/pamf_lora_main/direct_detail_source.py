"""Fit direct LoRa spectral corrections inside or outside a repeatable basis."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from . import repeatable_detail_source as prior
from .bypass_utility_source import ISOLATION
from .data import SOURCE
from .direct_detail import CoefficientHead, band_loss, coordinates, paired_target, physical_prediction, waveform
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .repeatable_detail import basis_spec, equal_dose, pack_spectrum
from .residual_dura import RUN as RESIDUAL_RUN
from .runner import activate


RUN = '20261123-phase1-lora-dura-direct-detail-source-m2-r01'
EPOCHS = 32
FIT_PER_RX = 600
HELD_PER_RX = 100
EVAL_PER_RX = 200
KEY = prior.KEY + 131
ARMS = ('unconstrained', 'subspace')
MODES = ('direct', 'reference', 'extra_r17', *ARMS, 'subspace_half_angle')


def source_basis(rt, route, device):
    root = Path(rt['run_root']).parent / prior.RUN
    return prior.load_basis(dict(rt, run_root=str(root)), route, device)


def proof(rt, route):
    contract, _ = prior.binding(rt, route)
    metadata = dict(status='SOURCE_FROZEN', route=route, seed=KEY, epochs=EPOCHS,
                    fit_per_rx=FIT_PER_RX, held_per_rx=HELD_PER_RX, basis_run=prior.RUN,
                    arms=list(ARMS), **ISOLATION)
    return contract, metadata


def constructors(basis):
    size, rank = basis.shape
    unconstrained = CoefficientHead(3 * size, size)
    subspace = CoefficientHead(3 * size, rank)
    subspace.encoder.load_state_dict(unconstrained.encoder.state_dict())
    subspace.context.load_state_dict(unconstrained.context.state_dict())
    return dict(unconstrained=unconstrained, subspace=subspace)


@torch.no_grad()
def collect(propose, tracker, arrays, indices, device):
    features, anchors, targets = [], [], []
    clean, *draws = arrays
    for draw, values in enumerate(draws, 1):
        count = 0
        for group in np.array_split(indices, 1 + len(indices) // 48):
            observed = tensor(values[group], device)
            ref, extra, active = propose(observed)
            if not active.any():
                continue
            inputs, anchor, info = coordinates(observed[active], ref[active], extra[active], tracker)
            truth = paired_target(tensor(clean[group], device)[active], anchor, info[0], tracker)
            features.append(inputs)
            anchors.append(anchor)
            targets.append(truth)
            count += len(inputs)
        print('DIRECT_DETAIL_PAIRED', draw, len(indices), count, flush=True)
    if not features:
        raise ValueError('Source practical gate produced no fitting pairs')
    out = tuple(torch.cat(rows) for rows in (features, anchors, targets))
    if not all(torch.isfinite(value).all() for value in out):
        raise ValueError('Invalid frozen paired source waveform coordinates')
    return out


def subspace_target(anchor, truth, basis):
    _, mask = basis_spec(anchor.device)
    return physical_prediction(pack_spectrum(truth, mask) @ basis, anchor, basis)


def fit(rt, route, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(KEY)
    contract, arrays = prior.fit_arrays(rt, route)
    expected, record = proof(rt, route)
    if contract != expected:
        raise ValueError('Frozen basis and source waveform contract mismatch')
    _, rx, _, _ = source_role(Path(rt['source_root']), contract, 'U_s')
    if any(a.shape != (len(rx), 6, 2, 256) or a.dtype != np.float32 for a in arrays):
        raise ValueError('Unexpected source paired waveform shape')
    count, held = (12, 6) if smoke else (FIT_PER_RX, HELD_PER_RX)
    rng = np.random.default_rng(KEY)
    groups = [rng.permutation(np.flatnonzero(rx == name))[:count + held] for name in SOURCE]
    fit_ids = np.concatenate([g[:count] for g in groups])
    held_ids = np.concatenate([g[count:] for g in groups])
    basis = source_basis(rt, route, device)
    propose, tracker = prior.route_predictor(rt, route, device)
    inputs, anchor, truth = collect(propose, tracker, arrays, fit_ids, device)
    hold = collect(propose, tracker, arrays, held_ids, device)
    matrices = dict(unconstrained=None, subspace=basis)
    label = dict(unconstrained=truth, subspace=subspace_target(anchor, truth, basis))
    models = {name: model.to(device) for name, model in constructors(basis).items()}
    optimizer = torch.optim.AdamW([p for net in models.values() for p in net.parameters()],
                                  lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(KEY)
    epochs = 1 if smoke else EPOCHS
    for epoch in range(1, epochs + 1):
        total = torch.zeros(len(ARMS), device=device)
        for rows in torch.randperm(len(inputs), generator=generator, device=device).split(128):
            losses = []
            for name, model in models.items():
                value = physical_prediction(model(inputs[rows]), anchor[rows], matrices[name])
                losses.append(band_loss(value, anchor[rows], label[name][rows]).mean())
            losses = torch.stack(losses)
            if not torch.isfinite(losses).all():
                raise FloatingPointError('Nonfinite direct correction objective')
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            for net in models.values():
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            if epoch == 1 and any(model.out.weight.grad.norm() <= 0 for model in models.values()):
                raise ValueError('Direct output must receive gradient at fixed dose')
            optimizer.step()
            total += len(rows) * losses.detach()
        print('DIRECT_DETAIL_FIT', route, epoch, *(total / len(inputs)).tolist(), flush=True)
    with torch.no_grad():
        xh, ah, th = hold
        matched_subspace = subspace_target(ah, th, basis)
        held_metrics = dict(pairs=len(xh), r17_to_paired=float(band_loss(ah, ah, th).mean()),
                            subspace_oracle_to_paired=float(band_loss(matched_subspace, ah, th).mean()))
        for name, model in models.items():
            model.eval()
            estimated = torch.cat([physical_prediction(model(x), b, matrices[name])
                                   for x, b in zip(xh.split(128), ah.split(128), strict=True)])
            held_metrics[name] = float(band_loss(estimated, ah, th).mean())
    root = Path(rt['run_root'])
    if smoke:
        write(root / f'smoke-{route}.json', dict(record, status='PASS', epochs=epochs,
              fit_per_rx=count, held_per_rx=held, held=held_metrics))
    else:
        torch.save(dict(proof=record, contract=contract,
                   weights={name: m.state_dict() for name, m in models.items()}), root / f'{route}.pt')
        write(root / f'{route}-complete.json', dict(record, fit_exposures=len(inputs), held=held_metrics))


def panel(rx):
    excluded = np.unique(np.concatenate((prior.validate_panel(rx), prior.anchor_panel(rx),
                                         prior.previous.validation_indices(rx))))
    rng = np.random.default_rng(KEY + 7)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             EVAL_PER_RX, replace=False)) for name in SOURCE])


def score(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    basis = source_basis(rt, route, device)
    expected, proof_record = proof(rt, route)
    state = torch.load(root / f'{route}.pt', map_location=device, weights_only=True)
    if state['contract'] != expected or state['proof'] != proof_record or set(state['weights']) != set(ARMS):
        raise ValueError('Matching route-specific direct direction weights required')
    models = constructors(basis)
    for name in ARMS:
        models[name].to(device).eval().requires_grad_(False)
        models[name].load_state_dict(state['weights'][name], strict=True)
    propose, tracker = prior.route_predictor(rt, route, device)

    def render(y):
        reference, extra, active = propose(y)
        result = [reference.clone() for _ in range(3)]
        drift = 0.
        if active.any():
            inputs, anchor, (_, rotation, rms) = coordinates(y[active], reference[active], extra[active], tracker)
            unconstrained = physical_prediction(models['unconstrained'](inputs), anchor, None)
            subspace = physical_prediction(models['subspace'](inputs), anchor, basis)
            bands, _ = basis_spec(anchor.device)
            blended = equal_dose(anchor, (anchor + subspace) * .5, bands)
            for output, spectrum in zip(result, (unconstrained, subspace, blended), strict=True):
                output[active] = waveform(reference[active], spectrum, rotation, rms)
                for band in bands:
                    norm = torch.linalg.vector_norm
                    drift = max(drift, float((norm(spectrum[..., band], dim=-1) -
                                               norm(anchor[..., band], dim=-1)).abs().max()))
        return torch.stack((y, reference, extra, *result)), int(active.sum()), drift

    prior.evaluate(rt, route, MODES, panel, render)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch = partial(start, module=rt['runner'])
    try:
        for action in ('smoke', 'fit', 'score'):
            write(root / 'state.json', dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, route) for route in prior.ROUTES], launch)
        write(root / 'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Registered LoRa source contract required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    commands.add_parser('dispatch').add_argument('runtime', type=Path)
    for kind in ('smoke', 'fit', 'score'):
        command = commands.add_parser(kind)
        command.add_argument('runtime', type=Path)
        command.add_argument('route', choices=prior.ROUTES)
    options = vars(parser.parse_args())
    kind = options.pop('kind')
    if kind == 'publish':
        publish(**options)
    else:
        runtime = read(options['runtime'])
        if kind == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            if kind == 'score':
                score(runtime, options['route'])
            else:
                fit(runtime, options['route'], smoke=kind == 'smoke')
