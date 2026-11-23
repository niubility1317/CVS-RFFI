"""Locate joint near-carrier recovery errors with paired direction substitutions."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .complex_vector_source import load_models, publish, score
from .data import SOURCE, VIEWS
from .denoise_components import bands
from .denoise_source import EVAL_PER_RX, normalized, replay
from .dura_repair import complex_iq, real_iq
from .full_packet_views import source_role
from .joint_unroll_source import coordinates, source_control
from .packet_subspace import target_vectors
from .packet_subspace_source import RECIPE, RUN as PARENT, panel as parent_panel
from .physical_inverse_source import dispatch, source_trace_implementation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_signal_probe_source import panel as probe_panel


RUN = '20261123-phase1-lora-dura-packet-subspace-cross-source-m2-r01'
MODES = ('direct', 'reference', 'learned', 'learned_half', 'mixed_heads', 'mixed_heads_half',
         'known_shared_direction', 'known_deviation_direction', 'known_joint_direction',
         'known_balance', 'known_full_dose', 'known_noise')


def physical_components(value, rms):
    weight = rms.square()
    center = (value*weight).sum(1, keepdim=True)/weight.sum(1, keepdim=True).clamp_min(1e-8)
    return center.expand_as(value), value-center


def band_norm(value, rms):
    # The receive phase has unit modulus, so Parseval includes just chirp RMS.
    return torch.linalg.vector_norm(value*rms, dim=(1, 2), keepdim=True)


def rotate_direction(reference, direction, rms):
    old, new = band_norm(reference, rms), band_norm(direction, rms)
    return torch.where(new>1e-8, direction*(old/new.clamp_min(1e-8)), reference)


def substitutes(learned, control, exact, rms):
    """Hold physical dose fixed independently in both near-carrier bands."""
    outputs = [learned.clone() for _ in range(5)]
    for band in bands(learned.shape[-1], learned.device):
        left, right = physical_components(learned[..., band], rms)
        known_left, known_right = physical_components(exact[..., band], rms)
        free_left = physical_components(control[..., band], rms)[0]
        corrected_left = rotate_direction(left, known_left, rms)
        corrected_right = rotate_direction(right, known_right, rms)
        a, b, c, d = (band_norm(v, rms) for v in (left, right, known_left, known_right))
        total = (a.square()+b.square()).sqrt()
        reference_total = (c.square()+d.square()).sqrt()
        valid = (a>1e-8)&(b>1e-8)&(reference_total>1e-8)
        scale = total/reference_total.clamp_min(1e-8)
        balance = left*(scale*c/a.clamp_min(1e-8)) + right*(scale*d/b.clamp_min(1e-8))
        changes = (rotate_direction(left, free_left, rms)+right, corrected_left+right,
                   left+corrected_right, corrected_left+corrected_right,
                   torch.where(valid, balance, learned[..., band]))
        for output, update in zip(outputs, changes, strict=True):
            output[..., band] = update
    return outputs


def panel(rx):
    used = np.union1d(parent_panel(rx), probe_panel(rx))
    rng = np.random.default_rng(2026112897)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), used),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


def directions(iq, noise, reference, tracker, models):
    observation, rotation, quality, rms = coordinates(tracker, iq)
    basis = rotation*rms
    base = complex_iq(reference)/basis
    control, learned = (m(observation, base, quality) for m in models)
    signal = observation-complex_iq(noise)/basis
    exact = target_vectors(base, signal)['factorized']
    variants = substitutes(learned, control, exact, rms)
    return learned, exact, basis, rms, variants


@torch.no_grad()
def render(iq, noise, baseline, tracker, models):
    reference, active = baseline(iq)
    outputs = [reference.clone() for _ in MODES[2:]]
    if active.any():
        y, eps = iq[active], noise[active]
        learned, exact, basis, _, variants = directions(y, eps, reference[active], tracker, models)
        mixed, *swaps = variants
        updates = learned, .5*learned, mixed, .5*mixed, *swaps, exact
        for wave, update in zip(outputs[:-1], updates, strict=True):
            wave[active] = reference[active]+real_iq(torch.fft.ifft(update, norm='ortho')*basis)
        outputs[-1][active] = normalized(y-eps, y)
    return torch.stack((iq, reference, *outputs)), int(active.sum())


@torch.no_grad()
def smoke(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    source, cache, contract, baseline, tracker = source_control(rt, route, device)
    parent = root.parent/PARENT
    if read(parent/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('The two source heads must be frozen and scored')
    models = load_models(parent, route, contract, RECIPE, device)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = panel(rx).reshape(len(SOURCE), EVAL_PER_RX)[:, :3].ravel()
    registered = None if cache is None else np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False)
    error, count = 0., 0
    for scene in ('clean', 'practical_low_urban'):
        wave, noise = (np.stack(values) for values in zip(
            *(replay(route, scene, row) for row in zip(raw[chosen], rx[chosen], ids[chosen], strict=True)),
            strict=True))
        if registered is not None and not np.array_equal(wave, registered[VIEWS.index(scene), chosen]):
            raise ValueError('Real source replay changed the existing observation')
        y, eps = (tensor(v, device) for v in (wave, noise))
        result, active = render(y, eps, baseline, tracker, models)
        if not torch.isfinite(result).all() or (scene=='clean' and active):
            raise ValueError('Invalid clean bypass or nonfinite source restoration')
        if scene=='practical_low_urban':
            reference, selected = baseline(y)
            count = int(selected.sum())
            if not count:
                raise ValueError('Need at least one genuine activated source observation')
            anchor, _, _, rms, variants = directions(y[selected], eps[selected], reference[selected], tracker, models)
            for mask in bands(anchor.shape[-1], device):
                fixed = band_norm(anchor[..., mask], rms)
                for update in variants:
                    error = max(error, float((band_norm(update[..., mask], rms)-fixed).abs().max()))
    if not count or error>1e-5:
        raise ValueError('Expected actual same-dose interventions')
    write(root/f'smoke-{route}.json', dict(status='PASS', route=route, active=count,
           dose_error=error, source_V_labels_read=0, F_reads=0, target_reads=0))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
    else:
        runtime = read(args['runtime'])
        if action == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            source_trace_implementation()
            if action == 'smoke':
                smoke(runtime, args['route'])
            else:
                score(runtime, args['route'], recipe=RECIPE, panel=panel,
                      modes=MODES, render=render, checkpoint_run=PARENT)
