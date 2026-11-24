"""Protect the observable private near-band direction of a frozen DURA readout."""
import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail_source import load_arm, read_stages
from .carrier_coupled import packet_frame
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_prior_width_source import RECIPES, RUN as WIDTH_RUN
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .private_transport_bands import named_band_masks
from .private_transport_source import dispatch
from .relative_carrier import mix
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-dura-private-projection-source-m2-r01'
MODES = ('direct', 'reference', 'wide_half', 'wide_quarter', 'near_off', 'radial_all', 'radial_observed', 'equal_near_edit')
PAIRS = tuple(('radial_observed', mode) for mode in MODES if mode not in ('radial_observed', 'direct', 'reference'))
RECIPE = RECIPES['wide']


def private(value):
    return value - value.mean(dim=1, keepdim=True)


def norm_squared(value):
    return value.abs().square().sum(dim=(1, 2), keepdim=True)


def protect_edits(observed, edit, mask):
    """Compare near-band edit directions at the same edit magnitude."""
    contrast = private(observed)
    axis = contrast * mask
    near = private(edit) * mask
    near_energy = norm_squared(axis)
    alias_energy = norm_squared(contrast * mask.roll(observed.size(-1)//2))
    reliability = ((near_energy - alias_energy) / near_energy.clamp_min(1e-10)).clamp(0., 1.)
    multiplier = (axis.conj() * near).real.sum(dim=(1, 2), keepdim=True) / near_energy.clamp_min(1e-10)
    contraction = multiplier.clamp_max(0.) * axis
    removed = reliability * contraction
    protected = near - removed
    near_edit_energy = norm_squared(near)
    scale = (norm_squared(protected) / near_edit_energy.clamp_min(1e-16)).clamp(0., 1.).sqrt()
    scale = torch.where(near_edit_energy > 1e-16, scale, 1.)
    # These differences are added to the original half-dose output.
    adjustments = (-near, -contraction, -removed, (scale - 1) * near)
    stats = torch.cat((reliability.flatten(1), scale.flatten(1), norm_squared(removed).sqrt().flatten(1)), dim=1)
    return adjustments, stats


def packet_reads(y, rotation, quality, rms, model):
    _, final = read_stages(model, y, quality, rotation, rms, RECIPE)
    half = mix(y, final, .5)
    mask = named_band_masks(y)[0]
    adjustments, stats = protect_edits(torch.fft.fft(y, norm='ortho'),
                                       torch.fft.fft(half - y, norm='ortho'), mask)
    waves = (half, mix(y, final, .25), *(half + torch.fft.ifft(a, norm='ortho') for a in adjustments))
    return waves, stats


def bind(root, contract, device):
    return load_arm(root, RECIPE, contract, device, {'wide': WIDTH_RUN})


@torch.no_grad()
def readouts(iq, tracker, model, threshold, *, evidence=None):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    count = int(active.sum())
    if not count:
        return (iq,) * (len(MODES) - 2), 0
    y, rotation, quality, rms = packet_frame(tuple(v[active] for v in frame))
    waves, stats = packet_reads(y, rotation, quality, rms, model)
    if evidence is not None:
        evidence.append(stats.cpu())
    rows = []
    for wave in waves:
        result = iq.clone()
        result[active] += real_iq((wave - y) * rotation * rms)
        rows.append(result)
    return tuple(rows), count


@torch.no_grad()
def render(iq, _unused, baseline, tracker, model, *, threshold, evidence=None):
    reference, _ = baseline(iq)
    waves, count = readouts(iq, tracker, model, threshold, evidence=evidence)
    return torch.stack((iq, reference, *waves)), count


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    model, threshold = bind(root, binding[2], device)
    evidence = []
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold, evidence=evidence),
          binding=binding, frozen_models=model, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze, reader_chunk=144)
    values = torch.cat(evidence, dim=0)
    write(root/f'{route}-gate.json', dict(active=len(values),
          labels=['parity_fraction', 'equal_edit_amplitude_scale', 'removed_frequency_norm'],
          quantiles=torch.quantile(values, torch.tensor([.1, .5, .9]), dim=0).tolist(),
          positive_fraction=(values>0).float().mean(0).tolist()))


def reserve(rt):
    def build(root, contract, device, baseline, tracker):
        model, threshold = bind(root, contract, device)
        return lambda iq: render(iq, None, baseline, tracker, model, threshold=threshold)
    predict_clean_reserve(rt, modes=MODES, bind=build)


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    _, _, matched, baseline, tracker = source_control(rt, 'full', device)
    if matched != contract:
        raise ValueError('Source IQ and frozen recognition input contracts differ')
    model, threshold = bind(root, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupt)[0]) > threshold
    iq = torch.cat((cal[:2], corrupt[active][:2]))
    waves, count = render(iq, None, baseline, tracker, model, threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (count != 2 or logits.shape != (len(MODES) * len(iq), 10)
            or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], iq[:2].expand_as(waves[2:, :2]))
            or torch.equal(waves[2, 2:], waves[4, 2:])):
        raise ValueError('Frozen DURA near-band intervention smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_U=len(iq), active=count, clean_bypassed=2, target_reads=0))
    print('LORA_DURA_PRIVATE_PROJECTION_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    model, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    pick = corruption_score(coordinates(tracker, corrupt)[0]).argmax().view(1)
    result = {}
    for name, iq in (('clean', cal[:1]), ('generic', corrupt[pick])):
        elapsed = []
        for _ in range(32):
            torch.cuda.synchronize()
            at = perf_counter()
            _, active = readouts(iq, tracker, model, threshold)
            torch.cuda.synchronize()
            elapsed.append(1000 * (perf_counter()-at))
        result[name] = dict(active=active, median_ms=float(np.median(elapsed[8:])))
    write(root/'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24,
          timing=result, parameters=sum(p.numel() for p in model.parameters()),
          scope='frozen wide DURA and all six comparison readouts, excludes historical DURA and classifier'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'timing', 'evaluate', 'score'):
        step = actions.add_parser(action)
        step.add_argument('runtime', type=Path)
        if action != 'dispatch':
            step.add_argument('row', choices=(*ROUTES,'reserve') if action=='evaluate' else ('source',))
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action=='publish':
        return publish(**options, run=RUN, runner=__spec__.name)
    rt = read(options['runtime'])
    if action=='dispatch':
        return dispatch(rt)
    if action=='score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if action=='smoke':
        return smoke(rt)
    if action=='timing':
        return timing(rt)
    if options['row']=='reserve':
        return reserve(rt)
    return evaluate(rt, options['row'])


if __name__ == '__main__':
    main()
