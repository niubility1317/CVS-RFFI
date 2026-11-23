"""Attribute recognition changes to frozen LoRa restoration edits in four bands."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import panel, proof, receive_only, setup
from .chirp_spectrum_source import freeze, read_frozen
from .complex_vector_source import load_reader, publish, score
from .data import SOURCE, VIEWS
from .denoise_source import EVAL_PER_RX, summarize
from .dura_repair import complex_iq, real_iq
from .full_packet_views import source_contract
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import MODES as PARENT_MODES, RECIPES, RUN as PARENT


RUN = '20261124-phase1-lora-tail-frequency-factorial-source-m2-r01'
BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))
COUNT = 1 << len(BANDS)
FAMILIES = {
    'direct_to_half': ('direct', *(f'half_{i:02x}' for i in range(1, COUNT))),
    'average_to_tail': tuple(f'swap_{i:02x}' for i in range(COUNT)),
}
MODES = ('direct', 'reference', *FAMILIES['direct_to_half'][1:], *FAMILIES['average_to_tail'])
ENDPOINTS = {
    'direct': 'direct', 'reference': 'reference', 'half_0f': 'tail_half',
    'swap_00': 'average', 'swap_0f': 'tail',
}
BOOTSTRAP_KEY = 2026112412


def band_path(low, high):
    """Apply the measured edit only at the selected dechirped Fourier bins."""
    if low.shape[-1] != 256 or low.shape != high.shape:
        raise ValueError('Expected paired 256-sample dechirped chirps')
    k = torch.arange(256, device=low.device)
    frequency = torch.minimum(k, 256 - k)
    masks = [(frequency >= first) & (frequency <= last) for first, last in BANDS]
    band_id = sum((1 << i) * keep for i, keep in enumerate(masks))
    delta = torch.fft.fft(high - low)
    yield low
    for mask in range(1, COUNT - 1):
        yield low + torch.fft.ifft(delta * ((band_id & mask) != 0))
    yield high


def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    observed, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed) > threshold
    if not bool(active.any()):
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, basis, q, scale = (part[active] for part in (observed, rotation, quality, rms))
    average, tail = (model(y, q, basis)[-1] for model in models)
    half = iter(band_path(y, torch.lerp(y, tail, .5)))
    next(half)
    values = [iq, reference]
    for path in (half, band_path(average, tail)):
        for wave in path:
            output = iq.clone()
            output[active] += real_iq((wave - y) * basis * scale)
            values.append(output)
    return torch.stack(values), int(active.sum())


def load_pairs(root, contract, device):
    parent = root.parent / PARENT
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed LoRa waveform-tail source run')
    models, thresholds = [], []
    for name, recipe in RECIPES.items():
        saved = torch.load(parent / f'{name}-model.pt', map_location=device, weights_only=True)
        if saved['proof'] != proof(contract, recipe):
            raise ValueError(f'Unexpected frozen LoRa restoration identity for {name}')
        model = recipe.model().to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['weights'], strict=True)
        models.append(model)
        thresholds.append(saved['threshold'])
    if thresholds[0] != thresholds[1] or not np.isfinite(thresholds[0]):
        raise ValueError('Expected a shared finite source U corruption threshold')
    return models, thresholds[0]


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, reference, tracker = source_control(rt, 'full', device)
    if expected != contract:
        raise ValueError('Source U and the frozen recognition checkpoint disagree')
    models, threshold = load_pairs(root, contract, device)
    reader = load_reader(root, contract, device)
    generator = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), generator))
    waves, active = render(batch, None, reference, tracker, models, threshold=-1.)
    logits = reader(waves.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Real LoRa no-query checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', variants=len(MODES), source_U=len(batch),
                                              threshold=threshold, target_reads=0))
    print('LORA_TAIL_BAND_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    binding = source_control(rt, route, 'cuda:0')
    models, threshold = load_pairs(Path(rt['run_root']), binding[2], 'cuda:0')
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold), binding=binding,
          frozen_models=models, replay_fn=receive_only, finalize=freeze, reader_chunk=144)


def shapley_twelfths(hits):
    """Return each band's exact contribution for every paired scene and packet."""
    values = hits.astype(np.int16)
    weights = (3, 1, 1, 3)
    effects = []
    for bit in (1 << i for i in range(len(BANDS))):
        effects.append(sum(weights[k.bit_count()] * (values[k | bit] - values[k])
                           for k in range(COUNT) if not k & bit))
    effects = np.stack(effects)
    if not np.array_equal(effects.sum(0), 12 * (values[-1] - values[0])):
        raise ValueError('Frequency contributions do not sum to the paired endpoint change')
    return effects


def analyze(hits, reference, draws):
    integer = hits.astype(np.int16)
    six = integer[:, 1:].sum(1)
    difference = six - reference[1:].sum(0)
    effects = shapley_twelfths(hits)
    six_effects = effects[:, 1:].sum(1)
    mask_intervals = [np.quantile(row[draws].sum(1), (.025, .975)).tolist() for row in difference]
    effect_intervals = [np.quantile(row[draws].sum(1) / 12, (.025, .975)).tolist() for row in six_effects]
    fixed = np.nonzero((difference.sum(1) > 0) & (hits[:, 0].sum(1) >= reference[0].sum()))[0]
    return dict(per_scene_correct=integer.sum(-1).tolist(),
                six_correct=six.sum(1).tolist(),
                six_net_vs_reference=difference.sum(1).tolist(),
                six_paired_interval_vs_reference=mask_intervals,
                six_by_rx=six.reshape(COUNT, len(SOURCE), EVAL_PER_RX).sum(-1).tolist(),
                beats_reference_with_clean=fixed.tolist(),
                per_scene_attribution=(effects.sum(-1) / 12).tolist(),
                six_attribution=(six_effects.sum(-1) / 12).tolist(),
                six_attribution_intervals=effect_intervals,
                six_both_correct=int((hits[0, 1:] & hits[-1, 1:]).sum()),
                six_endpoint_repairs=int((~hits[0, 1:] & hits[-1, 1:]).sum()),
                six_endpoint_harms=int((hits[0, 1:] & ~hits[-1, 1:]).sum()),
                six_oracle_correct=int(hits[:, 1:].any(axis=0).sum()))


def score_frozen(rt):
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    frozen = {route: read_frozen(root, route, MODES) for route in ROUTES}
    ids = frozen['full'][0]
    for route, (current_ids, rows, _) in frozen.items():
        previous_ids, previous, _ = read_frozen(root.parent / PARENT, route, PARENT_MODES)
        if current_ids != ids or current_ids != previous_ids:
            raise ValueError('Frequency and parent predictions do not cover the same LoRa packets')
        for view in VIEWS:
            for arm, parent_arm in ENDPOINTS.items():
                if not np.array_equal(rows[view][MODES.index(arm)], previous[view][PARENT_MODES.index(parent_arm)]):
                    raise ValueError(f'Frozen parent decisions disagree on {route}/{view}/{arm}')
    universe = np.asarray(source_contract(source)['role_ids']['V'])
    index = {identity: i for i, identity in enumerate(universe)}
    chosen = np.asarray([index[identity] for identity in ids])
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as data:
        if not np.array_equal(data['ids'][chosen], np.asarray(ids)):
            raise ValueError('Source truth IDs disagree with frozen predictions')
        if not np.array_equal(panel(np.asarray(SOURCE)[data['rx_i']]), chosen):
            raise ValueError('Unexpected source V attribution panel')
        truth = data['y'][chosen]
    rng = np.random.default_rng(BOOTSTRAP_KEY)
    draws = np.concatenate([rng.integers(i*EVAL_PER_RX, (i+1)*EVAL_PER_RX, (1600, EVAL_PER_RX))
                            for i in range(len(SOURCE))], axis=1)
    results = {}
    for route, (_, rows, metrics) in frozen.items():
        hits = np.stack([rows[view] == truth for view in VIEWS], axis=1)
        reference = hits[MODES.index('reference')]
        results[route] = {family: analyze(hits[[MODES.index(mode) for mode in modes]], reference, draws)
                          for family, modes in FAMILIES.items()}
        summarize(root, route, source, chosen, universe, rows, metrics, MODES,
                  predictions_written=True, comparisons=())
    shared = {name: sorted(set(results['full'][name]['beats_reference_with_clean'])
                           & set(results['residual'][name]['beats_reference_with_clean']))
              for name in FAMILIES}
    write(root / 'band-effects.json', dict(status='SOURCE_SCORED', parent=PARENT, target_reads=0,
          samples=len(ids), receivers=list(SOURCE), bands_khz=list(BANDS), views=list(VIEWS),
          masks={name: list(modes) for name, modes in FAMILIES.items()},
          endpoints_match=True, bootstrap_key=BOOTSTRAP_KEY, bootstrap_draws=1600,
          fixed_masks_beating_both=shared, routes=results))
    print('LORA_TAIL_BAND_SOURCE_SCORED', flush=True)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch = partial(start, module=rt['runner'])
    try:
        schedule(rt, [('checkpoint_smoke', 'source')], launch)
        schedule(rt, [('evaluate', route) for route in ROUTES], launch)
        if launch(rt, 'score', 'source').wait():
            raise RuntimeError('Independent LoRa frequency attribution scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root / 'state.json', dict(status='FAILED', error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'evaluate', 'score'):
        command = commands.add_parser(action)
        command.add_argument('runtime', type=Path)
        if action != 'dispatch':
            command.add_argument('name', choices=ROUTES if action == 'evaluate' else ('source',))
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
        return
    rt = read(options['runtime'])
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'score':
        score_frozen(rt)
    else:
        activate(rt)
        if action == 'evaluate':
            evaluate(rt, options['name'])
        else:
            checkpoint_smoke(rt)


if __name__ == '__main__':
    main()
