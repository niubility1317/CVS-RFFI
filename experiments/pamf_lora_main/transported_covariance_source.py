"""Evaluate matrix-free transport of received RF noise covariance on source LoRa."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .private_transport_source import dispatch, timing
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .transported_covariance import half_edits
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-transported-noise-covariance-source-m2-r01'
MODES = ('direct', 'reference', 'rank_half', 'transport_half', 'transport_same_norm')
PAIRS = (('transport_half', 'rank_half'), ('transport_same_norm', 'rank_half'),
         ('transport_same_norm', 'transport_half'))


@torch.no_grad()
def readouts(iq, tracker, model, threshold, *, errors=None):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return (iq, iq, iq), n
    local = tuple(v[active] for v in frame)
    *edits, residual = half_edits(*local, model)
    if errors is not None:
        errors.append(residual.flatten().cpu())
    _, rotation, _, rms = local
    result = []
    for edit in edits:
        wave = iq.clone()
        wave[active] += real_iq(edit * rotation * rms)
        result.append(wave)
    return tuple(result), n


@torch.no_grad()
def render(iq, _, baseline, tracker, models, *, threshold, errors=None):
    reference, _ = baseline(iq)
    waves, n = readouts(iq, tracker, models[0], threshold, errors=errors)
    return torch.stack((iq, reference, *waves)), n


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    model, threshold = parent_model(root, binding[2], device)
    errors = []
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold, errors=errors),
          binding=binding, frozen_models=(model,), require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze, reader_chunk=144)
    values = torch.cat(errors)
    finite = values.isfinite()
    write(root / f'{route}-solver.json', dict(active=len(values), fallback=int((~(values <= .02)).sum()),
          nonfinite=int((~finite).sum()), relative_residual_quantiles=torch.quantile(
              values[finite], torch.tensor([.50, .90, .99, 1.])).tolist()))


def reserve(rt):
    def bind(root, contract, device, baseline, tracker):
        model, threshold = parent_model(root, contract, device)
        return lambda iq: render(iq, None, baseline, tracker, (model,), threshold=threshold)
    predict_clean_reserve(rt, modes=MODES, bind=bind)


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, matched, baseline, tracker = source_control(rt, 'full', device)
    if matched != contract:
        raise ValueError('Source U and frozen recognizer contracts differ')
    model, threshold = parent_model(root, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026113627)
    corrupt = real_iq(colored_observation(complex_iq(calibration), rng))
    eligible = corruption_score(coordinates(tracker, corrupt)[0]) > threshold
    tested = corrupt[eligible][:2]
    y, rotation, quality, rms = coordinates(tracker, tested)
    direct, candidate, same_norm, error = half_edits(y, rotation, quality, rms, model)
    expected = torch.lerp(y, model(y, quality, rotation)[-1], .5) - y
    torch.testing.assert_close(direct, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close((same_norm * rms).abs().square().sum((-2, -1)),
                               (direct * rms).abs().square().sum((-2, -1)))
    iq = torch.cat((calibration[:2], tested))
    waves, count = render(iq, None, baseline, tracker, (model,), threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (count != 2 or not (error <= .02).all() or torch.equal(direct, candidate)
            or not torch.equal(waves[2:, :2], iq[:2].expand_as(waves[2:, :2]))
            or logits.shape != (len(MODES) * len(iq), 10) or not logits.isfinite().all()):
        raise ValueError('Real checkpoint transported covariance smoke failed')
    write(root / 'smoke.json', dict(status='PASS', active=count, clean_bypassed=2,
                                  maximum_relative_residual=float(error.max()), target_reads=0))
    print('LORA_TRANSPORTED_NOISE_REAL_CHECKPOINT_PASS', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    release = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        release.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'timing', 'evaluate', 'score'):
        step = actions.add_parser(action)
        step.add_argument('runtime', type=Path)
        if action != 'dispatch':
            step.add_argument('row', choices=(*ROUTES, 'reserve') if action == 'evaluate' else ('source',))
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        return publish(**args, run=RUN, runner=__spec__.name)
    rt = read(args['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    if action == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if action == 'smoke':
        return smoke(rt)
    if action == 'timing':
        return timing(rt, reader=readouts, timing_scope='one shared network proposal, diagonal and transported posterior readouts')
    if args['row'] == 'reserve':
        return reserve(rt)
    return evaluate(rt, args['row'])


if __name__ == '__main__':
    main()
