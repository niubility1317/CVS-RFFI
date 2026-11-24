"""Screen correlations between common and differential LoRa detail on source packets."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score, zero_carrier
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .joint_mode_covariance import (
    chirp_basis, decode, encode, match_rank_direction, moments, noise_covariance, noise_edit,
    paired_bins, without_cross_covariance,
)
from .packet_runner import schedule
from .private_transport_source import timing
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard_source import parent_model
from .weak_signal_probe import observed_noise_rms


RUN = '20261124-phase1-lora-joint-mode-covariance-source-m2-r01'
MODES = ('direct', 'reference', 'rank_half', 'separate_half', 'joint_half',
         'rank_separate_matched', 'rank_joint_matched')
PAIRS = (('joint_half', 'separate_half'), ('rank_joint_matched', 'rank_separate_matched'),
         ('rank_joint_matched', 'rank_half'), ('rank_separate_matched', 'rank_half'))


@torch.no_grad()
def fit_moments(packets, tracker):
    parts = []
    for batch in packets.split(80):
        y = coordinates(tracker, batch)[0]
        spectrum = torch.fft.fft(y, norm='ortho')
        parts.append(encode(spectrum, chirp_basis(y), paired_bins(y.device)))
    return moments(torch.cat(parts))


@torch.no_grad()
def fit(rt):
    (training, _, _), contract = setup(rt, 'cuda:0')
    root = Path(rt['run_root'])
    tracker = PacketRestorer('consensus').cuda().eval()
    started = perf_counter()
    prior = fit_moments(training, tracker)
    torch.linalg.cholesky(prior[1])
    torch.cuda.synchronize()
    with (root / 'source-moments.pt').open('xb') as stream:
        torch.save(dict(contract=contract, mean=prior[0].cpu(), covariance=prior[1].cpu()), stream)
    write(root / 'fit.json', dict(status='SOURCE_FROZEN', source_U=len(training),
          scalar_statistics=sum(x.numel() for x in prior), trainable_parameters=0,
          seconds=perf_counter() - started, target_reads=0, F_reads=0, TX_reads=0))


def load_moments(root, contract, device):
    if read(root / 'fit.json')['status'] != 'SOURCE_FROZEN':
        raise ValueError('Expected frozen source U moments')
    result = torch.load(root / 'source-moments.pt', map_location=device, weights_only=True)
    if result['contract'] != contract:
        raise ValueError('Joint covariance and evaluation show different source contracts')
    return result['mean'], result['covariance']


def candidate_edits(y, rot, quality, rms, model, prior):
    bins, basis = paired_bins(y.device), chirp_basis(y)
    spectrum = torch.fft.fft(y, norm='ortho')
    measurement = encode(spectrum, basis, bins)
    variance = observed_noise_rms(y).square().squeeze(-1).clamp_min(1e-7)
    noise = noise_covariance(variance, basis, bins)
    mean, covariance = prior
    independent = without_cross_covariance(covariance, bins)
    proposals = [decode(.5 * noise_edit(measurement, mean, signal, noise), basis, bins, spectrum)
                 for signal in (independent, covariance)]

    parent = torch.lerp(y, model(y, quality, rot)[-1], .5) - y
    parent_spectrum = torch.fft.fft(parent, norm='ortho')
    matched = [match_rank_direction(parent_spectrum, v, rms) for v in proposals]
    return (parent, *(torch.fft.ifft(v, norm='ortho') for v in (*proposals, *matched)))


@torch.no_grad()
def readouts(iq, tracker, model, threshold, *, prior):
    y, rot, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    if not count:
        return [iq] * (len(MODES) - 2), 0
    y, rot, quality, rms = (v[active] for v in (y, rot, quality, rms))
    waves = []
    for delta in candidate_edits(y, rot, quality, rms, model, prior):
        recovered = iq.clone()
        recovered[active] += real_iq(delta * rot * rms)
        waves.append(recovered)
    return waves, count


@torch.no_grad()
def render(iq, _, baseline, tracker, models, *, threshold):
    model, prior = models
    reference, _ = baseline(iq)
    waves, count = readouts(iq, tracker, model, threshold, prior=prior)
    return torch.stack((iq, reference, *waves)), count


def bind_models(root, contract, device):
    model, threshold = parent_model(root, contract, device)
    return (model, load_moments(root, contract, device)), threshold


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, threshold = bind_models(root, binding[2], device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=models, require_clean_bypass=False,
          replay_fn=receive_only, finalize=freeze, reader_chunk=140)


def reserve(rt):
    def bind(root, contract, device, baseline, tracker):
        models, threshold = bind_models(root, contract, device)
        return lambda iq: render(iq, None, baseline, tracker, models, threshold=threshold)
    predict_clean_reserve(rt, modes=MODES, bind=bind)


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, cal, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('Smoke source and frozen recognizer contracts differ')
    model, threshold = parent_model(root, contract, device)
    prior = fit_moments(training, tracker)
    rng = torch.Generator(device=device).manual_seed(2026114933)
    contaminated = real_iq(colored_observation(complex_iq(cal[:4]), rng))
    active = corruption_score(coordinates(tracker, contaminated)[0]) > threshold
    tested = contaminated[active][:2]
    compatible, count = readouts(cal[:2], tracker, model, threshold, prior=prior)
    if not len(tested) or count or any(not torch.equal(x, cal[:2]) for x in compatible):
        raise ValueError('Real source smoke needs active corruption and exact clean bypass')
    waves, count = render(tested, None, baseline, tracker, (model, prior), threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (count != len(tested) or logits.shape != (len(tested) * len(MODES), 10)
            or not torch.isfinite(logits).all() or torch.equal(waves[-1], waves[-2])):
        raise ValueError('Real source joint-covariance smoke failed')
    delta = coordinates(tracker, tested)
    direct, _, _, independent, joint = candidate_edits(
        *delta, model, prior)
    torch.testing.assert_close(zero_carrier(joint), joint, atol=1e-6, rtol=1e-5)
    for output in (independent, joint):
        torch.testing.assert_close(
            torch.linalg.vector_norm(output * delta[-1], dim=(-2, -1)),
            torch.linalg.vector_norm(direct * delta[-1], dim=(-2, -1)), atol=1e-5, rtol=1e-5)
    write(root / 'smoke.json', dict(status='PASS', source_U=len(tested), clean_bypassed=len(compatible[0]),
                                  target_reads=0))
    print('LORA_JOINT_MODE_COVARIANCE_REAL_CHECKPOINT_PASS', count, flush=True)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    execute = partial(start, module=rt['runner'])
    try:
        for phase, rows in (('smoke', ('source',)), ('fit', ('source',)), ('timing', ('source',)),
                            ('evaluate', ROUTES), ('evaluate', ('reserve',))):
            schedule(rt, [(phase, row) for row in rows], execute)
        if execute(rt, 'score', 'source').wait():
            raise RuntimeError('Independent joint-mode covariance scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)))
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    tasks = cli.add_subparsers(dest='action', required=True)
    submission = tasks.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        submission.add_argument('--' + key, type=Path, required=True)
    for phase in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        t = tasks.add_parser(phase)
        t.add_argument('runtime', type=Path)
        if phase != 'dispatch':
            t.add_argument('row', choices=(*ROUTES, 'reserve') if phase == 'evaluate' else ('source',))
    args = vars(cli.parse_args())
    phase = args.pop('action')
    if phase == 'publish':
        return publish(**args, run=RUN, runner=__spec__.name)
    rt = read(args['runtime'])
    if phase == 'dispatch':
        return dispatch(rt)
    if phase == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if phase == 'smoke':
        return smoke(rt)
    if phase == 'fit':
        return fit(rt)
    if phase == 'timing':
        _, contract = setup(rt, 'cuda:0', smoke=True)
        prior = load_moments(Path(rt['run_root']), contract, 'cuda:0')
        return timing(rt, reader=partial(readouts, prior=prior),
                      timing_scope='frozen rank and both joint-mode posteriors; excludes DURA and F')
    if args['row'] == 'reserve':
        return reserve(rt)
    return evaluate(rt, args['row'])


if __name__ == '__main__':
    main()
