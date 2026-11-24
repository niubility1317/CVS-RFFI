"""Screen a clean source-IQ likelihood prior on LoRa full and residual practical."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .empirical_packet_prior import bank_summary, candidate_edits, detail_bins
from .joint_mode_covariance_source import dispatch
from .joint_unroll_source import coordinates, source_control
from .private_transport_source import timing
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-empirical-iq-prior-source-m2-r01'
MODES = ('direct', 'reference', 'rank_half', 'split_half', 'joint_half',
         'rank_split_matched', 'rank_joint_matched')
PAIRS = (('joint_half', 'split_half'), ('rank_joint_matched', 'rank_split_matched'),
         ('rank_joint_matched', 'rank_half'), ('rank_split_matched', 'rank_half'))


@torch.no_grad()
def build_bank(packets, tracker):
    bins = detail_bins(packets.device)
    inputs = [torch.fft.fft(coordinates(tracker, x)[0], norm='ortho')[..., bins]
              for x in packets.split(80)]
    return torch.cat(inputs)


@torch.no_grad()
def fit(rt):
    (training, _, _), contract = setup(rt, 'cuda:0')
    tracker = PacketRestorer('consensus').cuda().eval()
    root = Path(rt['run_root'])
    start = perf_counter()
    bank = build_bank(training, tracker).cpu()
    if not torch.isfinite(bank).all():
        raise ValueError('Source empirical IQ bank is nonfinite')
    with (root / 'source-bank.pt').open('xb') as stream:
        torch.save(dict(contract=contract, bank=bank), stream)
    write(root / 'fit.json', dict(status='SOURCE_FROZEN', source_U=len(training),
          scalar_statistics=2 * bank.numel(), trainable_parameters=0, seconds=perf_counter()-start,
          target_reads=0, F_reads=0, TX_reads=0))


def bind_models(root, contract, device):
    model, threshold = parent_model(root, contract, device)
    if read(root / 'fit.json')['status'] != 'SOURCE_FROZEN':
        raise ValueError('Expected frozen empirical IQ bank')
    stored = torch.load(root/'source-bank.pt', map_location=device, weights_only=True)
    if stored['contract'] != contract:
        raise ValueError('Sources of IQ bank and evaluation are inconsistent')
    bank = stored['bank']
    return (model, (bank, *bank_summary(bank))), threshold


@torch.no_grad()
def readouts(iq, tracker, model, threshold, *, prior):
    y, rot, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    if not count:
        return [iq] * (len(MODES) - 2), 0
    y, rot, quality, rms = (v[active] for v in (y, rot, quality, rms))
    edits, _ = candidate_edits(y, rot, quality, rms, model, prior)
    waves = []
    for delta in edits:
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


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, threshold = bind_models(root, binding[2], device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze, reader_chunk=140)


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
        raise ValueError('Smoke bank and frozen recognizer contracts differ')
    model, threshold = parent_model(root, contract, device)
    bank = build_bank(training, tracker)
    prior = (bank, *bank_summary(bank))
    rng = torch.Generator(device=device).manual_seed(2026114933)
    contaminated = real_iq(colored_observation(complex_iq(cal[:4]), rng))
    active = corruption_score(coordinates(tracker, contaminated)[0]) > threshold
    tested = contaminated[active][:2]
    bypass, n = readouts(cal[:2], tracker, model, threshold, prior=prior)
    if not len(tested) or n or any(not torch.equal(value, cal[:2]) for value in bypass):
        raise ValueError('Source empirical smoke needs clean bypass and active perturbation')
    output, active = render(tested, None, baseline, tracker, (model, prior), threshold=threshold)
    logits = load_reader(root, contract, device)(output.flatten(0, 1))[0]
    if (active != len(tested) or logits.shape != (len(tested) * len(MODES), 10)
            or not logits.isfinite().all() or torch.equal(output[-1], output[-2])):
        raise ValueError('Real source IQ prior smoke failed')
    print('LORA_EMPIRICAL_IQ_PRIOR_REAL_CHECKPOINT_PASS', active, flush=True)
    write(root/'smoke.json', dict(status='PASS', source_U=active, clean_bypassed=2, target_reads=0))


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
        (_, prior), _ = bind_models(Path(rt['run_root']), contract, 'cuda:0')
        return timing(rt, reader=partial(readouts, prior=prior),
                      timing_scope='frozen rank and both empirical posteriors; excludes DURA and F')
    return reserve(rt) if args['row'] == 'reserve' else evaluate(rt, args['row'])


if __name__ == '__main__':
    main()
