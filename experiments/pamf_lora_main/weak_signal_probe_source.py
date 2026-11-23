"""Fit and score LoRa denoisers with a source-only weak-signal response test."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .band_phase_source import panel as previous_panel
from .complex_vector_source import dispatch, publish, record, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX, normalized
from .dura_repair import PacketRestorer, real_iq
from .inverse_noise_dose_source import panel as inverse_panel
from .joint_unroll_source import coordinates
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .signal_retention_source import build_arrays
from .spectral_unroll import ROUTES
from .weak_signal_probe import ProjectedDenoise, held_readouts, probe_response, reconstruction, source_probe


RUN = '20261123-phase1-lora-dura-weak-signal-probe-source-m2-r01'
RECIPE = SimpleNamespace(seed=2026112851, steps=2000, train_per_rx=900, held_per_rx=80,
                         arms=('ordinary', 'probe'), model=lambda _: ProjectedDenoise())
BATCH = 10
PROBE_WEIGHT = .12
MODES = ('direct', 'reference', 'ordinary', 'ordinary_half', 'probe', 'probe_half', 'known_noise')


def panel(rx):
    used = np.union1d(previous_panel(rx), inverse_panel(rx, 60))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), used),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


@torch.no_grad()
def held_metrics(model, arrays):
    measured, quality, exact, common, _, _ = arrays
    summary, totals = [], []
    model.eval()
    for take in torch.arange(len(measured), device=measured.device).split(BATCH):
        y, q, signal = (a[take] for a in (measured, quality, exact))
        value, restored = reconstruction(model, y, q, signal)
        residue, gain = probe_response(model, y, q, source_probe(signal, common[take], take))
        totals.append(held_readouts(restored, y, signal))
        summary.append(len(take)*torch.cat((value[None], residue, gain)))
    averages = torch.stack(summary).sum(0)/len(measured)
    physical = torch.stack(totals).sum(0)
    return dict(reconstruction=float(averages[0]), probe_loss=averages[1:3].tolist(),
                probe_gain=averages[3:].tolist(),
                error_fraction=(physical[:, 0]/physical[:, 1].clamp_min(1e-8)).tolist(),
                true_signal_gain=(physical[:, 2]/physical[:, 3].clamp_min(1e-8)).tolist())


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(RECIPE.seed)
    tracker = PacketRestorer('consensus').to(device).eval()
    count, hold = (6, 4) if smoke else (RECIPE.train_per_rx, RECIPE.held_per_rx)
    fitting, held, contract = build_arrays(rt, route, tracker, count, hold)
    measured, quality, exact, common, clean, clean_quality = fitting
    models = {arm: RECIPE.model(arm).to(device) for arm in RECIPE.arms}
    models['probe'].load_state_dict(models['ordinary'].state_dict())
    optimizer = torch.optim.AdamW([p for m in models.values() for p in m.parameters()],
                                  lr=2e-4, weight_decay=1e-4)
    steps = 1 if smoke else RECIPE.steps
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-5)
    rng = torch.Generator(device=device).manual_seed(RECIPE.seed)
    for step in range(1, steps+1):
        ix = torch.randint(len(measured), (BATCH,), device=device, generator=rng)
        y, q, source = (a[ix] for a in (measured, quality, exact))
        y[-2:] = source[-2:] = clean[ix[-2:], None]
        q[-2:] = clean_quality[ix[-2:], None]
        values = torch.stack([reconstruction(m, y, q, source)[0] for m in models.values()])
        loss = values.sum()
        if step == 1 or step % 2 == 0:
            used = ix[:4]
            delta = source_probe(source[:4], common[used], used)
            probe, gain = probe_response(models['probe'], y[:4], q[:4], delta)
            loss = loss+PROBE_WEIGHT*probe.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite weak-signal denoising loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for m in models.values():
            gradient = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and (gradient <= 0 or any(h.out.weight.grad.norm() <= 0
                                                 for h in (m.spectral, m.temporal))):
                raise ValueError('Expected a finite nonzero source gradient in both noise paths')
        optimizer.step()
        lr.step()
        if step == 1 or step % 50 == 0:
            print('WEAK_SIGNAL_FIT', route, step, *values.detach().tolist(), *probe.detach().tolist(),
                  *gain.detach().tolist(), flush=True)
    proof = record(contract, route, RECIPE)
    report = {k: v for k, v in proof.items() if k != 'contract'}
    report.update(probe_weight=PROBE_WEIGHT, fit_packets=len(measured), held_packets=len(held[0]),
                  held={arm: held_metrics(m, held) for arm, m in models.items()})
    root = Path(rt['run_root'])
    if smoke:
        write(root/f'smoke-{route}.json', dict(report, status='PASS', steps=1,
                                               train_per_rx=count, held_per_rx=hold))
    else:
        torch.save(dict(proof=proof, models={k:m.state_dict() for k,m in models.items()}),
                   root/f'{route}-model.pt')
        write(root/f'{route}-complete.json', report)


@torch.no_grad()
def render(iq, noise, baseline, tracker, models):
    reference, active = baseline(iq)
    outputs = [iq.clone() for _ in MODES[2:]]
    if active.any():
        observed = iq[active]
        wave, rot, quality, rms = coordinates(tracker, observed)
        ordinary, probe = (real_iq(m(wave, quality)*rot*rms) for m in models)
        estimates = (ordinary, .5*ordinary, probe, .5*probe, noise[active])
        for result, estimate in zip(outputs, estimates, strict=True):
            result[active] = normalized(observed-estimate, observed)
    return torch.stack((iq, reference, *outputs)), int(active.sum())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
    else:
        rt = read(args['runtime'])
        if action == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            route = args['route']
            if action == 'score':
                score(rt, route, recipe=RECIPE, modes=MODES, panel=panel, render=render)
            else:
                source_trace_implementation()
                fit(rt, route, smoke=action=='smoke')
