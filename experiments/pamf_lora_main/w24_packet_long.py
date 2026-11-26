"""Matched long-budget waveform objectives for the scratch LoRa packet reconstructor."""

import argparse
from pathlib import Path, PurePosixPath
from time import monotonic

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.w24_edit_budget_audit import score
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_packet_reconstruction import (
    CENTER, FIT_BATCH, FIT_PACKETS, HOLD_PACKETS, PARENT, SEED, PacketReconstructor,
    predict_panel, risks, source_tools, training_pair,
)


RUN = '20261126-phase1-lora-w24-packet-long-objective-s2026112991-r01'
ARMS = ('original_loss', 'balanced_loss')
SNAPSHOTS = (1200, 6000, 12000)
BANDS = ((1, 8), (9, 32), (33, 64), (65, 192))
READOUTS = tuple(f'{arm}_{step}_{readout}' for step in SNAPSHOTS for arm in ARMS for readout in ('center','full'))
MODES = ('direct', *READOUTS, 'outer_half')
CONTRASTS = (
    *((f'balanced_loss_{step}_full', f'original_loss_{step}_full') for step in SNAPSHOTS),
    *((f'{arm}_{right}_full', f'{arm}_{left}_full')
      for arm in ARMS for left, right in zip(SNAPSHOTS[:-1], SNAPSHOTS[1:], strict=True)),
    ('balanced_loss_12000_full', 'balanced_loss_12000_center'),
    ('balanced_loss_12000_full', 'outer_half'), ('original_loss_12000_full', 'outer_half'),
)


def band_index(device):
    f = torch.arange(CENTER, device=device)
    return torch.minimum(f, CENTER-f)


def band_risk(estimated, target, distance):
    value = []
    for low, high in BANDS:
        band = (distance >= low) & (distance <= high)
        signal = target[..., band]
        error = (estimated[..., band]-signal).abs().square().mean((-1,-2))
        scale = signal.abs().square().mean((-1,-2))
        value.append(torch.log1p(error / (scale + .003)))
    return torch.stack(value, dim=-1).mean()


def enhanced_terms(output, target, guard, inputs, distance):
    center, _, _, edges = output
    estimated, clean = (torch.fft.fft(value, norm='ortho') for value in (center, target))
    frequency = band_risk(estimated, clean, distance)
    differential = band_risk(estimated[:,1:]-estimated[:,:-1], clean[:,1:]-clean[:,:-1], distance)
    received, measured_edges, _ = inputs
    identity = ((center[-1]-received[-1]).abs().square().mean()
               + (edges[-1]-measured_edges[-1]).abs().square().mean())
    edge_risk = (edges-guard).abs().square().mean()
    return torch.stack((frequency, differential, identity, edge_risk))


def losses(output, target, guard, inputs, distance):
    inherited = risks(output, target, guard)
    new = enhanced_terms(output, target, guard, inputs, distance)
    standard = inherited @ inherited.new_tensor((1., .65, .2, .2, .2, .15))
    balanced = (inherited @ inherited.new_tensor((1., .25, .2, .2, 0., 0.))
                + new @ new.new_tensor((.35, .22, .5, .2)))
    return (standard, balanced), torch.cat((inherited, new))


@torch.no_grad()
def heldout(models, tracker, generic, iq, selected, device, distance):
    heldout_rng = torch.Generator(device=device).manual_seed(SEED+1)
    collection = []
    for start in range(0, len(selected), FIT_BATCH):
        row = selected[start:start+FIT_BATCH]
        inputs, target, guard = training_pair(iq[row], tracker, generic, heldout_rng, device)
        measures = [losses(model(*inputs), target, guard, inputs, distance)[1] for model in models]
        collection.append(torch.stack(measures))
    return torch.stack(collection).mean(0).tolist()


def fit(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    tracker, generic = source_tools(rt, device)
    contract = source_contract(source)
    iq = native(source, 'U_s', contract['counts']['U_s'])
    sampling = np.random.default_rng(SEED)
    selected = sampling.choice(len(iq), FIT_PACKETS+HOLD_PACKETS, replace=False)
    fit_rows, hold_rows = selected[:FIT_PACKETS], selected[FIT_PACKETS:]
    torch.manual_seed(SEED)
    models = [PacketReconstructor(True).to(device) for _ in ARMS]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.0006) for model in models]
    rng = torch.Generator(device=device).manual_seed(SEED)
    distance = band_index(device)
    with torch.no_grad():
        inputs, target, guard = training_pair(iq[fit_rows[:2]], tracker, generic, rng, device)
        metrics = [losses(model(*inputs), target, guard, inputs, distance)[1] for model in models]
        if not torch.isfinite(torch.stack(metrics)).all():
            raise ValueError('Long objective source-only smoke failed')
    write(root/'fit-smoke.json', dict(status='PASS', F_reads=0, target_reads=0, source_u_packets=2))
    print('W24_PACKET_LONG_FIT_SMOKE PASS', flush=True)
    start = monotonic()
    observations, snapshots = [], {}
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, SNAPSHOTS[-1]+1):
        rows = sampling.choice(fit_rows, FIT_BATCH, replace=False)
        with torch.no_grad():
            inputs, target, guard = training_pair(iq[rows], tracker, generic, rng, device)
        values = []
        for idx, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
            objective, metrics = losses(model(*inputs), target, guard, inputs, distance)
            optimizer.zero_grad(set_to_none=True)
            objective[idx].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            values.append(metrics.detach())
        observations.append(torch.stack(values))
        if step % 200 == 0:
            mean = torch.stack(observations).mean(0)
            if not torch.isfinite(mean).all():
                raise ValueError('Nonfinite long-objective risk')
            print('W24_PACKET_LONG_STEP', step, mean.tolist(), flush=True)
            observations.clear()
        if step in SNAPSHOTS:
            hold = heldout(models, tracker, generic, iq, hold_rows, device, distance)
            if not np.isfinite(hold).all():
                raise ValueError('Nonfinite locked source holdout')
            states = [{k:v.detach().cpu() for k,v in model.state_dict().items()} for model in models]
            torch.save(states, root/f'packet-{step}.pt')
            snapshots[str(step)] = dict(heldout=hold, elapsed_s=monotonic()-start)
            write(root/f'freeze-{step}.json', dict(status='SOURCE_FROZEN', step=step, F_reads=0,
                  target_reads=0, source_truth_reads=0, source_V_reads=0, **snapshots[str(step)]))
            print('W24_PACKET_LONG_FROZEN', step, flush=True)
    write(root/'fit-complete.json', dict(status='SOURCE_FROZEN', initialization='scratch', seed=SEED,
          arms=list(ARMS), snapshots=snapshots, parameters=sum(p.numel() for p in models[0].parameters()),
          peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
          source_fit_packets=FIT_PACKETS, source_holdout_packets=HOLD_PACKETS,
          F_reads=0, target_reads=0, source_truth_reads=0, source_V_reads=0))


def predict(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    fit_record = read(root/'fit-complete.json')
    if fit_record['status'] != 'SOURCE_FROZEN' or fit_record['initialization'] != 'scratch' or any(
            fit_record[k] for k in ('F_reads','target_reads','source_truth_reads','source_V_reads')):
        raise ValueError('All locked source fits must be complete')
    models = []
    for step in SNAPSHOTS:
        proof = read(root/f'freeze-{step}.json')
        if proof['step'] != step or proof['status'] != 'SOURCE_FROZEN' or any(
                proof[k] for k in ('F_reads','target_reads','source_truth_reads','source_V_reads')):
            raise ValueError('Missing locked source checkpoint')
        states = torch.load(root/f'packet-{step}.pt', map_location=device, weights_only=True)
        for state, _ in zip(states, ARMS, strict=True):
            model = PacketReconstructor(True).to(device).eval()
            model.load_state_dict(state)
            models.append(model)
    predict_panel(rt, models, MODES[:-1])


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    phases = ('fit-source','predict-source','score-source')
    for phase in ('dispatch', *phases):
        subs.add_parser(phase).add_argument('runtime', type=Path)
    pub = subs.add_parser('publish')
    for field in ('repo','context','ssh-config','output'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__, phases=phases, gpu_phases=phases[:2])
    elif args.action == 'fit-source':
        fit(read(args.runtime))
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        score(args.runtime, modes=MODES, contrasts=CONTRASTS)
    else:
        ctx = read(args.context)
        rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs', project/'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3, 6, 0], cpu_workers=8)
        rt['external_releases'] = [rt[field] for field in ('source_root','classifier_root','w24_release_root',
                                                           'training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
