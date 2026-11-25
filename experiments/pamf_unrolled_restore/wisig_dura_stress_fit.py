"""Train the expanded two-stage backbone from blind WiSig source pairs."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.channel_blind_source import waveform_risk
from experiments.pamf_lora_main.dura_capacity import DuraCapacity
from .wisig_dura_stress_data import EPOCHS, PROFILES, read_plan, use_route
from .wisig_lora_adaptation import as_iq
from .wisig_lora_source_fit import source_iq
from .wisig_receiver_estimates import receive_coordinates


SEED = 2026092701
STEPS = 40000
BATCH = 16


def stages(model, iq):
    y, level, quality, receiver_variance = receive_coordinates(iq)
    result, action, variance, _, _ = model.stages(
        y[:, None], quality[:, None], level[:, None], receiver_variance=receiver_variance[:, None])
    return y[:, None], level[:, None], result, action, variance


def fit_loss(model, observed, clean, identity):
    y, level, (coarse, first, final), action, variance = stages(model, observed)
    target = torch.complex(clean[:, :1], clean[:, 1:]) / level
    # Keep the receiver's packet-wide complex gain when pairing waveform targets.
    gain = (target.conj() * y).sum(-1, keepdim=True) / target.abs().square().sum(-1, keepdim=True).clamp_min(1e-8)
    target = target * gain
    half = y + .5 * (final - y)
    error = (y - action(target)).abs().square()
    penalty = (error / (.01 + variance.detach())).log1p().mean()
    likelihood = (error.detach() / variance + variance.log()).mean()
    identity_loss = (half - y).abs().square().mean((1, 2)).mul(identity).mean()
    signal = (.6 * waveform_risk(coarse, target) + .4 * waveform_risk(first, target)
              + .35 * waveform_risk(final, target) + .65 * waveform_risk(half, target))
    return signal + .07 * penalty + .015 * likelihood + .35 * identity_loss


def source_views(cache, output, profile, *, smoke=False):
    if profile not in PROFILES:
        raise ValueError('Unknown practical training profile')
    result = []
    for epoch in EPOCHS[:1] if smoke else EPOCHS:
        folder, quantile = read_plan(cache, epoch)
        full = np.load(folder / 'full.npy', mmap_mode='r', allow_pickle=False)
        residual = np.load(folder / 'residual.npy', mmap_mode='r', allow_pickle=False)
        if profile == 'strong':
            strong_root = output / 'strong_source_iq'
            if read(strong_root / 'complete.json')['epochs'] != list(EPOCHS):
                raise ValueError('Harder source IQ cache is incomplete')
            full = residual = np.load(strong_root / f'{epoch:02}.npy', mmap_mode='r', allow_pickle=False)
        result.append((quantile, full, residual))
    return result


def examples(clean, view, selected):
    quantile, full, residual = view
    target = clean[selected]
    q = quantile[selected]
    observed = target.copy()
    for route, array in (('full', full), ('residual', residual)):
        selected_route = use_route(q, route)
        observed[selected_route] = array[selected[selected_route]]
    return observed, target, q < .15


@torch.no_grad()
def threshold(clean, ids, device):
    estimates = [receive_coordinates(tensor(clean[ids[start:start + 128]], device))[-1]
                 for start in range(0, len(ids), 128)]
    return float(torch.quantile(torch.cat(estimates).flatten(), .999) * 1.05 + 1e-6)


def fit(rt, profile, *, smoke=False):
    root = Path(rt['run_root'])
    cache = Path(rt['source_root'])
    clean, fit_ids, calibration = source_iq(cache)
    views = source_views(cache, root, profile, smoke=smoke)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    model = DuraCapacity('expanded').to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    sampler = np.random.default_rng(SEED + 2)
    start = perf_counter()
    for step in range(1, 4 if smoke else STEPS + 1):
        selection = sampler.choice(fit_ids, BATCH, replace=False)
        rows = examples(clean, views[(step - 1) % len(views)], selection)
        loss = fit_loss(model, *(tensor(item, device) for item in rows))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step == 1 or step % 100 == 0:
            print('WISIG_STRESS_FIT', profile, step, float(loss.detach()), perf_counter()-start, flush=True)
    if smoke:
        print('SOURCE_IQ_SMOKE_PASS', profile, 'target_reads=0', flush=True)
        return
    record = dict(status='SOURCE_FROZEN', profile=profile, steps=STEPS, seed=SEED, batch=BATCH,
                  threshold=threshold(clean, calibration, device), fit_count=len(fit_ids),
                  params=sum(p.numel() for p in model.parameters()),
                  seconds=perf_counter()-start, peak_mib=torch.cuda.max_memory_allocated()/2**20,
                  calibration_count=len(calibration), source_epochs=list(EPOCHS), target_reads=0,
                  TX_reads=0, F_reads=0, simulator_parameter_reads=0)
    with (root / f'{profile}.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), metadata=record), stream)
    write(root / f'{profile}-complete.json', record)


def frozen_fit(root, profile):
    record = read(root / f'{profile}-complete.json')
    cp = torch.load(root / f'{profile}.pt', map_location='cuda:0', weights_only=True)
    if (record, record['status'], record['profile'], record['steps']) != (
            cp['metadata'], 'SOURCE_FROZEN', profile, STEPS):
        raise ValueError('Expected the terminal WiSig restorer')
    if any(record[key] for key in ('target_reads', 'TX_reads', 'F_reads', 'simulator_parameter_reads')):
        raise ValueError('Unexpected restorer training input')
    model = DuraCapacity('expanded').cuda().eval().requires_grad_(False)
    model.load_state_dict(cp['weights'], strict=True)
    return model, record['threshold']


def restore(model, threshold, iq, variance):
    active = variance.flatten() > threshold
    if not active.any():
        return iq, active
    y, level, result, _, _ = stages(model, iq[active])
    out = iq.clone()
    out[active] += as_iq(.5 * (result[-1] - y).squeeze(1) * level.squeeze(1))
    return out, active
