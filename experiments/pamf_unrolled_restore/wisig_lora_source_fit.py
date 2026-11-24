"""Fit WiSig adaptations using only the fixed TX-hidden source IQ cache.

Each arm is run in its own process with its original LoRa implementation on
the Python path. No channel trace or simulator-provided noise is an input.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .wisig_lora_adaptation import expanded_readout, rank_readout
from .wisig_receiver_estimates import guard_features, guard_score, receive_coordinates


SEED = 2026092701
STEPS = 40000
BATCH = 16
CALIBRATION = 4096


def source_iq(root):
    root = Path(root)
    if root.parent.name != '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02':
        raise ValueError('Expected the fixed WiSig source cache')
    contract = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
    if (contract['train_count'], contract['U_count'], contract['TX_labels_read'], contract['target_reads']) != (
            63000, 56700, False, False):
        raise ValueError('Expected the TX-hidden source cache contract')
    iq = np.load(root / 'train' / 'clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(root / 'train' / 'roles.npy', allow_pickle=False)
    if iq.shape != (63000, 2, 256) or iq.dtype != np.float32 or roles.shape != (63000,):
        raise ValueError('Unexpected source IQ or role layout')
    ids = np.random.default_rng(SEED).permutation(np.flatnonzero(roles == 'U_s'))
    if len(ids) != 56700:
        raise ValueError('Expected 56,700 U_s records')
    return iq, ids[CALIBRATION:], ids[:CALIBRATION]


def as_tensor(rows, device):
    return torch.as_tensor(np.array(rows, copy=True), device=device)


def perturb(clean, generator):
    """Return only augmented IQ; perturbation settings are not model inputs."""
    x = torch.complex(clean[:, 0], clean[:, 1])
    random = torch.rand((len(x), 4), generator=generator, device=x.device)
    theta = (random[:, :1] - .5) * .8
    echo = torch.complex((random[:, 1:2] - .5) * .4, (random[:, 2:3] - .5) * .4)
    signal = (x + echo * x.roll(2, -1)) * torch.polar(torch.ones_like(theta), theta)
    noise = torch.complex(torch.randn(x.shape, generator=generator, device=x.device),
                          torch.randn(x.shape, generator=generator, device=x.device))
    level = x.abs().square().mean(-1, keepdim=True).sqrt()
    y = signal + noise * level * random[:, 3:4].square() * .6
    return torch.stack((y.real, y.imag), 1)


@torch.no_grad()
def source_rank_statistics(iq, ids, device):
    total = torch.zeros(512, device=device, dtype=torch.float64)
    cross = torch.zeros((512, 512), device=device, dtype=torch.float64)
    for start in range(0, len(ids), 128):
        x = as_tensor(iq[ids[start:start + 128]], device)
        y, _, _, _ = receive_coordinates(x)
        f = torch.fft.fft(y, norm='ortho')
        features = torch.cat((f.real, f.imag), -1).double()
        total += features.sum(0)
        cross += features.T @ features
    mean = total / len(ids)
    covariance = cross / len(ids) - mean[:, None] * mean[None, :]
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    factors = vectors[:, -4:] * eigenvalues[-4:].clamp_min(0).sqrt()
    diagonal = (covariance.diagonal() - factors.square().sum(-1)).clamp_min(1e-5)
    return diagonal.float(), factors.float()


def make_model(arm):
    if arm == 'expanded':
        # The frozen transfer arm imports the original release. This arm uses
        # the current committed WiSig variance-input extension explicitly.
        import importlib.util
        import sys
        path = Path(__file__).resolve().parents[1] / 'pamf_lora_main' / 'dura_capacity.py'
        name = 'experiments.pamf_lora_main.wisig_dura_capacity_impl'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.DuraCapacity('expanded')
    if arm == 'guard':
        from experiments.pamf_lora_main.spectral_covariance_source import RECIPES
        return RECIPES['rank'].model()
    raise ValueError('Expected expanded or guard')


@torch.no_grad()
def fit_guard(source, fit_ids, calibration, device):
    from experiments.pamf_lora_main.weak_guard import monotone_scores
    generator = torch.Generator(device=device).manual_seed(SEED + 7)
    negative, positive = [], []
    for start in range(0, min(len(fit_ids), 20000), 128):
        clean = as_tensor(source[fit_ids[start:min(start + 128, 20000)]], device)
        negative.append(guard_features(clean).cpu())
        positive.append(guard_features(perturb(clean, generator)).cpu())
    midpoint, dispersion, weight, offset = monotone_scores(torch.cat(positive), torch.cat(negative))
    record = dict(midpoint=midpoint.tolist(), dispersion=dispersion.tolist(), weight=weight.tolist(), offset=float(offset))
    scores = []
    for start in range(0, len(calibration), 128):
        clean = as_tensor(source[calibration[start:start + 128]], device)
        scores.append(guard_score(clean, record))
    record['threshold'] = float(torch.cat(scores).max() + .01)
    return record


def fit(root, source_root, arm, *, smoke=False):
    root = Path(root)
    source, fit_ids, calibration = source_iq(source_root)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    device = 'cuda:0'
    model = make_model(arm).to(device).train()
    diagonal, factors = (source_rank_statistics(source, fit_ids[:256] if smoke else fit_ids, device)
                         if arm == 'guard' else (None, None))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    sampler = np.random.default_rng(SEED + 2)
    for step in range(1, 2 if smoke else STEPS + 1):
        clean = as_tensor(source[sampler.choice(fit_ids, BATCH, replace=False)], device)
        observed = perturb(clean, generator)
        observed[-2:] = clean[-2:]
        output = (expanded_readout(model, observed) if arm == 'expanded'
                  else rank_readout(model, observed, diagonal, factors))
        loss = ((output - clean).square().mean((1, 2)) / (.08 + clean.square().mean((1, 2)))).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step == 1 or step % 100 == 0:
            print('WISIG_LORA_SOURCE_FIT', arm, step, float(loss.detach()), flush=True)
    if smoke:
        return
    model.eval()
    with torch.no_grad():
        estimates = []
        for start in range(0, len(calibration), 128):
            estimates.append(receive_coordinates(as_tensor(source[calibration[start:start + 128]], device))[-1].flatten())
        threshold = float(torch.quantile(torch.cat(estimates), .999) * 1.05 + 1e-6)
    guard = fit_guard(source, fit_ids, calibration, device) if arm == 'guard' else None
    record = dict(arm=arm, steps=STEPS, seed=SEED, batch=BATCH, threshold=threshold, weak_guard=guard,
                  fit_count=len(fit_ids), calibration_count=len(calibration), target_reads=0,
                  TX_reads=0, F_reads=0, simulator_parameter_reads=0)
    with (root / f'{arm}.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), diagonal=diagonal, factors=factors, metadata=record), stream)
    (root / f'{arm}-complete.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
