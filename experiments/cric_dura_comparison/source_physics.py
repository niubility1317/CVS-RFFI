"""Frozen source-L diagnostics, without fitted probes or target access."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .io import read, write


def interventions(x):
    z = torch.complex(x[:, 0], x[:, 1])
    n = torch.arange(z.shape[-1], device=x.device, dtype=x.dtype)
    values = [('identity', z)]
    for gain in (.5, 2.):
        values.append(('flat_gain_' + str(gain), z * gain))
    values.append(('flat_phase_pi2', z * 1j))
    for hz in (-100000., -25000., 25000., 100000.):
        values.append(('relative_CFO_' + str(int(hz)), z * torch.exp(2j * math.pi * hz / 25e6 * n)))
    delayed = torch.cat((torch.zeros_like(z[:, :3]), z[:, :-3]), 1)
    values.append(('FIR_delay3_echo02', (z + .2 * complex(.8, .6) * delayed) / math.sqrt(1.04)))
    power = z.abs().square() / z.abs().square().mean(1, keepdim=True).clamp_min(1e-8)
    for alpha in (.05, .1):
        value = z * (1 + alpha * power)
        value = value * (z.abs().square().mean(1, keepdim=True)
            / value.abs().square().mean(1, keepdim=True).clamp_min(1e-8)).sqrt()
        values.append(('synthetic_PA_alpha_' + str(alpha), value))
    return [(name, torch.stack((v.real, v.imag), 1)) for name, v in values]


def pair_geometry(z, y, rx, day):
    distance = 1 - F.normalize(z, dim=1) @ F.normalize(z, dim=1).T
    same_day = day[:, None] == day[None, :]
    positive = (y[:, None] == y[None, :]) & (rx[:, None] != rx[None, :]) & same_day
    negative = (y[:, None] != y[None, :]) & (rx[:, None] == rx[None, :]) & same_day
    def mean(mask):
        # Equal contribution per anchor, rather than weighting large groups twice.
        valid = mask.any(1)
        return float(((distance * mask).sum(1) / mask.sum(1).clamp_min(1))[valid].mean()) if valid.any() else None
    pos, neg = mean(positive), mean(negative)
    return dict(same_TX_cross_RX_distance=pos, different_TX_same_RX_distance=neg,
        separation_margin=None if pos is None or neg is None else neg-pos,
        positive_anchors=int(positive.any(1).sum()), negative_anchors=int(negative.any(1).sum()),
        pair_rule='same-day; anchor-balanced; no fitted RX/TX probe')


@torch.no_grad()
def run(reader, c, output, device):
    from .f_data import SourceDataset
    contract = read(c['source_contract'])
    data = SourceDataset(c['source_cache'], 'L_s', contract)
    m = data.metadata
    groups = sorted(set(zip(m['y'].tolist(), m['rx_i'].tolist(), m['day_i'].tolist())))
    index = np.asarray([i for y, r, d in groups
        for i in np.flatnonzero((m['y']==y)&(m['rx_i']==r)&(m['day_i']==d))[:8]], dtype=np.int64)
    x = data.x[index].to(device)
    def encode(value):
        return torch.cat([reader.features(part) for part in value.split(64)], 0)
    base = encode(x)
    pred = reader.classify_features(base).argmax(1)
    rows = []
    response = getattr(reader, 'response', None)
    base_response = response(x) if response is not None else None
    for name, value in interventions(x):
        z = encode(value)
        item = dict(intervention=name, count=len(index),
            feature_cosine_distance=float((1-F.cosine_similarity(base,z)).mean()),
            default_head_flip_fraction=float((reader.classify_features(z).argmax(1)!=pred).float().mean()))
        if response is not None:
            diff = response(value)-base_response
            item['reference_descriptor_relative_L2'] = float(diff.norm(dim=1).mean()/base_response.norm(dim=1).mean().clamp_min(1e-8))
        rows.append(item)
    to_tensor = lambda key: torch.tensor(m[key][index], device=device)
    write(output, dict(status='FROZEN_SOURCE_L_PHYSICS_DIAGNOSTICS', samples=len(index),
        groups=len(groups), samples_per_TX_RX_day_at_most=8,
        physical_ids=[str(v) for v in m['ids'][index]], source_IQ_binding=data.iq_binding,
        geometry=pair_geometry(base,to_tensor('y'),to_tensor('rx_i'),to_tensor('day_i')),
        interventions=rows, source_U_reads=0, target_reads=0, fitted_probe_updates=0,
        interpretation='Controlled sensitivity/inductive-bias evidence; PA changes are not label-preserving identity tests; no calibrated hardware recovery or full RX/channel invariance claim'))
