"""Source-only disturbance gate for half-dose residual practical DURA."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import FS_HZ, SOURCE
from .dura_repair import PacketRestorer, no_tone
from .full_packet_views import source_contract, source_role
from .residual_dura import COUNT, L_COUNT, ROLES, ROUTE


SCHEMA = 'lora_residual_disturbance_gate_v1'
SUPPORT_SEED = 2026112379
MARGIN = .5
QUALITY = .94
BANDS = ((62500, 72000), (72000, 85000), (85000, 102000), (102000, 123000))
FEATURES = 4 * (4 + len(BANDS))
SAMPLES = (('L_s', 0, 150, 200), ('U_s', L_COUNT, 650, 1000))
ISOLATION = dict(F_reads=0, TX_labels_read=0, target_reads=0)


class DisturbanceFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.tracker = PacketRestorer('consensus')
        freq = torch.fft.fftfreq(256, d=1 / FS_HZ).abs()
        self.register_buffer('band_masks', torch.stack([(freq >= lo) & (freq < hi) for lo, hi in BANDS]))
        self.register_buffer('window', torch.hann_window(256))

    @torch.no_grad()
    def forward(self, iq):
        wave, rotation, coherence, *_ = self.tracker.track(iq)
        z = torch.complex(iq[:, :, 0], iq[:, :, 1])
        spectrum = torch.fft.fft(z * self.window, norm='ortho').abs().square()
        total = spectrum.sum(-1).clamp_min(1e-8)
        power = z.abs().square().mean(-1).clamp_min(1e-8)
        phase = rotation / self.tracker.reference.chirp
        cfo = (phase[..., 1:] * phase[..., :-1].conj()).mean(-1).angle() * FS_HZ / (2 * np.pi)
        cfo_deviation = (cfo - cfo.median(1, keepdim=True).values).abs()
        residual = no_tone(wave) / power.sqrt()[..., None]
        centered = residual - residual.mean(1, keepdim=True)
        dispersion = centered.abs().square().mean(-1).clamp_min(1e-7).log()
        scalars = [(1 - coherence).clamp_min(1e-6).log(), power.log(),
                   cfo_deviation.clamp_min(1e-4).log(), dispersion]
        scalars.extend((spectrum[..., mask].sum(-1) / total).clamp_min(1e-10).log()
                       for mask in self.band_masks)
        features = torch.cat([value.sort(1).values[:, [0, 2, 4, 5]] for value in scalars], dim=1)
        return features, coherence.median(1).values


class DisturbanceGate(nn.Module):
    def __init__(self, record):
        super().__init__()
        self.features = DisturbanceFeatures()
        self.register_buffer('mean', torch.tensor(record['mean'], dtype=torch.float64))
        self.register_buffer('scale', torch.tensor(record['scale'], dtype=torch.float64))
        self.register_buffer('weight', torch.tensor(record['weight'], dtype=torch.float64))
        self.register_buffer('bias', torch.tensor(record['bias'], dtype=torch.float64))
        self.register_buffer('threshold', torch.tensor(record['threshold'], dtype=torch.float64))

    @torch.no_grad()
    def score(self, iq):
        features, _ = self.features(iq)
        normalized = ((features.double() - self.mean) / self.scale).clamp(-10, 10)
        return normalized @ self.weight + self.bias

    @torch.no_grad()
    def forward(self, iq):
        return self.score(iq) > self.threshold

    @torch.no_grad()
    def restore(self, iq, model, projector):
        """Apply half-dose DURA to active packets with exact bypass elsewhere."""
        active = self(iq)
        result = iq.clone()
        if active.any():
            selected = iq[active]
            flat = selected.flatten(0, 1)
            projected = projector.restore(flat, model(flat)).reshape_as(selected)
            result[active] = selected + .5 * (projected - selected)
        return result, active


def support_indices(source, contract):
    rng = np.random.default_rng(SUPPORT_SEED)
    fitting, calibration = [], []
    for role, offset, fit_count, cal_count in SAMPLES:
        _, receivers, _, _ = source_role(source, contract, role)
        for receiver in SOURCE:
            indices = rng.permutation(np.flatnonzero(receivers == receiver))
            if len(indices) < fit_count + cal_count:
                raise ValueError('Incomplete source support for the disturbance gate')
            fitting.extend(indices[:fit_count] + offset)
            calibration.extend(indices[fit_count:fit_count + cal_count] + offset)
    return np.asarray(fitting), np.asarray(calibration)


def collect(dataset, indices, features, device):
    values, qualities = [], []
    for group in np.array_split(indices, 1 + len(indices) // 64):
        vectors, quality = features(tensor(dataset[group], device))
        values.append(vectors.double().cpu())
        qualities.append(quality.cpu())
    return torch.cat(values), torch.cat(qualities)


def fit(source, pairs, output, device):
    """Fit only on source L/U IQ and store one immutable gate record."""
    source, pairs = Path(source), Path(pairs)
    contract = source_contract(source)
    pair_status = dict(status='SOURCE_READY', draws=2, roles=list(ROLES), n=COUNT,
                       F_reads=0, TX_reads=0, target_reads=0)
    if read(pairs / 'contract.json') != dict(contract, practical=ROUTE) or read(pairs / 'complete.json') != pair_status:
        raise ValueError('Completed, matched residual source pairs required')
    clean = np.load(pairs / 'clean.npy', mmap_mode='r', allow_pickle=False)
    received = np.load(pairs / 'received.npy', mmap_mode='r', allow_pickle=False)
    shape = (COUNT, 6, 2, 256)
    if clean.shape != shape or received.shape != (2, *shape) or any(a.dtype != np.float32 for a in (clean, received)):
        raise ValueError('Unexpected disturbance gate source IQ shape or dtype')

    fit_indices, cal_indices = support_indices(source, contract)
    features = DisturbanceFeatures().to(device).eval()
    original, _ = collect(clean, fit_indices, features, device)
    observed = [collect(draw, fit_indices, features, device) for draw in received]
    practical = torch.cat([item[0] for item in observed])
    hard = torch.cat([item[1] for item in observed]) >= QUALITY
    if not hard.any():
        raise ValueError('Missing high-coherence practical source observations')
    data = torch.cat((original, practical))
    mean, scale = data.mean(0), data.std(0).clamp_min(.025)
    normalized = ((data - mean) / scale).clamp(-10, 10)
    target = torch.cat((original.new_zeros(len(original)), original.new_ones(len(practical))))
    positive = torch.where(hard, 3., 1.).double()
    weights = torch.cat((original.new_ones(len(original)), positive * (len(original) / positive.sum())))
    layer = nn.Linear(FEATURES, 1, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    optimizer = torch.optim.LBFGS(layer.parameters(), lr=1., max_iter=120, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = layer(normalized)[:, 0]
        losses = binary_cross_entropy_with_logits(logits, target, reduction='none')
        loss = (losses * weights).sum() / weights.sum() + .008 * layer.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    calibrated, _ = collect(clean, cal_indices, features, device)
    with torch.no_grad():
        clean_scores = layer(((calibrated - mean) / scale).clamp(-10, 10))[:, 0]
        high_scores = layer(normalized[len(original):][hard])[:, 0]
        clean_max = float(clean_scores.max())
        practical_p02 = float(torch.quantile(high_scores, .02))
        if practical_p02 <= clean_max:
            raise ValueError('Source calibration leaves no clean to practical margin')
        decision = clean_max + MARGIN * (practical_p02 - clean_max)
        record = dict(schema=SCHEMA, protocol=contract['protocol'], practical=ROUTE,
                      source_receivers=list(SOURCE), role_ids={role: contract['role_ids'][role] for role in ROLES},
                      support_seed=SUPPORT_SEED, margin=MARGIN,
                      fit_packets=len(fit_indices), calibration_packets=len(cal_indices),
                      practical_realizations=len(practical), high_coherence_observations=int(hard.sum()),
                      clean_max=clean_max, practical_p02=practical_p02, threshold=decision,
                      mean=mean.tolist(), scale=scale.tolist(), weight=layer.weight[0].tolist(),
                      bias=float(layer.bias[0]), **ISOLATION)
    write(output, record)
    return record


def load(path, contract, device):
    record = read(path)
    if (record['schema'] != SCHEMA or record['protocol'] != contract['protocol'] or record['practical'] != ROUTE
            or record['source_receivers'] != list(SOURCE) or record['margin'] != MARGIN
            or record['role_ids'] != {role: contract['role_ids'][role] for role in ROLES}
            or any(record[key] != value for key, value in ISOLATION.items())):
        raise ValueError('Incompatible disturbance gate')
    return DisturbanceGate(record).to(device).eval().requires_grad_(False)
