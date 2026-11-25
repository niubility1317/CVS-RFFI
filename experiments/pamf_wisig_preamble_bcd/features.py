"""Single-packet WiFi short-training structure and source-only nuisance fitting."""

from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 12
RANK = 2
DOSE = .5
MAX_NOISE_NORM = .20


def cycle_descriptor(x):
    if x.ndim != 3 or x.shape[1:] != (2, 256):
        raise ValueError('Expected 25 MS/s IQ[B,2,256]')
    iq = torch.complex(x[:, 0, 20:180], x[:, 1, 20:180])
    anchor = iq[:, 60:140]  # samples 80:160 in the full packet
    a, b = anchor[:, :-20], anchor[:, 20:]
    cross = (b * a.conj()).sum(1)
    angle = torch.atan2(cross.imag, cross.real + 1e-12) / 20
    phase = -angle[:, None] * torch.arange(160, device=x.device, dtype=x.dtype)
    corrected = iq * torch.complex(phase.cos(), phase.sin())
    periods = corrected.reshape(-1, 8, 20)
    template = periods[:, 3:7].mean(1)
    power = template.abs().square().mean(1)
    denominator = power + 1e-6
    relative = (periods * template[:, None].conj()).mean(2) / denominator[:, None]
    difference = periods - relative[:, :, None] * template[:, None]
    residual_energy = difference.abs().square().mean(2)
    energy = periods.abs().square().mean(2)
    reliability = (cross.abs() / (a.abs().square().sum(1) * b.abs().square().sum(1) + 1e-12).sqrt())
    reliability = reliability.clamp(max=1.) * power / denominator

    def summarize(start, end):
        gain = relative[:, start:end].mean(1)
        magnitude = energy[:, start:end].mean(1)
        error = residual_energy[:, start:end].mean(1)
        return torch.stack(((gain.real - 1).tanh(), gain.imag.tanh(),
                            ((magnitude + 1e-6) / denominator).log().tanh(),
                            error / (denominator + error)), dim=1)

    values = torch.cat([summarize(lo, hi) for lo, hi in ((0, 2), (2, 6), (6, 8))], dim=1)
    return values * reliability[:, None]


def fit_drift(values, tx, rx, day, *, expected):
    """Fit one shared perturbation from same-TX, same-day receiver mean contrasts."""
    if values.ndim != 2 or values.shape[1] != WIDTH or not values.isfinite().all():
        raise ValueError('Finite source descriptor matrix required')
    groups = [torch.as_tensor(col, device=values.device) for col in (tx, rx, day)]
    if any(col.ndim != 1 or len(col) != len(values) or (col < 0).any() for col in groups):
        raise ValueError('Valid labeled source groups required')
    tx, rx, day = (col.unique(return_inverse=True)[1] for col in groups)
    n_tx, n_rx, n_day = expected
    if tuple(int(col.max()) + 1 for col in (tx, rx, day)) != expected or n_rx < 2:
        raise ValueError('Expected all registered TX, RX and day groups')
    index = (tx * n_day + day) * n_rx + rx
    n_group = n_tx * n_day * n_rx
    count = index.bincount(minlength=n_group)
    if (count < 2).any():
        raise ValueError('Expected at least two labeled source packets in every registered group')
    values = values.to(torch.float64)
    centers = values.new_zeros(n_group, WIDTH).index_add_(0, index, values) / count[:, None]
    packet_variance = (values - centers[index]).square().mean(0)
    means = centers.reshape(n_tx, n_day, n_rx, WIDTH)
    contrasts = torch.cat([(means[:, :, i] - means[:, :, j]).reshape(-1, WIDTH)
                           for i, j in combinations(range(n_rx), 2)])
    scatter = contrasts.T @ contrasts / len(contrasts)
    variances = scatter.diagonal()
    trace = variances.sum()
    if not torch.isfinite(trace) or trace <= 0:
        raise ValueError('Source receiver contrast has no usable variance')
    shrinkage = .2
    shrunk = (1 - shrinkage) * scatter + shrinkage * torch.diag(variances)
    eigenvalues, eigenvectors = torch.linalg.eigh(shrunk)
    top = eigenvalues[-RANK:].clamp_min(0)
    directions = eigenvectors[:, -RANK:].T * (top * (trace / top.sum())).sqrt()[:, None]
    scale = min(DOSE, MAX_NOISE_NORM / float(trace.sqrt()))
    return dict(diagonal=(torch.diag(variances.sqrt()) * scale).float(),
                low_rank=(directions * scale).float(),
                variance=variances.tolist(), packet_variance=packet_variance.tolist(),
                rx_pairs=len(contrasts), group_sizes=count.tolist(), shrinkage=shrinkage,
                trace=float(trace), scale=scale, top_two_fraction=float(top.sum() / trace))


class CycleResponse(nn.Module):
    def __init__(self, reference, noise_factor, seed):
        super().__init__()
        self.reference = reference
        self.register_buffer('noise_factor', noise_factor, persistent=False)
        self.seed = seed
        self.generator = None
        self.perturbed_packets = 0

    def forward(self, x):
        original = self.reference(x)
        descriptor = cycle_descriptor(x)
        if self.training and self.noise_factor is not None:
            if self.generator is None:
                self.generator = torch.Generator(device=x.device).manual_seed(self.seed)
            shape = (len(x), len(self.noise_factor))
            signs = torch.randint(0, 2, shape, generator=self.generator, device=x.device)
            noise = (signs.to(descriptor.dtype) * 2 - 1) @ self.noise_factor.to(descriptor.dtype)
            descriptor = descriptor + noise
            self.perturbed_packets += len(x)
        return torch.cat((original, descriptor), dim=1)

    def diagnostics(self, x):
        return self.reference.diagnostics(x)

    def contract(self):
        return dict(self.reference.contract(), preamble_cycle_features=WIDTH)


class CycleProjection(nn.Linear):
    def __init__(self, original, allow_prefix):
        # Keep the original parameter objects and their initialization/RNG.
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.cycle_weight = nn.Parameter(self.weight.new_zeros(self.out_features, WIDTH))
        self.allow_prefix = allow_prefix

    def _load_from_state_dict(self, state_dict, prefix, *args):
        key = prefix + 'cycle_weight'
        if self.allow_prefix and key not in state_dict:
            state_dict[key] = torch.zeros_like(self.cycle_weight)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x):
        original, descriptor = x.split((self.in_features, WIDTH), dim=1)
        return super().forward(original) + F.linear(descriptor, self.cycle_weight)
