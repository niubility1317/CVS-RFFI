"""Receiver-observed WiSig timing with a shared preamble channel estimate."""

import torch

from .receiver_pilot import BEGIN, END, PERIOD, SIGNS, TONES, estimate, phase, power


NOMINAL = 10 * PERIOD
HALO = 2 * PERIOD
ANCHOR = PERIOD // 4
LONG_MATCH = 16
QUANTILE = .999
READOUTS = ('boundary_protocol', 'joint_mirror', 'joint_protocol')
LONG = (
    0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
    1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
    0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
)


def preamble_template(time):
    """Public complex reference at 25 Msps with t=0 at the L-STF/L-LTF boundary."""
    short_bins = time.new_tensor(TONES) / PERIOD
    long_bins = torch.arange(-32, 32, device=time.device, dtype=time.dtype) / (4 * PERIOD)
    short = (phase(2 * torch.pi * time[..., None] * short_bins)
             @ time.new_tensor(SIGNS).to(torch.complex64)) * ((13 / 6) ** .5 * (1 + 1j))
    long = phase(2 * torch.pi * (time[..., None] - 2 * PERIOD) * long_bins) @ time.new_tensor(
        LONG).to(torch.complex64)
    return torch.where(time < 0, short, long) / 52 ** .5


def ridge_inverse(design, penalty):
    gram = design.H @ design
    return torch.linalg.solve(gram + penalty * len(design), design.H)


class Timing:
    def __init__(self, device):
        self.time = torch.arange(256, device=device)
        self.starts = torch.arange(NOMINAL - HALO, NOMINAL + HALO + 1, device=device)
        offset = torch.arange(-2 * PERIOD, LONG_MATCH, device=device)
        self.observed_indices = self.starts[:, None] + offset
        self.long_indices = self.observed_indices[:, -LONG_MATCH:]
        self.near = (self.starts - NOMINAL).abs() <= ANCHOR
        taps = torch.arange(-2, 3, device=device)
        design = preamble_template((offset[:, None] - taps).float())
        penalty = torch.diag(design.new_tensor([.03, .03, .001, .03, .03]))
        self.inverse = ridge_inverse(design, penalty)
        self.folds = []
        even, odd = slice(None, None, 2), slice(1, None, 2)
        for fit, held in ((even, odd), (odd, even)):
            self.folds.append((fit, held, design[held] @ ridge_inverse(design[fit], penalty)))
        self.edge_basis = preamble_template((self.time[:, None] - NOMINAL - taps).float())
        self.edge_single = preamble_template((self.time - NOMINAL).float())
        self.long_ref = self.edge_single[NOMINAL:NOMINAL + LONG_MATCH]

    @staticmethod
    def fractional_error(error, value):
        return power(error).sum(dim=-1) / power(value).sum(dim=-1).clamp_min(1e-8)

    def joint_score(self, observations):
        split = PERIOD
        total = 0
        for fit, held, projection in self.folds:
            value = observations[..., held]
            error = value - observations[..., fit] @ projection.T
            total = total + self.fractional_error(error[..., :split], value[..., :split])
            total = total + self.fractional_error(error[..., split:], value[..., split:])
        return -total * .25

    def boundary_score(self, aligned):
        previous = aligned[:, :-PERIOD]
        current = aligned[:, PERIOD:]
        residual = power(current - previous)
        energy = power(current) + power(previous)
        before = self.observed_indices[:, :PERIOD]
        after = self.long_indices - PERIOD
        preceding = residual[:, before].sum(dim=-1) / energy[:, before].sum(dim=-1).clamp_min(1e-8)
        following = residual[:, after].sum(dim=-1) / energy[:, after].sum(dim=-1).clamp_min(1e-8)
        samples = aligned[:, self.long_indices]
        correlation = power(samples @ self.long_ref.conj())
        scale = power(samples).sum(dim=-1) * power(self.long_ref).sum()
        match = (correlation / scale.clamp_min(1e-8)).clamp(0, 1)
        return ((following + .05) / (preceding + .05)).log() + 2 * match

    def choose(self, score):
        far = score.masked_fill(self.near, -torch.inf)
        value, index = far.max(dim=-1)
        margin = value - score.masked_fill(~self.near, -torch.inf).amax(dim=-1)
        return index, margin

    def hypotheses(self, iq):
        observed = estimate(iq)
        window = observed.aligned[:, self.observed_indices]
        boundary = self.choose(self.boundary_score(observed.aligned))
        joint = self.choose(self.joint_score(window))
        return observed, window, boundary, joint

    def shift(self, iq, index, margin, threshold):
        delta = torch.where(margin > threshold, self.starts[index] - NOMINAL, 0)
        position = self.time + delta[:, None]
        valid = (position >= 0) & (position < iq.shape[-1])
        mirrored = torch.where(position < 0, -position, position.clamp_max(2 * (iq.shape[-1] - 1) - position))
        return iq.gather(1, mirrored), position, valid, delta

    def margins(self, iq):
        _, _, boundary, joint = self.hypotheses(iq)
        return torch.stack((boundary[1], joint[1]), dim=1)

    def restore(self, iq, thresholds):
        observed, window, (boundary, margin_b), (joint, margin_j) = self.hypotheses(iq)
        row = torch.arange(len(iq), device=iq.device)

        staged, pos_b, _, delta_b = self.shift(iq, boundary, margin_b, thresholds['boundary'])
        gain = window[row, boundary, -LONG_MATCH:] @ self.long_ref.conj() / power(self.long_ref).sum()
        center = (BEGIN + END) / 2
        rotation_b = phase(observed.slope[:, None] * (pos_b - center))
        edge_b = gain[:, None] * self.edge_single * rotation_b
        boundary_protocol = torch.where(pos_b >= iq.shape[-1], edge_b, staged)

        mirrored, pos_j, valid_j, delta_j = self.shift(iq, joint, margin_j, thresholds['joint'])
        response = window[row, joint] @ self.inverse.T
        rotation_j = phase(observed.slope[:, None] * (pos_j - center))
        edge_j = (response @ self.edge_basis.T) * rotation_j
        joint_protocol = torch.where(valid_j, mirrored, edge_j)
        return (dict(boundary_protocol=boundary_protocol, joint_mirror=mirrored, joint_protocol=joint_protocol),
                dict(boundary_margin=margin_b, joint_margin=margin_j, boundary_shift=delta_b, joint_shift=delta_j))
