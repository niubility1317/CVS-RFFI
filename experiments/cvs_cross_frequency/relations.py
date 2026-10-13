"""Packet-local full-spectrum relation residual with a matched power control.

Common constant phase is removed by the cross-products. Power normalization
uses an epsilon, so amplitude scaling is only approximately invariant. Arbitrary
channel phase, shifts and CFO are not removed by these fixed frequency edges.
"""
import math
import torch
from torch import nn

COMPLEX_RELATIONS = 'complex'
POWER_RELATIONS = 'power'
EPSILON = 1e-4
BINS = 256
FEATURE_DIM = 160


class CrossFrequencyRelations(nn.Module):
    """The two modes differ only in their six fixed observation channels."""
    def __init__(self, mode):
        super().__init__()
        if mode not in (COMPLEX_RELATIONS, POWER_RELATIONS):
            raise ValueError('Unregistered cross-frequency relation mode')
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(start_dim=1),
            nn.Linear(256, FEATURE_DIM), nn.LayerNorm(FEATURE_DIM, eps=EPSILON))

    @staticmethod
    def _validate(iq):
        if not torch.is_tensor(iq) or iq.ndim != 3 or iq.shape[1:] != (2, BINS) or len(iq) == 0:
            raise ValueError('Relations require nonempty IQ[B,2,256]')
        if iq.dtype not in (torch.float32, torch.float64):
            raise ValueError('Relations require float32 or float64 IQ')
        if not bool(torch.isfinite(iq).all()):
            raise ValueError('Relations reject nonfinite IQ')

    def relation_channels(self, iq):
        """Return [adjacent, mirror, offset16] x [real/imag or log-ratio/log-mean]."""
        self._validate(iq)
        with torch.autocast(device_type=iq.device.type, enabled=False):
            spectrum = torch.fft.fft(torch.complex(iq[:, 0], iq[:, 1]), dim=-1)
            raw_power = spectrum.real.square() + spectrum.imag.square()
            spectrum = spectrum / torch.sqrt(raw_power.mean(-1, keepdim=True) + EPSILON)
            power = spectrum.real.square() + spectrum.imag.square()
            k = torch.arange(BINS, device=iq.device)
            edges = ((k + 1) % BINS, (-k) % BINS, (k + 16) % BINS)
            values = []
            for target in edges:
                if self.mode == COMPLEX_RELATIONS:
                    relation = spectrum * spectrum[:, target].conj() / torch.sqrt(
                        (power + EPSILON) * (power[:, target] + EPSILON))
                    values.extend((relation.real, relation.imag))
                else:
                    first = torch.log(power + EPSILON)
                    second = torch.log(power[:, target] + EPSILON)
                    values.extend((torch.tanh(first - second), torch.tanh(.5 * (first + second))))
            return torch.stack(values, dim=1)

    def input_features(self, iq):
        relations = self.relation_channels(iq)
        angle = torch.arange(BINS, device=iq.device, dtype=iq.dtype) * (2. * math.pi / BINS)
        position = torch.stack((angle.sin(), angle.cos()), dim=0)
        return torch.cat((relations, position[None].expand(len(iq), -1, -1)), dim=1)

    def forward(self, iq):
        if torch.is_tensor(iq) and iq.device != self.encoder[0].weight.device:
            raise ValueError('IQ and relation encoder must share the same device')
        with torch.autocast(device_type=iq.device.type if torch.is_tensor(iq) else 'cpu', enabled=False):
            channels = self.input_features(iq)
            # A cast follows the existing parameter precision, never mutating
            # model state during a query. A .double() encoder stays FP64.
            return self.encoder(channels.to(dtype=self.encoder[0].weight.dtype))
