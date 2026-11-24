import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_restoration.models import complex_iq, pair
from .operators import Action, soft_map

KERNEL_LENGTH = 64
SCALE = .1169459182403555


class ActionEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core(incoming=4, outgoing=4)
        self.global_head = nn.Linear(128, 2*KERNEL_LENGTH+1)
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        impulse = torch.zeros(KERNEL_LENGTH)
        impulse[24] = 1
        self.register_buffer('initial_kernel', impulse)

    def _encode(self, received, coarse):
        if received.ndim != 3 or received.shape[1:] != (2, 256) or coarse.shape != received.shape:
            raise ValueError('Only current received/coarse IQ2x256 is accepted')
        return self.core.encoder(torch.cat((received, coarse), 1))

    @staticmethod
    def _variance(logit):
        # Fixed physical range; output IQ itself is never clipped.
        log_variance = math.log(1e-8)+(math.log(1.)-math.log(1e-8))*torch.sigmoid(logit+.351)
        return log_variance.exp()

    def variance_only(self, received, coarse):
        encoded = self._encode(received, coarse)[-1]
        return self._variance(self.global_head(encoded.mean(-1))[:, -1])

    def forward(self, received, coarse):
        features = self._encode(received, coarse)
        value = features[-1]
        global_value = self.global_head(value.mean(-1))
        kernel = torch.complex(global_value[:, :KERNEL_LENGTH]+self.initial_kernel,
                               global_value[:, KERNEL_LENGTH:2*KERNEL_LENGTH])
        kernel = kernel/kernel.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
        for block, skip in zip(self.core.decode, features[-2::-1]):
            value = block(torch.cat((F.interpolate(value, size=skip.shape[-1], mode='linear',
                                                    align_corners=False), skip), 1))
        coefficients = self.core.out(value)
        main = 1+torch.complex(coefficients[:, 0], coefficients[:, 1])
        image = torch.complex(coefficients[:, 2], coefficients[:, 3])
        return Action(kernel, main, image), self._variance(global_value[:, -1])


class ActionRestorer(nn.Module):
    def __init__(self, coarse):
        super().__init__()
        self.coarse = coarse.requires_grad_(False).eval()
        self.estimator = ActionEstimator()

    def train(self, mode=True):
        super().train(mode)
        self.coarse.eval()
        return self

    def estimate(self, received):
        with torch.no_grad():
            base = self.coarse(received)
        action, variance = self.estimator(received, base)
        return base, action, variance

    def forward(self, received):
        base, action, variance = self.estimate(received)
        return pair(soft_map(action, complex_iq(received), complex_iq(base), variance, SCALE**2, 6))
