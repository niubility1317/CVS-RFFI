"""FP64 CUDA evaluation of fixed direct-C states; CPU fallback near ties."""
import numpy as np
import torch

from . import orbit_ensemble as orbit
from .prior_fusion import PRIOR_WEIGHT
from ..fast_transfer import to_tensor, to_numpy


class CudaDirectC:
    def __init__(self, state, device='cuda:0', pair_budget=16384):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('CUDA device required')
        if pair_budget < 1:
            raise ValueError('Positive pair budget required')
        self.state = state
        self.pair_budget = int(pair_budget)
        self.support = self._tensor_dict(state.support)
        self.prototypes = None if state.prototypes is None else self._tensor_dict(state.prototypes)
        self.alpha = to_tensor(state.alpha, self.device, np.float64)
        self.intercept = to_tensor(state.intercept, self.device, np.float64)
        self.cpu_tie_rows = 0

    def _tensor_dict(self, values):
        return {k: to_tensor(v, self.device, np.float64) for k, v in values.items()}

    @staticmethod
    def _similarity(left, right, prefix, geometry):
        sig = left[prefix + 'sig_views'] @ right[prefix + 'sig_views'].T
        if geometry == 'signature':
            return sig
        z = left[prefix + 'z_id'] @ right[prefix + 'z_id'].T
        if geometry == 'signature_identity':
            return .5 * sig + .5 * z
        if geometry != 'signature_historical':
            raise ValueError('Unknown source kernel geometry')
        fft = left[prefix + 'fft'] @ right[prefix + 'fft'].T
        aux = sum(left[prefix + name] @ right[prefix + name].T for name in ('t_emb', 'f_emb', 'pa_local')) / 3
        bg = (z + 16 * fft) / 17
        interaction = (bg + aux + bg * aux) / 3
        return .5 * sig + .5 * interaction

    def _similarities(self, left, right):
        geometry = self.state.config[0]
        return tuple(self._similarity(left, right, prefix, geometry) for prefix in ('C4_', ''))

    @staticmethod
    def _kernel(sim, tau, mix, scales):
        distance = torch.clamp(2. - 2. * sim, min=0.)
        levels = (1.,) if scales == 'single' else (.25, 1., 4.)
        local = sum(torch.exp(-distance / (tau * scale)) for scale in levels) / len(levels)
        return mix * sim + (1. - mix) * local

    def score(self, features):
        values = orbit.normalize(features)
        count = len(values['z_id'])
        if count == 0:
            raise ValueError('Empty query')
        chunk = max(1, self.pair_budget // len(self.state.support['z_id']))
        result = []
        for start in range(0, count, chunk):
            x = self._tensor_dict({k: v[start:start + chunk] for k, v in values.items()})
            sims = self._similarities(x, self.support)
            _, _, mix, scales = self.state.config
            kernel = sum(self._kernel(sim, tau, mix, scales) for sim, tau in zip(sims, self.state.tau)) / 2.
            p = x['teacher_probs']
            if self.prototypes is not None:
                sim_proto = self._similarities(x, self.prototypes)
                affinity = sum(torch.exp(-torch.clamp(2. - 2. * sim, min=0.) / tau)
                               for sim, tau in zip(sim_proto, self.state.tau)) / 2.
                p = p * affinity
                total = p.sum(dim=1, keepdim=True)
                if not bool(torch.isfinite(total).all()) or bool((total <= 0).any()):
                    raise FloatingPointError('Source-prototype prior has no probability mass')
                p = p / total
            result_part = kernel @ self.alpha + self.intercept
            result_part[:, :6] += PRIOR_WEIGHT * (p - 1. / 6.)
            if not bool(torch.isfinite(result_part).all()):
                raise FloatingPointError('Nonfinite CUDA direct-C score')
            scores = to_numpy(result_part, np.float64)
            ordered = np.partition(scores, -2, axis=1)
            gap = ordered[:, -1] - ordered[:, -2]
            old = np.partition(scores[:, :6], -2, axis=1)
            old_gap = old[:, -1] - old[:, -2]
            scale = 1e-9 * np.maximum(1., np.max(np.abs(scores), axis=1))
            ambiguous = (gap <= scale) | (old_gap <= scale)
            if ambiguous.any():
                chosen = np.flatnonzero(ambiguous) + start
                scores[ambiguous] = self.state.score({k: v[chosen] for k, v in features.items()})
                self.cpu_tie_rows += int(ambiguous.sum())
            result.append(scores)
        return np.concatenate(result)
