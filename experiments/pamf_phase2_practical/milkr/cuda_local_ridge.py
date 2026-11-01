"""Batched FP64 CUDA evaluation of an already fitted local-ridge state.

Only the current query is scored against the frozen support. No query-derived
quantity is shared with any other query or fed back into the fitted state.
"""
import numpy as np
import torch

from . import d92_branch_interaction as interaction
from .d92_branch_local_ridge import _score_local


class CudaLocalRidge:
    def __init__(self, state, device='cuda:0', pair_budget=2048):
        self.state = state
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('CUDA device required')
        if pair_budget < 1:
            raise ValueError('pair_budget must be positive')
        self.pair_budget = int(pair_budget)
        def tensor(value):
            # The N607 PyTorch 2.1 build cannot use its NumPy 2 array bridge.
            return torch.tensor(np.asarray(value).tolist(), dtype=torch.float64, device=self.device)
        self.background = tensor(state.support_background)
        self.auxiliary = tensor(state.support_auxiliary)
        self.alpha = tensor(state.alpha)
        self.reference = tensor(state.reference_kernel)
        self.mean = tensor(state.center_mean)
        self.cpu_tie_rows = 0

    def score(self, *, z_id, fft, t_emb, f_emb, pa_local):
        b, a = interaction._blocks(z_id, fft, t_emb, f_emb, pa_local, allow_empty=True)
        state = self.state
        if not len(b) or state.trace_scale is None:
            return np.zeros((len(b), len(state.classes)), dtype=np.float64)
        result = []
        size = max(1, self.pair_budget // len(self.background))
        for start in range(0, len(b), size):
            qb = torch.tensor(b[start:start+size].tolist(), dtype=torch.float64, device=self.device)[:, None, :]
            qa = torch.tensor(a[start:start+size].tolist(), dtype=torch.float64, device=self.device)[:, None, :]
            db = qb-self.background[None, :, :]
            da = qa-self.auxiliary[None, :, :]
            # The rank-2 QR product R@[a_i, a_i-a_j] is evaluated by
            # Gram-Schmidt without forming the huge outer-product features.
            r11 = torch.linalg.vector_norm(db, dim=-1)
            q1 = db / torch.where(r11 == 0, 1., r11)[..., None]
            r12 = (q1*self.background[None, :, :]).sum(-1)
            residual = self.background[None, :, :]-q1*r12[..., None]
            r22 = torch.linalg.vector_norm(residual, dim=-1)
            p = r11[..., None]*qa+r12[..., None]*da
            da2 = (da*da).sum(-1)
            distance = (db*db).sum(-1)+da2+(p*p).sum(-1)+r22*r22*da2
            same = (db == 0).all(-1) & (da == 0).all(-1)
            distance = torch.where(same, 0., distance)
            if not bool(torch.isfinite(distance).all()) or bool(((~same) & (distance == 0)).any()):
                raise FloatingPointError('Invalid or underflowed CUDA interaction distance')
            if state.bandwidth_tau == 0:
                raw = -(distance != 0).to(torch.float64)
            else:
                raw = torch.expm1(-distance/state.bandwidth_tau)
            diff = raw-raw[:, :1]-self.reference+state.reference_self
            value = (state.trace_scale*(diff-diff.mean(1, keepdim=True)-self.mean+state.center_grand))@self.alpha
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite CUDA local-ridge score')
            scores = np.asarray(value.cpu().tolist(), dtype=np.float64)
            # Use the exact original routine for very close decisions, where
            # floating-point rearrangement could otherwise change a winner.
            if scores.shape[1] > 1:
                ordered = np.partition(scores, -2, axis=1)
                gap = ordered[:, -1]-ordered[:, -2]
                ambiguous = gap <= 1e-9*np.maximum(1., np.max(np.abs(scores), axis=1))
                # The runner also records the prediction restricted to the
                # original six classes, so its closest pair needs protection.
                if scores.shape[1] >= 6:
                    old = scores[:, :6]
                    old_ordered = np.partition(old, -2, axis=1)
                    ambiguous |= (old_ordered[:, -1]-old_ordered[:, -2]
                                  <= 1e-9*np.maximum(1., np.max(np.abs(old), axis=1)))
                if ambiguous.any():
                    sl = slice(start, start+len(scores))
                    scores[ambiguous] = _score_local(b[sl][ambiguous], a[sl][ambiguous],
                        state.support_background, state.support_auxiliary, state.alpha,
                        state.reference_kernel, state.reference_self, state.center_mean,
                        state.center_grand, state.bandwidth_tau, state.trace_scale)
                    self.cpu_tie_rows += int(ambiguous.sum())
            result.append(scores)
        return np.concatenate(result)
