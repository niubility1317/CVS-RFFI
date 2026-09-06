"""Source-only ECRS objectives and explicit successful-update training state.

Physical outputs must be detached before the caller's response encoder. This
module never receives U transmitter truth; U pairs are aligned by their caller.
"""
from __future__ import annotations

import copy
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _zero(x):
    # Selecting an empty slice retains autograd without propagating NaN inputs.
    return x.reshape(-1)[:0].sum()


def _label_mask(labels, label_mask, classes=None):
    valid = labels >= 0
    if classes is not None:
        valid = valid & (labels < classes)
    if label_mask is not None:
        valid = valid & label_mask.to(device=labels.device, dtype=torch.bool)
    return valid


def masked_response_ce(logits, labels, label_mask=None):
    """Independent plain CE over valid L only; return (mean, sample count)."""
    valid = _label_mask(labels, label_mask, logits.shape[-1])
    count = int(valid.sum().item())
    return (F.cross_entropy(logits[valid].float(), labels[valid].long())
            if count else _zero(logits)), count


def cross_rx_triplet_loss(z, labels, rx, day, view, label_mask=None, margin=0.2):
    """All legal triplets, averaged within anchor then across valid anchors.

    Positives: same TX, different RX. Negatives: different TX and same
    RX/day/view as anchor. An anchor lacking either set contributes no weight.
    Sorted negative distances and prefix sums implement the exact all-pair
    hinge mean with O(B^2) storage, instead of materializing B^3 triplets.
    """
    valid = _label_mask(labels, label_mask)
    legal = valid[:, None] & valid[None, :]
    positive = legal & (labels[:, None] == labels[None, :]) & (rx[:, None] != rx[None, :])
    negative = legal & (labels[:, None] != labels[None, :])
    for meta in (rx, day, view):
        negative = negative & (meta[:, None] == meta[None, :])
    pc, nc = positive.sum(1), negative.sum(1)
    anchors = (pc > 0) & (nc > 0)
    count = int(anchors.sum().item())
    if not count:
        return _zero(z), 0
    unit = F.normalize(z.float(), dim=-1)
    dist = 1 - unit @ unit.T
    # +inf padding is excluded from the prefix sum to avoid inf-inf backward.
    sorted_neg = dist.masked_fill(~negative, float('inf')).sort(dim=1).values
    prefix = F.pad(sorted_neg.masked_fill(~torch.isfinite(sorted_neg), 0).cumsum(1), (1, 0))
    threshold = (dist + float(margin)).contiguous()
    active = torch.searchsorted(sorted_neg.contiguous(), threshold, right=False)
    hinge_sum = active * threshold - prefix.gather(1, active)
    per_anchor = (hinge_sum * positive).sum(1) / (pc * nc).clamp_min(1)
    return per_anchor[anchors].mean(), count


def one_way_u_consistency(student_leo, teacher_clean, pair_mask=None):
    """One-to-one clean/LEO U pairs; mean by physical sample, teacher detached."""
    if student_leo.shape != teacher_clean.shape:
        raise ValueError('U clean/LEO embeddings must have identical aligned shapes')
    valid = torch.ones(student_leo.shape[0], device=student_leo.device, dtype=torch.bool)
    if pair_mask is not None:
        valid &= pair_mask.to(device=student_leo.device, dtype=torch.bool)
    count = int(valid.sum().item())
    if not count:
        return _zero(student_leo), 0
    return (1 - F.cosine_similarity(student_leo[valid].float(),
                                    teacher_clean[valid].detach().float(), dim=-1)).mean(), count


def assemble_ecrs_losses(*, logits, z_resp, labels, label_mask=None,
                         rx=None, day=None, view=None, student_leo=None,
                         teacher_clean=None, pair_mask=None, weights=None,
                         enabled=None, margin=0.2):
    """Return additive named losses and detached execution telemetry.

    Keys are ``resp_ce``, ``cross_rx``, ``u_pair``; defaults are .15/.05/.03.
    A disabled/zero-weight objective is not evaluated. ``executed`` means the
    objective had a nonempty legal set, even when its hinge loss equals zero.
    """
    weights = dict({'resp_ce': .15, 'cross_rx': .05, 'u_pair': .03}, **(weights or {}))
    enabled = enabled or {}
    losses, telemetry = {}, {}
    total = _zero(z_resp)
    for name in ('resp_ce', 'cross_rx', 'u_pair'):
        configured = bool(enabled.get(name, True))
        weight = float(weights[name])
        loss, count = _zero(z_resp), 0
        reason = 'disabled' if not configured else 'zero_weight' if weight == 0 else ''
        if configured and weight != 0:
            if name == 'resp_ce':
                loss, count = masked_response_ce(logits, labels, label_mask)
            elif name == 'cross_rx':
                if any(x is None for x in (rx, day, view)):
                    raise ValueError('cross_rx requires RX/day/view metadata')
                loss, count = cross_rx_triplet_loss(z_resp, labels, rx, day, view, label_mask, margin)
            elif student_leo is not None and teacher_clean is not None:
                loss, count = one_way_u_consistency(student_leo, teacher_clean, pair_mask)
            else:
                reason = 'no_u_pairs'
            if not count and not reason:
                reason = 'empty_legal_set'
        weighted = loss * weight
        total = total + weighted
        losses[name] = loss
        telemetry[name] = {'configured': configured, 'executed': bool(count),
                           'valid_count': count, 'raw_loss': float(loss.detach()),
                           'weighted_loss': float(weighted.detach()), 'reason': reason}
    return {'total': total, 'losses': losses, 'telemetry': telemetry}


class ResponseEMA(nn.Module):
    """Frozen eval-only teacher; updates occur only via explicit update()."""
    def __init__(self, student, decay=0.99):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must lie in [0,1)')
        self.decay = float(decay)
        self.module = copy.deepcopy(student).eval().requires_grad_(False)
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        # Parent model.train() must never enable teacher BN/dropout mutation.
        super().train(False)
        self.module.eval()
        return self

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self.module.eval()
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def update(self, student, successful=True):
        if not successful:
            return False
        current = student.state_dict()
        target = self.module.state_dict()
        if current.keys() != target.keys():
            raise ValueError('EMA and student state structures differ')
        for name, value in target.items():
            if value.is_floating_point() or value.is_complex():
                value.lerp_(current[name].to(value), 1 - self.decay)
            else:
                value.copy_(current[name])
        self.updates.add_(1)
        return True

    def get_extra_state(self):
        return {'decay': self.decay}

    def set_extra_state(self, state):
        self.decay = float(state['decay'])


class EffectiveStepLR:
    """Per-group LR clocks; call after optimizer/scaler step, before zero_grad.

    ``successful`` must reflect actual GradScaler/optimizer update success.
    Gradients still need to be present at this call. Groups without gradients
    do not advance. Zero gradients count because momentum/decay may update them.
    Existing optimizer groups and optimizer state are never replaced.
    """
    def __init__(self, optimizer, group_configs):
        self.optimizer = optimizer
        self.configs = {int(i): dict(c) for i, c in group_configs.items()}
        self.steps = {i: 0 for i in self.configs}
        for i, config in self.configs.items():
            if not 0 <= i < len(optimizer.param_groups):
                raise ValueError('Unknown optimizer parameter group')
            config.setdefault('peak_lr', optimizer.param_groups[i]['lr'])
            config.setdefault('min_lr', 0.)
            config.setdefault('warmup_steps', 0)
            config.setdefault('total_steps', 1)
            if not 0 <= config['min_lr'] <= config['peak_lr'] or not 0 <= config['warmup_steps'] < config['total_steps']:
                raise ValueError('Invalid effective-step LR configuration')
            optimizer.param_groups[i]['lr'] = self._lr(i, 0)

    def _lr(self, index, completed):
        c = self.configs[index]
        # LR at count k is the LR for the next successful update (k+1).
        if completed < c['warmup_steps']:
            return c['peak_lr'] * (completed + 1) / c['warmup_steps']
        progress = min(1., max(0., (completed - c['warmup_steps']) /
                              (c['total_steps'] - c['warmup_steps'])))
        return c['min_lr'] + .5 * (c['peak_lr'] - c['min_lr']) * (1 + math.cos(math.pi * progress))

    def step(self, successful=True, active_groups=None):
        advanced = []
        if not successful:
            return advanced
        allowed = None if active_groups is None else set(active_groups)
        for i in self.configs:
            if allowed is not None and i not in allowed:
                continue
            grads = [p.grad for p in self.optimizer.param_groups[i]['params'] if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                continue
            self.steps[i] += 1
            self.optimizer.param_groups[i]['lr'] = self._lr(i, self.steps[i])
            advanced.append(i)
        return advanced

    def state_dict(self):
        return copy.deepcopy({'configs': self.configs, 'steps': self.steps})

    def load_state_dict(self, state):
        configs = {int(i): dict(c) for i, c in state['configs'].items()}
        if set(configs) != set(self.configs):
            raise ValueError('Effective-step scheduler group mismatch')
        self.configs = configs
        self.steps = {int(i): int(v) for i, v in state['steps'].items()}
        for i in self.configs:
            self.optimizer.param_groups[i]['lr'] = self._lr(i, self.steps[i])


def capture_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
