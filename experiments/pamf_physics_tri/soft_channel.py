"""Use continuous branch evidence to set source-U physical-view strength."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .practical_gate import capture_pair, installed as practical_scope, view_strength
from .recovery import branch_scores
from .single_rx import SingleRxDay


def physical_weights(raw, changed, labels, l_day, u_day, pseudo, classes, mode):
    scores, altered, valid = branch_scores(raw, changed, labels, l_day, u_day, classes)
    votes_raw = (scores.argmax(2) == pseudo[:, None]).sum(1)
    votes_changed = (altered.argmax(2) == pseudo[:, None]).sum(1)
    weight = .5 * (votes_raw + votes_changed).float() / scores.shape[1]
    if mode == 'vote_stability':
        n = len(labels)
        similarity = torch.stack([F.cosine_similarity(raw[k][n:].float(), changed[k][n:].float(), dim=1, eps=1e-8)
                                  for k in raw], dim=1).clamp(0., 1.).mean(1)
        weight = weight * similarity
    elif mode != 'vote':
        raise ValueError('Unknown source-channel evidence rule')
    return weight * valid, valid & (votes_raw >= 4) & (votes_changed >= 4)


class SoftChannel(SingleRxDay):
    def __init__(self, c):
        super().__init__(c)
        self.soft_calls = self.soft_selected = 0
        self.relative_strength_sum = 0.

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        if self.labeled is None:
            raise ValueError('Current source labeled anchors required')
        indices = physical.nonzero(as_tuple=True)[0]
        meta = {k: v[indices] if torch.is_tensor(v) and v.ndim and len(v) == len(x) else v
                for k, v in metadata.items()}
        simulated = student._cd_practical(x[indices], meta, epoch, step)
        labeled, labels, days = self.labeled
        raw, changed = capture_pair(student, torch.cat((labeled, x[indices])), torch.cat((labeled, simulated)))
        weights, certified = physical_weights(raw, changed, labels, days, metadata['day_i'][indices],
                                              pseudo[indices], self.c['num_classes'], self.c['soft_channel'])
        offered = accepted[indices]
        strength = view_strength(self.c['practical_gate'], epoch)
        selected = offered & (weights > 0) if strength else torch.zeros_like(offered)
        mixed = ordinary.clone()
        mixed[indices] = x[indices]
        chosen = indices[selected]
        mixed[chosen] = x[chosen] + (strength * weights[selected])[:, None, None] * (simulated[selected] - x[chosen])
        self.as_calls += 1
        self.as_offered += int(offered.sum())
        self.as_certified += int((offered & certified).sum())
        self.as_active_certified += int((offered & certified).sum()) if strength else 0
        self.as_used += int(selected.sum())
        total_weight = float(weights[selected].sum())
        self.relative_strength_sum += total_weight
        self.strength_weighted_uses += strength * total_weight
        self.soft_calls += 1
        self.soft_selected += int(selected.sum())
        return mixed

    def state_dict(self):
        state = super().state_dict()
        if self.soft_calls != self.calls or self.soft_selected != self.as_used:
            raise ValueError('One bounded soft-channel assignment per source-U loss required')
        state['U_soft_channel'] = dict(mode=self.c['soft_channel'], calls=self.soft_calls,
            selected=self.soft_selected, relative_strength_sum=self.relative_strength_sum,
            base_strength=self.c['practical_gate']['strength'], hard_pseudo_mask_unchanged=True)
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .soft_channel_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=SoftChannel):
        yield
