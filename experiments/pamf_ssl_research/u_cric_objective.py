"""Training-only U feature hooks; confidence and pair targets never receive gradients."""
import torch
from torch.nn import functional as F
from experiments.cric_dura_comparison.f_model import PREFIX
from .objective import Objective


def soft_cross_receiver_contrast(h, probability, weight, receivers, allowed_receivers, temperature):
    n = len(h)
    p, w = probability.detach(), weight.detach()
    if receivers.shape != (n,) or set(receivers.tolist()) - set(allowed_receivers):
        raise ValueError('Only known source receivers may define U pairs')
    if (p.ndim != 2 or len(p) != n or w.shape != (n,) or not bool(torch.isfinite(p).all())
            or not bool(torch.isfinite(w).all()) or bool((p < 0).any())
            or not torch.allclose(p.sum(1), torch.ones(n, device=p.device), atol=1e-5)
            or bool(((w < 0) | (w > 1)).any()) or temperature <= 0):
        raise ValueError('Detached normalized probabilities and bounded confidence required')
    if h.ndim == 3:
        h = torch.cat((h.mean(-1), h.std(-1, unbiased=False)), 1)
    if h.ndim != 2:
        raise ValueError('Actual shallow/middle sequence maps required')
    # Use the same descriptor and detached channel standardization as native L CRIC.
    z = (h - h.detach().mean(0)) / h.detach().std(0, unbiased=False).clamp_min(1e-4)
    z = F.normalize(z, dim=1, eps=1e-4)
    same = p.argmax(1)[:, None] == p.argmax(1)[None, :]
    other_rx = receivers[:, None] != receivers[None, :]
    similarity = (p @ p.T).clamp(0., 1.)
    positive = same & other_rx
    negative = ~same
    positive_weight = similarity * positive * w[None, :]
    negative_weight = (1-similarity) * negative * w[None, :]
    mass = positive_weight.sum(1)
    valid = (mass > 1e-6) & (negative_weight.sum(1) > 1e-6) & (w > 0)
    detail = dict(valid_anchors=int(valid.sum()), total_anchors=n)
    if not bool(valid.any()):
        return z.sum()*0., detail
    scores = (z @ z.T / temperature)[valid]
    candidate_weight = (positive_weight + negative_weight)[valid]
    denominator = torch.logsumexp((scores + candidate_weight.clamp_min(1e-30).log())
                                  .masked_fill(candidate_weight <= 0, -torch.inf), 1)
    positive_score = (scores * positive_weight[valid]).sum(1) / mass[valid]
    per_anchor = denominator - positive_score - mass[valid].log()
    # Full batch denominator preserves the low-confidence/absent-evidence attenuation.
    return (w[valid] * per_anchor).sum()/n, detail


class UnlabeledCRICObjective(Objective):
    def __init__(self, student, config):
        super().__init__(student, config)
        self.u_cric_calls = self.u_cric_seen = 0
        self.u_cric_anchors = {name: 0 for name in config['u_cric']['levels']}
        self._u_maps = None

    def loss(self, student, device):
        if self.current_epoch < self.c['u_cric']['start_epoch'] or self._u_maps is not None:
            raise ValueError('One E31+ U feature context required')
        maps, handles = {}, []
        self._u_maps = maps
        try:
            for name in self.c['u_cric']['levels']:
                def capture(module, inputs, output, name=name):
                    if name in maps or not torch.is_tensor(output):
                        raise ValueError('Exactly one actual U student map at each declared level required')
                    maps[name] = output
                    if self.c.get('u_cric_smoke'):
                        output.retain_grad()
                handles.append(student.get_submodule(PREFIX + name).register_forward_hook(capture))
            value = super().loss(student, device)
            self.metrics.update(self._u_cric_metrics)
            if self.c.get('u_cric_smoke'):
                self._smoke_maps = maps
            return value
        finally:
            for handle in handles: handle.remove()
            self._u_maps = None

    def unlabeled_feature_loss(self, output, target, weight, receivers):
        recipe = self.c['u_cric']
        if set(self._u_maps or {}) != set(recipe['levels']):
            raise ValueError('Actual U shallow and middle features missing')
        terms, details = [], {}
        for name in recipe['levels']:
            value, detail = soft_cross_receiver_contrast(self._u_maps[name], target, weight, receivers,
                                                        self.c['source_rxs'], recipe['temperature'])
            terms.append(value)
            details[name] = detail
            self.u_cric_anchors[name] += detail['valid_anchors']
        value = recipe['weight'] * torch.stack(terms).mean()
        self.u_cric_calls += 1
        self.u_cric_seen += len(target)
        self._u_cric_metrics = dict(SSL_U_CRIC_loss=value.detach(), SSL_U_CRIC_calls=self.u_cric_calls,
            SSL_U_CRIC_seen=self.u_cric_seen,
            **{'SSL_U_CRIC_valid_' + name: detail['valid_anchors'] for name, detail in details.items()})
        return value

    def state_dict(self):
        value = super().state_dict()
        value.update(u_cric=self.c['u_cric'], u_cric_calls=self.u_cric_calls,
                     u_cric_seen=self.u_cric_seen, u_cric_anchors=self.u_cric_anchors)
        return value

    def load_state_dict(self, state):
        if state.get('u_cric') and state['u_cric'] != self.c['u_cric']:
            raise ValueError('U-CRIC state belongs to another recipe')
        super().load_state_dict(state)
        self.u_cric_calls = state.get('u_cric_calls', 0)
        self.u_cric_seen = state.get('u_cric_seen', 0)
        self.u_cric_anchors = dict(state.get('u_cric_anchors', self.u_cric_anchors))
