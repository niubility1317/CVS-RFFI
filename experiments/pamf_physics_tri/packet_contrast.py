"""Source physical same-packet fusion contrast with pseudo-class-aware negatives."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .fused_physics import FusedPhysics
from .practical_gate import installed as practical_scope


def packet_contrast_loss(raw, changed, pseudo, accepted, selected, temperature=.2):
    n = len(raw)
    if (raw.shape != changed.shape or raw.ndim != 2 or pseudo.shape != (n,) or accepted.shape != (n,)
            or selected.shape != (n,) or accepted.dtype != torch.bool or selected.dtype != torch.bool
            or temperature <= 0 or bool((selected & ~accepted).any())):
        raise ValueError('Aligned source-U fusion, accepted labels and physical slots required')
    negative = (pseudo[:, None] != pseudo[None, :]) & accepted[None, :]
    valid = selected & negative.any(1)
    if not bool(valid.any()):
        return changed.sum() * 0., 0
    scores = F.normalize(changed, dim=1, eps=1e-8) @ F.normalize(raw.detach(), dim=1, eps=1e-8).T
    scores = scores / temperature
    candidates = negative | torch.eye(n, dtype=torch.bool, device=raw.device)
    terms = torch.logsumexp(scores.masked_fill(~candidates, -torch.inf), dim=1) - scores.diagonal()
    return (terms * valid).mean(), int(valid.sum())


class PacketContrast(FusedPhysics):
    def __init__(self, c):
        super().__init__(c)
        self.packet_labels = None
        self.packet_calls = self.packet_anchors = 0

    def include_fused_cosine(self):
        return self.c['packet_contrast']['cosine']

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self._rxe_forward_index == 0:
            if self.packet_labels is not None:
                raise ValueError('Only one source U identity set per loss allowed')
            self.packet_labels = pseudo.detach(), accepted.detach()
        return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)

    def additional_fusion_loss(self, views, selected):
        if self.packet_labels is None:
            raise ValueError('Accepted source-U pseudo identities required')
        pseudo, accepted = self.packet_labels
        values, counts = zip(*(packet_contrast_loss(views[i], views[i + 2], pseudo, accepted, selected,
                                                   temperature=self.c['packet_contrast']['temperature'])
                               for i in range(2)))
        loss = torch.stack(values).mean()
        if self.c.get('hard_pl_smoke'):
            # The bound student is supplied by the loss wrapper only for this read-only gradient check.
            named = [(name, p) for name, p in self.packet_student.named_parameters() if p.requires_grad and
                     ('.sequence_pool.' in name or '.response_projection.' in name)]
            gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
            norms = {'signature': 0., 'reference': 0.}
            for (name, _), gradient in zip(named, gradients):
                if gradient is not None:
                    if not bool(torch.isfinite(gradient).all()):
                        raise ValueError('Finite source packet-contrast gradients required')
                    group = 'signature' if '.sequence_pool.' in name else 'reference'
                    norms[group] += float(gradient.detach().double().square().sum())
            self.packet_smoke_gradients = {key: value ** .5 for key, value in norms.items()}
        self.packet_calls += 1
        self.packet_anchors += sum(counts)
        self.metrics.update(HPL_packet_contrast=loss.detach(), HPL_packet_anchors=self.packet_anchors)
        return loss

    def loss(self, student, device):
        self.packet_student = student if self.c.get('hard_pl_smoke') else None
        try:
            return super().loss(student, device)
        finally:
            self.packet_labels = self.packet_student = None

    def state_dict(self):
        state = super().state_dict()
        if (self.packet_labels is not None or self.packet_student is not None or self.packet_calls != self.calls
                or not 0 <= self.packet_anchors <= 2 * self.fusion_selected):
            raise ValueError('One same-packet contrast per U loss and no persistent capture required')
        state['U_packet_contrast'] = dict(recipe=self.c['packet_contrast'], calls=self.packet_calls,
            anchors=self.packet_anchors, positive='same source packet raw/physical',
            negative='accepted different pseudo class', teacher_detached=True, extra_student_forwards=0)
        if self.c.get('hard_pl_smoke'):
            state['U_packet_contrast']['smoke_gradients'] = self.packet_smoke_gradients
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .packet_contrast_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=PacketContrast):
        yield
