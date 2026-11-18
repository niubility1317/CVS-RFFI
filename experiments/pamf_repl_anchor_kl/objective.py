"""Change only U KL, reusing the original accepted mask and source forwards."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from experiments.pamf_repl_uaux_soft import objective as soft
from experiments.pamf_ssl_research import hard_pl_objective as hard


def directional_kl(student, teacher, temperature):
    log_student = (student / temperature).log_softmax(-1)
    log_teacher = (teacher.detach() / temperature).log_softmax(-1)
    return F.kl_div(log_student, log_teacher, reduction='none', log_target=True).sum(-1) * temperature**2


def anchor_kl(ordinary, raw, teacher_logits, accepted, recipe):
    """Return the full-population masked mean, including zero-weight rejected rows."""
    source = recipe['kl_teacher']
    t = recipe['kl_temperature']
    if source == 'cross':
        def symmetric(a, b):
            p, q = a.log_softmax(-1), b.log_softmax(-1)
            return .5 * ((p.exp() - q.exp()) * (p - q)).sum(-1)
        per_sample = .5 * (symmetric(ordinary[0], raw[0]) + symmetric(ordinary[1], raw[1]))
    elif source == 'raw_train':
        per_sample = .5 * (directional_kl(ordinary[0], raw[0], t) + directional_kl(ordinary[1], raw[1], t))
    else:
        if source != 'raw_eval' or teacher_logits is None:
            raise ValueError('Expected the original detached raw evaluation logits')
        outputs = {'ordinary': ordinary, 'raw': raw, 'both': (*ordinary, *raw)}[recipe['kl_students']]
        per_sample = sum(directional_kl(logits, teacher_logits, t) for logits in outputs) / len(outputs)
    return (per_sample * accepted).mean()


class AnchorKLREPL(soft.SoftREPL):
    def __init__(self, config):
        super().__init__(config)
        self.anchor_pairs = []
        self.anchor_eval_logits = []

    def loss(self, student, device):
        if self.anchor_pairs or self.anchor_eval_logits:
            raise ValueError('Unconsumed source KL inputs')

        def capture(module, args, output):
            if not module.training and not torch.is_grad_enabled():
                self.anchor_eval_logits.append(output.detach())

        handle = student.register_forward_hook(capture)
        try:
            super().loss(student, device)
        finally:
            handle.remove()
        if len(self.anchor_eval_logits) != 1 or len(self.anchor_pairs) != 4:
            raise ValueError('Expected one raw pseudo-label forward and four original paired losses')
        ordinary, ordinary_mask = self.anchor_pairs[0]
        raw, raw_mask = self.anchor_pairs[3]
        if not torch.equal(ordinary_mask, raw_mask):
            raise ValueError('Ordinary and raw must share the original U acceptance mask')
        teacher = self.anchor_eval_logits[0]
        if teacher.shape != ordinary[0].shape or raw[0].shape != teacher.shape:
            raise ValueError('The KL teacher and students must align by U record')
        recipe = self.c['tuning_recipe']
        kl = .8 * recipe['weight'] * recipe['kl_factor'] * anchor_kl(ordinary, raw, teacher, ordinary_mask, recipe)
        self.parts['pair_kl'] = kl
        effective = sum(self.parts.values())
        self.metrics.update(HPL_pre_route_U_loss=self.pending_gradient_loss.detach(),
                            HPL_effective_CE_loss=effective.detach()*0.,
                            HPL_effective_KL_loss=kl.detach(),
                            HPL_effective_aux_loss=effective.detach(),
                            HPL_anchor_population=ordinary_mask.numel(),
                            HPL_anchor_acceptance=ordinary_mask.float().mean().detach(),
                            HPL_loss=effective.detach(),
                            HPL_base_loss=(effective / self.settings['weight']).detach())
        self.pending_gradient_loss = effective
        self.anchor_pairs.clear()
        self.anchor_eval_logits.clear()
        return effective.detach()*0.


@contextmanager
def installed(config, holder):
    original_class = soft.SoftREPL
    soft.SoftREPL = AnchorKLREPL
    try:
        with soft.installed(config, holder):
            priority_pair = hard.masked_pair

            def capture_pair(a, b, targets, accepted, kl_weight):
                result = priority_pair(a, b, targets, accepted, kl_weight)
                holder['objective'].anchor_pairs.append(((a, b), accepted))
                return result

            hard.masked_pair = capture_pair
            try:
                yield
            finally:
                hard.masked_pair = priority_pair
    finally:
        soft.SoftREPL = original_class
