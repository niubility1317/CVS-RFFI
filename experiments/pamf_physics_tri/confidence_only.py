"""Use every source-U packet whose original hard pseudo-label confidence is >= .95."""
from collections import Counter

import torch
from torch.nn import functional as F

from .hybrid_contrast import HybridContrast
from .practical_gate import view_strength
from .source_prototype import SourcePrototype, capture_source_bank


def complete_prototype_loss(query, raw, bank, labels, bank_days, query_days, pseudo, selected, classes, temperature):
    """Use available cross-day L centers; fall back to this packet's raw view."""
    members = (F.one_hot(labels.long(), classes).T.bool().unsqueeze(0)
               & (query_days[:, None, None] != bank_days[None, None, :]))
    available = members.any(2)
    bank = F.normalize(bank.detach().float(), dim=1)
    centers = F.normalize(members.float() @ bank, dim=2)
    positions = torch.arange(len(query), device=query.device)
    missing_positive = ~available[positions, pseudo]
    centers[positions[missing_positive], pseudo[missing_positive]] = F.normalize(raw.detach().float(), dim=1)[missing_positive]
    available[positions, pseudo] = True
    scores = torch.einsum('ncd,nd->nc', centers, F.normalize(query.float(), dim=1)) / temperature
    scores = scores.masked_fill(~available, -torch.inf)
    terms = F.cross_entropy(scores, pseudo.long(), reduction='none')
    no_negative = available.sum(1) == 1
    pair = 1. - F.cosine_similarity(raw.detach().float(), query.float(), dim=1, eps=1e-8)
    loss = ((terms + no_negative * pair) * selected).mean()
    return loss, int((missing_positive & selected).sum()), int((no_negative & selected).sum())


class ConfidenceOnlyPrototype(SourcePrototype):
    def __init__(self, c):
        super().__init__(c)
        self.as_physical_used = 0
        self.accepted_classes = Counter()
        self.labeled_diagnostic = Counter()
        self.fallback_calls = 0
        self.prototype_fallbacks = Counter()

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        raw_l, labels, days = self.labeled
        fusion, branches = capture_source_bank(student, raw_l)
        self.prototype_bank = fusion, branches, labels.detach(), days.detach()
        self.prototype_days = metadata['day_i'].long().detach()
        indices = accepted.nonzero(as_tuple=True)[0]
        mixed = ordinary.clone()
        mixed[physical] = x[physical]
        strength = view_strength(self.c['practical_gate'], epoch)
        if len(indices):
            selected_meta = {key: value[indices] if torch.is_tensor(value) and value.ndim and len(value) == len(x) else value
                             for key, value in metadata.items()}
            simulated = student._cd_practical(x[indices], selected_meta, epoch, step)
            mixed[indices] = x[indices] + strength * (simulated - x[indices])
        self.record_practical_selection(indices, len(x))
        count = len(indices)
        self.as_calls += 1
        self.as_offered += count
        self.as_certified += count
        self.as_active_certified += count
        self.as_used += count
        self.as_physical_used += int((accepted & physical).sum())
        self.strength_weighted_uses += strength * count
        self.accepted_classes.update(pseudo[accepted].detach().cpu().tolist())
        if step % 50 == 0:
            modes = [(module, module.training) for module in student.modules()]
            try:
                student.eval()
                with torch.no_grad():
                    probability, prediction = student(raw_l[:32]).softmax(1).max(1)
            finally:
                for module, mode in modes:
                    module.training = mode
            use = probability >= .95
            self.labeled_diagnostic.update(count=int(use.sum()), correct=int((use & (prediction == labels[:32])).sum()))
        return mixed

    def additional_fusion_loss(self, views, selected):
        result = HybridContrast.additional_fusion_loss(self, views, selected)
        pseudo, accepted = self.hybrid_labels
        _, branches, labels, days = self.prototype_bank
        mask = selected & accepted
        losses = [complete_prototype_loss(self.hybrid_branches[i][key], self.hybrid_branches[i - 2][key],
                   branches[key], labels, days, self.prototype_days, pseudo, mask, self.c['num_classes'],
                   self.c['source_prototype']['temperature']) for i in (2, 3) for key in branches]
        prototype = torch.stack([item[0] for item in losses]).mean()
        self.prototype_calls += 1
        self.prototype_used += 2 * int(mask.sum())
        self.prototype_fallbacks.update(missing_positive=sum(item[1] for item in losses),
                                        no_negative=sum(item[2] for item in losses))
        self.metrics['HPL_source_prototype_branch'] = prototype.detach()
        if self.c.get('hard_pl_smoke'):
            named = [(name, p) for name, p in self.hybrid_student.named_parameters() if p.requires_grad and
                     ('.sequence_pool.' in name or '.response_projection.' in name)]
            gradients = torch.autograd.grad(prototype, [p for _, p in named], retain_graph=True, allow_unused=True)
            norms = {'signature': 0., 'reference': 0.}
            for (name, _), gradient in zip(named, gradients):
                if gradient is not None:
                    group = 'signature' if '.sequence_pool.' in name else 'reference'
                    norms[group] += float(gradient.detach().double().square().sum())
            self.prototype_smoke_gradients = {'branch': {key: value ** .5 for key, value in norms.items()}}
        result = result + prototype
        # If a batch has no different accepted pseudo-class, keep the same-packet
        # relation defined without inventing negative identities.
        if bool(selected.any()) and pseudo[accepted].unique().numel() == 1:
            def cosine(a, b):
                return ((1 - F.cosine_similarity(a.detach(), b, dim=1, eps=1e-8)) * selected).mean()
            fusion = torch.stack([cosine(views[i], views[i + 2]) for i in range(2)]).mean()
            branch = torch.stack([cosine(self.hybrid_branches[i][key], self.hybrid_branches[i + 2][key])
                                  for i in range(2) for key in self.hybrid_branches[0]]).mean()
            result = result + fusion + branch
            self.fallback_calls += 1
        return result

    def state_dict(self):
        state = super().state_dict()
        if self.as_offered != self.accepted or self.as_used != self.accepted:
            raise ValueError('Every confidence-accepted U sample must reach auxiliary routing')
        state['U_confidence_only'] = dict(threshold=.95, offered=self.as_offered, used=self.as_used,
            accepted_by_class=dict(self.accepted_classes), in_original_physical_half=self.as_physical_used,
            source_labeled_diagnostic=dict(self.labeled_diagnostic), single_class_fallback_calls=self.fallback_calls,
            prototype_fallbacks=dict(self.prototype_fallbacks), prototype_used=self.prototype_used,
            branch_votes_used=False, U_truth_read=False, target_access=False)
        return state
