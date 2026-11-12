"""Hybrid auxiliary training for every confidence-accepted source-U packet."""
from collections import Counter

import torch
from torch.nn import functional as F

from .hybrid_contrast import HybridContrast
from .practical_gate import view_strength


class HybridConfidence(HybridContrast):
    def __init__(self, c):
        super().__init__(c)
        self.as_physical_used = 0
        self.accepted_classes = Counter()
        self.labeled_diagnostic = Counter()
        self.fallback_calls = 0

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        raw_l, labels, _ = self.labeled
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
        result = super().additional_fusion_loss(views, selected)
        pseudo, accepted = self.hybrid_labels
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
            raise ValueError('Every confidence-accepted U sample must reach Hybrid auxiliary routing')
        state['U_confidence_only'] = dict(threshold=.95, offered=self.as_offered, used=self.as_used,
            accepted_by_class=dict(self.accepted_classes), in_original_physical_half=self.as_physical_used,
            source_labeled_diagnostic=dict(self.labeled_diagnostic), single_class_fallback_calls=self.fallback_calls,
            branch_votes_used=False, prototype_used=False, U_truth_read=False, target_access=False)
        return state
