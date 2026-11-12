"""Source-only cross-day reference memory and weighted certificate tiers."""
from collections import Counter, deque

import torch

from .packet_contrast import packet_contrast_loss
from .practical_gate import capture_pair, view_strength
from .recovery import branch_scores
from .source_prototype import SourcePrototype, capture_source_bank, prototype_loss


def certificate_tiers(raw, changed, have_all, pseudo):
    raw_votes = (raw.argmax(2) == pseudo[:, None]).sum(1)
    changed_votes = (changed.argmax(2) == pseudo[:, None]).sum(1)
    strong = have_all & (raw_votes >= 4) & (changed_votes >= 4)
    weak = have_all & (raw_votes >= 3) & (changed_votes >= 3) & ~strong
    return strong, weak, raw_votes, changed_votes


class RollingSourceMemory:
    """Bounded raw L samples per class/day; cached features refresh every 20 steps."""
    def __init__(self, capacity=8, interval=20):
        self.capacity, self.interval = capacity, interval
        self.groups, self.cached = {}, None

    def add(self, x, labels, days):
        for sample, label, day in zip(x.detach(), labels.tolist(), days.tolist()):
            self.groups.setdefault((label, day), deque(maxlen=self.capacity)).append(sample.detach())

    def features(self, student, step):
        if self.cached is None or step % self.interval == 0:
            items = [(key, sample) for key, samples in sorted(self.groups.items()) for sample in samples]
            raw = torch.stack([sample for _, sample in items])
            labels = torch.tensor([key[0] for key, _ in items], device=raw.device, dtype=torch.long)
            days = torch.tensor([key[1] for key, _ in items], device=raw.device, dtype=torch.long)
            fusion, branches = capture_source_bank(student, raw)
            self.cached = fusion, branches, labels, days
        return self.cached


class CoveragePrototype(SourcePrototype):
    def __init__(self, c):
        super().__init__(c)
        self.labeled_metadata = None
        self.memory = RollingSourceMemory() if c['coverage']['reference_memory'] else None
        self.weak_mask = self.diagnostic_aux = None
        self.coverage_counts, self.labeled_diagnostic = Counter(), Counter()
        self.strong_classes, self.weak_classes = Counter(), Counter()
        self.loss_sums = Counter()
        self.gradient_diagnostics = Counter()
        self.last_step = 0

    def _source_bank(self, student, raw, labels, days, step):
        if self.memory is None:
            fusion, branches = capture_source_bank(student, raw)
            return fusion, branches, labels.detach(), days.detach()
        self.memory.add(raw, labels, days)
        return self.memory.features(student, step)

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        if self.labeled is None:
            raise ValueError('Source labeled batch required')
        self.last_step = step
        raw_l, labels, days = self.labeled
        fusion, bank, bank_labels, bank_days = self._source_bank(student, raw_l, labels, days, step)
        self.prototype_bank = fusion, bank, bank_labels, bank_days
        self.prototype_days = metadata['day_i'].to(x.device).long().detach()
        indices = physical.nonzero(as_tuple=True)[0]
        selected_meta = {k: v[indices] if torch.is_tensor(v) and v.ndim and len(v) == len(x) else v
                         for k, v in metadata.items()}
        simulated = student._cd_practical(x[indices], selected_meta, epoch, step)
        raw_query, changed_query = capture_pair(student, x[indices], simulated)
        raw = {key: torch.cat((bank[key], raw_query[key])) for key in bank}
        changed = {key: torch.cat((bank[key], changed_query[key])) for key in bank}
        query_days = metadata['day_i'][indices].long()
        scores, changed_scores, available = branch_scores(raw, changed, bank_labels, bank_days,
                                                           query_days, self.c['num_classes'])
        strong, weak, _, _ = certificate_tiers(scores, changed_scores, available, pseudo[indices])
        offered = accepted[indices]
        strong &= offered
        weak &= offered
        if not self.c['coverage']['tiered']:
            weak &= False
        strength = view_strength(self.c['practical_gate'], epoch)
        if strength == 0.:
            strong &= False
            weak &= False
        selected = strong | weak
        self.record_practical_selection(indices[selected], len(x))
        self.weak_mask = torch.zeros_like(accepted)
        self.weak_mask[indices[weak]] = True
        mixed = ordinary.clone()
        mixed[indices] = x[indices]
        mixed[indices[selected]] = x[indices[selected]] + strength * (simulated[selected] - x[indices[selected]])
        self.as_calls += 1
        self.as_offered += int(offered.sum())
        self.as_certified += int(selected.sum())
        self.as_active_certified += int(selected.sum())
        self.as_used += int(selected.sum())
        self.strength_weighted_uses += strength * int(selected.sum())
        self.coverage_counts.update(offered=int(offered.sum()), reference_missing=int((offered & ~available).sum()),
                                    strong=int(strong.sum()), weak=int(weak.sum()),
                                    other_rejected=int((offered & available & ~selected).sum()))
        self.strong_classes.update(pseudo[indices[strong]].detach().cpu().tolist())
        self.weak_classes.update(pseudo[indices[weak]].detach().cpu().tolist())
        if step % 50 == 0:
            self._diagnose_labeled(student, raw_l, labels, days, bank, bank_labels, bank_days, epoch, step)
        return mixed

    def _diagnose_labeled(self, student, raw_l, labels, days, bank, bank_labels, bank_days, epoch, step):
        from experiments.cric_dura_comparison.t1_practical import Practical
        raw_l, labels, days = raw_l[:32], labels[:32], days[:32]
        meta = {key: value[:32] if torch.is_tensor(value) and value.ndim else value
                for key, value in self.labeled_metadata.items()}
        altered = Practical(self.c)(raw_l, meta, epoch, step)
        modes = [(m, m.training) for m in student.modules()]
        try:
            student.eval()
            with torch.no_grad():
                logits = student(raw_l, return_aux=True)['tx_logits']
                probability, pseudo = logits.softmax(1).max(1)
        finally:
            for module, mode in modes:
                module.training = mode
        raw_query, altered_query = capture_pair(student, raw_l, altered)
        raw = {key: torch.cat((bank[key], raw_query[key])) for key in bank}
        changed = {key: torch.cat((bank[key], altered_query[key])) for key in bank}
        first, second, have = branch_scores(raw, changed, bank_labels, bank_days, days, self.c['num_classes'])
        strong, weak, _, _ = certificate_tiers(first, second, have, pseudo)
        offered = probability >= .95
        for name, mask in (('strong', offered & strong), ('weak', offered & weak)):
            self.labeled_diagnostic[name + '_count'] += int(mask.sum())
            self.labeled_diagnostic[name + '_correct'] += int((mask & (pseudo == labels)).sum())
        self.labeled_diagnostic.update(offered=int(offered.sum()), missing=int((offered & ~have).sum()))

    def additional_fusion_loss(self, views, selected):
        strong = selected & ~self.weak_mask
        original = super().additional_fusion_loss(views, strong)
        weak = selected & self.weak_mask
        pseudo, accepted = self.hybrid_labels
        _, branches, labels, days = self.prototype_bank
        temperature = self.c['hybrid_contrast']['temperature']
        fusion = [packet_contrast_loss(views[i], views[i + 2], pseudo, accepted, weak, temperature)[0]
                  for i in range(2)]
        branch = [packet_contrast_loss(self.hybrid_branches[i][key], self.hybrid_branches[i + 2][key],
                                       pseudo, accepted, weak, temperature)[0]
                  for i in range(2) for key in branches]
        proto = [prototype_loss(self.hybrid_branches[i][key], branches[key], labels, days,
                                self.prototype_days, pseudo, weak, self.c['num_classes'],
                                self.c['source_prototype']['temperature'])[0]
                 for i in (2, 3) for key in branches]
        added = self.c['coverage']['weak_weight'] * (torch.stack(fusion).mean() +
                torch.stack(branch).mean() + torch.stack(proto).mean())
        self.loss_sums.update(strong=float(original.detach()), weak=float(added.detach()))
        self.diagnostic_aux = original + added
        self.metrics.update(HPL_coverage_strong=int(strong.sum()), HPL_coverage_weak=int(weak.sum()),
                            HPL_coverage_strong_loss=original.detach(), HPL_coverage_weak_loss=added.detach())
        return self.diagnostic_aux

    def guide(self, labeled, student):
        if self.last_step % 50 == 0 and self.diagnostic_aux is not None:
            params = [p for name, p in student.named_parameters() if p.requires_grad and
                      ('.sequence_pool.' in name or '.response_projection.' in name)]
            left = torch.autograd.grad(labeled, params, retain_graph=True, allow_unused=True)
            right = torch.autograd.grad(self.diagnostic_aux, params, retain_graph=True, allow_unused=True)
            dot = sum(float((a.detach() * b.detach()).sum()) for a, b in zip(left, right)
                      if a is not None and b is not None)
            self.gradient_diagnostics.update(samples=1, conflicts=int(dot < 0))
        self.diagnostic_aux = None
        return super().guide(labeled, student)

    def loss(self, student, device):
        try:
            return super().loss(student, device)
        finally:
            self.weak_mask = None

    def state_dict(self):
        state = super().state_dict()
        state['U_coverage'] = dict(recipe=self.c['coverage'], counts=dict(self.coverage_counts),
            strong_classes=dict(self.strong_classes), weak_classes=dict(self.weak_classes),
            auxiliary_loss_sums=dict(self.loss_sums), gradient_diagnostic=dict(self.gradient_diagnostics),
            source_labeled_diagnostic=dict(self.labeled_diagnostic),
            memory_sizes={f'{label}:{day}': len(samples) for (label, day), samples in self.memory.groups.items()}
            if self.memory else {}, U_truth_read=False, target_access=False)
        return state
