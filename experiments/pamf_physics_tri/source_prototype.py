"""Source-label, cross-day prototypes for physically routed U features."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint
from .hybrid_contrast import HybridContrast
from .practical_gate import installed as practical_scope


def prototype_loss(query, bank, labels, bank_days, query_days, pseudo, selected, classes, temperature=.2):
    """Contrast U against true-L class centroids excluding U's acquisition day."""
    if (query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]
            or labels.shape != bank.shape[:1] or bank_days.shape != labels.shape
            or query_days.shape != query.shape[:1] or pseudo.shape != query_days.shape
            or selected.shape != query_days.shape or selected.dtype != torch.bool
            or temperature <= 0 or classes < 2):
        raise ValueError('Aligned source-L bank and source-U query required')
    if bool(((labels < 0) | (labels >= classes)).any()) or bool(((pseudo < 0) | (pseudo >= classes)).any()):
        raise ValueError('Source and pseudo identities must be in range')
    memberships = (F.one_hot(labels.long(), classes).T.bool().unsqueeze(0)
                   & (query_days[:, None, None] != bank_days[None, None, :]))
    valid = memberships.any(2).all(1) & selected
    normalized_bank = F.normalize(bank.detach().float(), dim=1)
    centers = F.normalize(memberships.float() @ normalized_bank, dim=2)
    scores = torch.einsum('ncd,nd->nc', centers, F.normalize(query.float(), dim=1)) / temperature
    loss = (F.cross_entropy(scores, pseudo.long(), reduction='none') * valid).mean()
    return loss, int(valid.sum())


def capture_source_bank(student, raw):
    modes = [(module, module.training) for module in student.modules()]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(raw.device) if raw.is_cuda else None
    try:
        student.eval()
        core, sequence, reference = _actual_paths(student)
        with torch.no_grad(), capture_branches(core, sequence, reference) as paths:
            with capture_joint(student.id_backbone.encoder) as joint:
                student(raw)
    finally:
        for module, mode in modes:
            module.training = mode
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, raw.device)
    return joint[0].detach(), {key: value.detach() for key, value in paths.items()}


class SourcePrototype(HybridContrast):
    def __init__(self, c):
        super().__init__(c)
        self.prototype_bank = self.prototype_days = None
        self.prototype_calls = self.prototype_used = 0

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        if self.prototype_bank is not None or self.labeled is None:
            raise ValueError('Exactly one current labeled source bank is required')
        raw, labels, days = self.labeled
        fusion, branches = capture_source_bank(student, raw)
        self.prototype_bank = fusion, branches, labels.detach(), days.detach()
        self.prototype_days = metadata['day_i'].to(x.device).long().detach()
        return mixed

    def additional_fusion_loss(self, views, selected):
        original = super().additional_fusion_loss(views, selected)
        if self.prototype_bank is None or self.hybrid_labels is None or len(self.hybrid_branches) != 4:
            raise ValueError('Current source prototypes and four U branch views required')
        fusion, branches, labels, days = self.prototype_bank
        pseudo, accepted = self.hybrid_labels
        mask = selected & accepted
        mode = self.c['source_prototype']['mode']
        components, counts = {}, []
        def pair(query, bank):
            return prototype_loss(query, bank, labels, days, self.prototype_days, pseudo, mask,
                                  self.c['num_classes'], self.c['source_prototype']['temperature'])
        if mode in ('fusion', 'fusion_branch'):
            losses = [pair(views[index], fusion) for index in (2, 3)]
            components['fusion'] = torch.stack([item[0] for item in losses]).mean()
            counts.extend(item[1] for item in losses)
        if mode in ('branch', 'fusion_branch'):
            if any(value.keys() != branches.keys() for value in self.hybrid_branches):
                raise ValueError('Source and U deployed branches must match')
            losses = [pair(self.hybrid_branches[index][key], branches[key])
                      for index in (2, 3) for key in branches]
            components['branch'] = torch.stack([item[0] for item in losses]).mean()
            counts.extend(item[1] for item in losses)
        if not components or len(set(counts)) != 1:
            raise ValueError('Prototype components must use the same physical U slots')
        if self.c.get('hard_pl_smoke'):
            named = [(name, p) for name, p in self.hybrid_student.named_parameters() if p.requires_grad and
                     ('.sequence_pool.' in name or '.response_projection.' in name)]
            self.prototype_smoke_gradients = {}
            for kind, loss in components.items():
                gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
                norms = {'signature': 0., 'reference': 0.}
                for (name, _), gradient in zip(named, gradients):
                    if gradient is not None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise ValueError('Prototype gradients must be finite')
                        group = 'signature' if '.sequence_pool.' in name else 'reference'
                        norms[group] += float(gradient.detach().double().square().sum())
                self.prototype_smoke_gradients[kind] = {key: value ** .5 for key, value in norms.items()}
        self.prototype_calls += 1
        self.prototype_used += 2 * counts[0]
        self.metrics.update({f'HPL_source_prototype_{key}': value.detach() for key, value in components.items()})
        return original + sum(components.values())

    def loss(self, student, device):
        try:
            return super().loss(student, device)
        finally:
            self.prototype_bank = self.prototype_days = None

    def state_dict(self):
        state = super().state_dict()
        if (self.prototype_bank is not None or self.prototype_days is not None or self.prototype_calls != self.calls
                or not 0 <= self.prototype_used <= 2 * self.fusion_selected):
            raise ValueError('One source-only prototype objective per U loss required')
        state['U_source_prototype'] = dict(recipe=self.c['source_prototype'], calls=self.prototype_calls,
            used=self.prototype_used, extra_labeled_eval_forwards=self.prototype_calls,
            extra_student_train_forwards=0, bank_detached=True, cross_day=True,
            U_truth_read=False, target_access=False)
        if self.c.get('hard_pl_smoke'):
            state['U_source_prototype']['smoke_gradients'] = self.prototype_smoke_gradients
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .source_prototype_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=SourcePrototype):
        yield
