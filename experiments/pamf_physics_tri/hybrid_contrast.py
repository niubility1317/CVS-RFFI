"""Same-packet physical contrast in fusion, reverse, and branch spaces."""
from contextlib import contextmanager

import torch

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from .fused_physics import FusedPhysics
from .packet_contrast import packet_contrast_loss
from .practical_gate import installed as practical_scope


class HybridContrast(FusedPhysics):
    def __init__(self, c):
        super().__init__(c)
        self.hybrid_labels = self.hybrid_branches = self.hybrid_student = None
        self.hybrid_calls = self.hybrid_anchors = 0

    def include_fused_cosine(self):
        return False

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        index = self._rxe_forward_index
        if index == 0:
            if self.hybrid_labels is not None:
                raise ValueError('Only one current U pseudo-label set allowed')
            self.hybrid_labels = pseudo.detach(), accepted.detach()
        if index < 4 and self.c['hybrid_contrast']['mode'] in ('branch', 'fusion_branch'):
            core, sequence, reference = _actual_paths(student)
            with capture_branches(core, sequence, reference) as paths:
                logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
            self.hybrid_branches.append(paths)
            return logits
        return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)

    def additional_fusion_loss(self, views, selected):
        if self.hybrid_labels is None:
            raise ValueError('Current accepted U pseudo identities required')
        pseudo, accepted = self.hybrid_labels
        mode = self.c['hybrid_contrast']['mode']
        def pair(first, second):
            return packet_contrast_loss(first, second, pseudo, accepted, selected,
                                        temperature=self.c['hybrid_contrast']['temperature'])
        components = {}
        counts = []
        if mode in ('symmetric', 'fusion_branch'):
            forward = [pair(views[i], views[i + 2]) for i in range(2)]
            components['fusion'] = torch.stack([x[0] for x in forward]).mean()
            counts.extend(x[1] for x in forward)
        if mode == 'symmetric':
            backward = [pair(views[i + 2], views[i]) for i in range(2)]
            components['reverse'] = torch.stack([x[0] for x in backward]).mean()
            counts.extend(x[1] for x in backward)
        if mode in ('branch', 'fusion_branch'):
            if len(self.hybrid_branches) != 4:
                raise ValueError('Two ordinary and two physical branch captures required')
            keys = self.hybrid_branches[0].keys()
            if not keys or any(v.keys() != keys for v in self.hybrid_branches):
                raise ValueError('Aligned deployed branch paths required')
            branch = [pair(self.hybrid_branches[i][key], self.hybrid_branches[i + 2][key])
                      for i in range(2) for key in keys]
            components['branch'] = torch.stack([x[0] for x in branch]).mean()
            counts.extend(x[1] for x in branch)
        if not components or len(set(counts)) != 1:
            raise ValueError('Same physical slots must define every active contrast')
        result = .5 * (components['fusion'] + components['reverse']) if mode == 'symmetric' else sum(components.values())
        if self.c.get('hard_pl_smoke'):
            named = [(name, p) for name, p in self.hybrid_student.named_parameters() if p.requires_grad and
                     ('.sequence_pool.' in name or '.response_projection.' in name)]
            self.hybrid_smoke_gradients = {}
            for kind, loss in components.items():
                grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
                norms = {'signature': 0., 'reference': 0.}
                for (name, _), gradient in zip(named, grads):
                    if gradient is not None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise ValueError('Finite physical contrast gradients required')
                        group = 'signature' if '.sequence_pool.' in name else 'reference'
                        norms[group] += float(gradient.detach().double().square().sum())
                self.hybrid_smoke_gradients[kind] = {key: value ** .5 for key, value in norms.items()}
        self.hybrid_calls += 1
        self.hybrid_anchors += 2 * counts[0]
        self.metrics.update({f'HPL_hybrid_{kind}_contrast': loss.detach() for kind, loss in components.items()})
        return result

    def loss(self, student, device):
        self.hybrid_branches = []
        self.hybrid_student = student if self.c.get('hard_pl_smoke') else None
        try:
            return super().loss(student, device)
        finally:
            self.hybrid_labels = self.hybrid_branches = self.hybrid_student = None

    def state_dict(self):
        state = super().state_dict()
        if (self.hybrid_labels is not None or self.hybrid_branches is not None or self.hybrid_student is not None
                or self.hybrid_calls != self.calls or not 0 <= self.hybrid_anchors <= 2 * self.fusion_selected):
            raise ValueError('One source-only physical contrast per U loss required')
        state['U_hybrid_contrast'] = dict(recipe=self.c['hybrid_contrast'], calls=self.hybrid_calls,
            anchors=self.hybrid_anchors, teacher_detached=True, extra_student_forwards=0)
        if self.c.get('hard_pl_smoke'):
            state['U_hybrid_contrast']['smoke_gradients'] = self.hybrid_smoke_gradients
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .hybrid_contrast_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=HybridContrast):
        yield
