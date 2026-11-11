"""Gate source-day U relations by L-anchored branch and channel evidence."""
from contextlib import contextmanager

import torch

from experiments.pamf_physics_branch.objective import branch_views
from .recovery import branch_certificate
from .single_rx import SingleRxDay
from .practical_gate import installed as practical_scope


class CertifiedDay(SingleRxDay):
    def __init__(self, c):
        super().__init__(c)
        self.relation_mask = None
        self.certificate_calls = self.certificate_offered = self.branch_pass = self.echo_pass = self.relation_used = 0

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        if self.labeled is None or self.relation_mask is not None:
            raise ValueError('One current source L anchor batch per U loss required')
        l_x, labels, days = self.labeled
        raw, echo = branch_views(student, torch.cat((l_x, x)), self.c['model_seed'] * 100_000 + step)
        branch, stable = branch_certificate(raw, echo, labels, days, metadata['day_i'], pseudo,
                                            self.c['num_classes'], 4)
        self.relation_mask = accepted & (stable if self.c['relation_certificate'] == 'echo' else branch)
        self.certificate_calls += 1
        self.certificate_offered += int(accepted.sum())
        self.branch_pass += int((accepted & branch).sum())
        self.echo_pass += int((accepted & stable).sum())
        self.relation_used += int(self.relation_mask.sum())
        return mixed

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self.relation_mask is None or bool((self.relation_mask & ~accepted).any()):
            raise ValueError('Relation mask must be a subset of accepted source U')
        # HardPLObjective retains its original accepted mask for each U CE/KL.
        # Only the Tri feature pairing and new branch relation see this subset.
        return super().unlabeled_student_forward(student, x, domain, pseudo, self.relation_mask, metadata)

    def loss(self, student, device):
        try:
            return super().loss(student, device)
        finally:
            self.relation_mask = None

    def state_dict(self):
        state = super().state_dict()
        expected = self.echo_pass if self.c['relation_certificate'] == 'echo' else self.branch_pass
        if (self.relation_mask is not None or self.certificate_calls != self.calls
                or not 0 <= self.echo_pass <= self.branch_pass <= self.certificate_offered <= self.seen
                or self.relation_used != expected):
            raise ValueError('Source-only relation certificate counts differ')
        state['U_relation_certificate'] = dict(kind=self.c['relation_certificate'], calls=self.certificate_calls,
            offered=self.certificate_offered, branch_pass=self.branch_pass, echo_pass=self.echo_pass,
            relation_used=self.relation_used, additional_eval_forwards=2 * self.certificate_calls,
            raw_pseudo_CE_mask_unchanged=True)
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .certified_day_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=CertifiedDay):
        yield
