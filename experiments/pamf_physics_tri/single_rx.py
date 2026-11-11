"""Source-day association for the single-source-receiver Tri continuation."""
from contextlib import contextmanager

import torch

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import branch_relation_loss, capture_branches
from .branch_guard import BranchGuard
from .practical_gate import installed as practical_scope


class SingleRxDay(BranchGuard):
    def __init__(self, c):
        super().__init__(c)
        self.day_relations = None
        self.day_calls = self.day_anchors = 0

    def labeled_group(self, metadata, labels):
        days = metadata['day_i'].to(labels.device).long()
        if set(days.tolist()) - set(self.c['source_days']):
            raise ValueError('L anchor days must belong to the source contract')
        return days

    def query_group(self, metadata, indices):
        return metadata['day_i'][indices].long()

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        days = metadata['day_i'].to(x.device).long()
        actual_rx = metadata['rx_i'].to(x.device).long()
        if set(actual_rx.tolist()) != set(self.c['source_rxs']) or set(days.tolist()) - set(self.c['source_days']):
            raise ValueError('Only the declared single source receiver and source days are allowed')
        # The existing feature-pairing implementation groups by metadata rx_i.
        # Substitute day only inside its six U feature passes; signal simulation
        # and the upstream source-data validation continue to see the real RX.
        grouped = dict(metadata, rx_i=days)
        original = self.c
        self.c = dict(original, source_rxs=original['source_days'])
        try:
            if self._rxe_forward_index < 2:
                core, sequence, reference = _actual_paths(student)
                with capture_branches(core, sequence, reference) as features:
                    output = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, grouped)
                self.day_relations.append((features, pseudo, accepted, days))
                return output
            return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, grouped)
        finally:
            self.c = original

    def loss(self, student, device):
        self.day_relations = []
        try:
            result = super().loss(student, device)
            if len(self.day_relations) != 2 or self.pending_gradient_loss is None:
                raise ValueError('Two existing U branch captures and the guarded loss are required')
            losses, anchors = [], 0
            for features, pseudo, accepted, days in self.day_relations:
                ids = torch.arange(len(days), device=days.device)
                value, stats = branch_relation_loss(features, pseudo, days, ids, accepted, single_receiver=False)
                losses.append(value)
                anchors += stats['cross_receiver_anchors']
            relation = torch.stack(losses).mean()
            self.pending_gradient_loss = self.pending_gradient_loss + self.c['day_relation_weight'] * relation
            self.day_calls += 1
            self.day_anchors += anchors
            self.metrics.update(HPL_day_branch_relation=relation.detach(), HPL_day_anchors=self.day_anchors)
            return result
        finally:
            self.day_relations = None

    def state_dict(self):
        state = super().state_dict()
        if self.day_relations is not None or self.day_calls != self.calls:
            raise ValueError('One cross-day branch relation per U loss required')
        state['U_cross_day'] = dict(calls=self.day_calls, branch_anchors=self.day_anchors,
                                    relation_weight=self.c['day_relation_weight'],
                                    actual_source_receivers=self.c['source_rxs'], grouping='source_day')
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .single_rx_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=SingleRxDay):
        yield
