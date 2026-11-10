"""Train-only branch objectives for the original full PAMF graph."""
from contextlib import contextmanager
from types import MethodType

import torch

from experiments.pamf_ssl_research.hard_pl_objective import HardPLObjective
from .relation import branch_relation_loss, capture_branches


def _actual_paths(student):
    reference = student.id_backbone.encoder
    projection = getattr(reference, 'response_projection', None)
    if projection is None:
        raise ValueError('Original full PAMF reference-response branch is required')
    return reference.core.id_backbone, reference.sequence_pool, projection


def _source_meta(metadata, device, allowed_receivers):
    receivers = metadata['rx_i'].to(device).long()
    ids = metadata['branch_record_id'].to(device).long()
    if set(receivers.tolist()) - set(allowed_receivers):
        raise ValueError('Only declared source receivers may form branch relations')
    return receivers, ids


def install_labeled(student, config):
    """Augment the original labeled objective with branch relations."""
    original_context = student._phase1_labeled_feature_context
    original_hook = student._source_labeled_feature_hook
    student._branch_pending = None
    student._branch_l_calls = 0

    @contextmanager
    def labeled_context(self):
        if self._branch_pending is not None:
            raise RuntimeError('Previous labeled branch graph was not consumed')
        core, sequence, reference = _actual_paths(self)
        with original_context():
            with capture_branches(core, sequence, reference) as packet:
                yield
        self._branch_pending = packet

    def hook(self, **kwargs):
        packet, self._branch_pending = self._branch_pending, None
        if packet is None:
            raise RuntimeError('Labeled branch capture missing')
        y = kwargs['labels']
        rx, ids = _source_meta(kwargs['extra'][1], y.device, config['source_rxs'])
        branch, counts = branch_relation_loss(packet, y, rx, ids, torch.ones_like(y, dtype=torch.bool),
                                              single_receiver=len(config['source_rxs']) == 1)
        value, metrics = original_hook(**kwargs)
        self._branch_l_calls += 1
        metrics.update(branch_L_loss=branch.detach(), branch_L_calls=self._branch_l_calls,
                       branch_L_cross_anchors=counts['cross_receiver_anchors'],
                       branch_L_single_rx_anchors=counts['same_receiver_anchors'])
        return value + config['branch_relation']['l_weight'] * branch, metrics

    hook._accepts_labeled_inputs = True
    student._phase1_labeled_feature_context = MethodType(labeled_context, student)
    student._source_labeled_feature_hook = MethodType(hook, student)
    return student


class BranchPL(HardPLObjective):
    """Ordinary PL plus a per-branch relation on accepted source U examples."""
    def __init__(self, config):
        super().__init__(config)
        if self.settings['method'] != 'HPL2':
            raise ValueError('Branch PL requires the matched ordinary one-view PL base')
        self.branch_u_calls = 0
        self._branch_u = None

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self._branch_u is not None:
            raise RuntimeError('Exactly one ordinary PL student forward required')
        core, sequence, reference = _actual_paths(student)
        with capture_branches(core, sequence, reference) as packet:
            logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        self._branch_u = (packet, pseudo, accepted, metadata)
        return logits

    def loss(self, student, device):
        if self._branch_u is not None:
            raise RuntimeError('Previous U branch graph was not consumed')
        try:
            value = super().loss(student, device)
            if self._branch_u is None:
                raise RuntimeError('Ordinary PL did not execute its branch capture')
            packet, pseudo, accepted, metadata = self._branch_u
            rx, ids = _source_meta(metadata, pseudo.device, self.c['source_rxs'])
            branch, counts = branch_relation_loss(packet, pseudo, rx, ids, accepted,
                                                   single_receiver=len(self.c['source_rxs']) == 1)
            value = value + self.c['branch_relation']['u_weight'] * branch
            self.branch_u_calls += 1
            self.metrics.update(branch_U_loss=branch.detach(), branch_U_calls=self.branch_u_calls,
                                branch_U_cross_anchors=counts['cross_receiver_anchors'],
                                branch_U_single_rx_anchors=counts['same_receiver_anchors'])
            return value
        finally:
            self._branch_u = None

    def state_dict(self):
        state = super().state_dict()
        if self._branch_u is not None or self.branch_u_calls != self.calls:
            raise RuntimeError('U branch objective did not match ordinary PL calls')
        state['branch_U'] = dict(calls=self.branch_u_calls, extra_backbone_forwards=0,
                                 target_access=False, U_truth_read=False)
        return state
