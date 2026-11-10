"""Add a guarded physical relation solely on accepted hidden-label U."""
from contextlib import contextmanager
import torch

from experiments.pamf_branch_training.source_ids import indexed_source_scope
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_physics_branch.objective import physical_directions, select_u_candidates, source_relation
from experiments.pamf_repl_view_shuffle.source import installed as original_tri
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded


def selected_relation(packet, directions, pseudo, metadata, selected, accepted, c):
    return source_relation(packet, directions, pseudo, metadata, selected, c, accepted=accepted)


class PhysicalU(ReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self._physical_u_packet = None
        self.physical_u_calls = self.physical_u_selected = self.physical_u_accepted = self.physical_u_anchors = 0

    def validate_unlabeled_batch(self, batch):
        super().validate_unlabeled_batch(batch)
        visible = batch[3]['tx_label_visible']
        if visible.dtype != torch.bool or visible.shape != batch[1].shape or bool(visible.any()):
            raise ValueError('Only hidden-label source U may use the physical relation')

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self._rxe_forward_index != 0:
            return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        if self._physical_u_packet is not None:
            raise ValueError('Only one physical U packet per objective call')
        core, sequence, reference = _actual_paths(student)
        with capture_branches(core, sequence, reference) as packet:
            logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        self._physical_u_packet = (packet, x, pseudo, accepted, metadata)
        return logits

    def loss(self, student, device):
        if self._physical_u_packet is not None:
            raise ValueError('Unconsumed physical U packet')
        try:
            value = super().loss(student, device)
            packet, x, pseudo, accepted, metadata = self._physical_u_packet
            step = student._feature_l_steps + self.c.get('step_offset', 0)
            selected = select_u_candidates(accepted, 128, 202611112 + 2 * step)
            directions, _ = physical_directions(student, x.index_select(0, selected), 202611114 + 2 * step)
            relation, stats = selected_relation(packet, directions, pseudo, metadata, selected, accepted, self.c)
            self.physical_u_calls += 1
            self.physical_u_selected += len(selected)
            self.physical_u_accepted += int(accepted.index_select(0, selected).sum())
            self.physical_u_anchors += stats['valid_anchors']
            self.metrics.update(HPL_physical_U_loss=relation.detach(), HPL_physical_U_calls=self.physical_u_calls,
                                HPL_physical_U_selected=self.physical_u_selected,
                                HPL_physical_U_accepted=self.physical_u_accepted,
                                HPL_physical_U_anchors=self.physical_u_anchors)
            return value + self.c['physical_relation']['u_weight'] * relation
        finally:
            self._physical_u_packet = None

    def state_dict(self):
        state = super().state_dict()
        if self._physical_u_packet is not None or self.physical_u_calls != self.calls:
            raise ValueError('One physical U relation per original REPL call required')
        state['U_physical_relation'] = dict(calls=self.physical_u_calls, selected=self.physical_u_selected,
            accepted=self.physical_u_accepted, valid_anchors=self.physical_u_anchors,
            extra_eval_forwards=2*self.physical_u_calls, U_truth_read=False)
        return state


class GuardedPhysicalU(Guarded, PhysicalU):
    pass


@contextmanager
def installed(c, holder):
    from experiments.pamf_repl_completion import objectives
    previous = objectives.objective
    objectives.objective = GuardedPhysicalU
    try:
        with indexed_source_scope(), original_tri(c, holder, enabled=True):
            yield
    finally:
        objectives.objective = previous
