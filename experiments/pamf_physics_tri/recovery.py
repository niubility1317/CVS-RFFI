"""Source-label cross-receiver branch evidence for rejected Tri-SIG-TIME U."""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F

from experiments.pamf_physics_branch.objective import branch_views
from experiments.pamf_repl_view_shuffle.source import installed as original_tri
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation
from .pairing import BRANCHES


def branch_certificate(clean, echo, labels, labeled_rx, query_rx, pseudo, classes, required_votes):
    """Vote for the original fused pseudo-class using only other-RX L centroids."""
    count = len(labels)
    known = F.one_hot(labels, classes).T.bool()[None, :, :]
    members = known & (query_rx[:, None, None] != labeled_rx[None, None, :])
    have_all_classes = members.any(2).all(1)
    raw_votes = torch.zeros_like(pseudo)
    echo_votes = torch.zeros_like(pseudo)
    for name in BRANCHES:
        features = F.normalize(clean[name].float(), dim=1, eps=1e-8)
        centroids = F.normalize(members.to(features.dtype) @ features[:count], dim=2, eps=1e-8)
        raw_scores = (centroids @ features[count:, :, None]).squeeze(2)
        echo_features = F.normalize(echo[name][count:].float(), dim=1, eps=1e-8)
        echo_scores = (centroids @ echo_features[:, :, None]).squeeze(2)
        raw_votes += raw_scores.argmax(1) == pseudo
        echo_votes += echo_scores.argmax(1) == pseudo
    branch = have_all_classes & (raw_votes >= required_votes)
    return branch, branch & (echo_votes >= required_votes)


class RejectedURecovery(ReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self.recipe = c['rejected_u_recovery']
        self.labeled = None
        self.candidates = self.recovered = None
        self.recovery_calls = self.probe_steps = self.probed = self.branch_pass = self.echo_pass = self.used = 0

    def validate_unlabeled_batch(self, batch):
        super().validate_unlabeled_batch(batch)
        visible = batch[3]['tx_label_visible']
        if visible.dtype != torch.bool or visible.shape != batch[1].shape or bool(visible.any()):
            raise ValueError('Only hidden-label source U may receive branch certificates')

    def accept_pseudo(self, student, x, confidence, pseudo):
        accepted = super().accept_pseudo(student, x, confidence, pseudo)
        self.candidates = ~accepted & (confidence >= self.recipe['minimum_confidence'])
        return accepted

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        if self.labeled is None or self.candidates is None or self.recovered is not None:
            raise ValueError('One current true L packet and one rejected U selection required')
        raw_l, labels, l_rx = self.labeled
        indices = self.candidates.nonzero(as_tuple=True)[0]
        eligible = torch.zeros_like(accepted)
        if indices.numel():
            seed = self.c['model_seed'] * 100_000 + step
            clean, echo = branch_views(student, torch.cat((raw_l, x[indices])), seed)
            branch, stable = branch_certificate(clean, echo, labels, l_rx, metadata['rx_i'][indices],
                                                pseudo[indices], self.c['num_classes'], self.recipe['required_votes'])
            eligible[indices] = stable if self.recipe['require_echo'] else branch
            self.probe_steps += 1
            self.probed += len(indices)
            self.branch_pass += int(branch.sum())
            self.echo_pass += int(stable.sum())
        if bool((eligible & accepted).any()):
            raise ValueError('Recovery may use only originally rejected packets')
        self.recovered = pseudo, eligible
        self.used += int(eligible.sum())
        return mixed

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        if self._rxe_forward_index in (5, 6):
            self.raw_logits.append(logits)
        return logits

    def loss(self, student, device):
        self.raw_logits = []
        try:
            value = super().loss(student, device)
            if self.recovered is None or len(self.raw_logits) != 2:
                raise ValueError('Two existing raw U student logits required')
            pseudo, mask = self.recovered
            first, second = (F.cross_entropy(v, pseudo, reduction='none', label_smoothing=.01)
                             for v in self.raw_logits)
            rescue = self.recipe['weight'] * (.5 * (first + second) * mask).mean()
            value = value + rescue
            self.recovery_calls += 1
            self.metrics.update(HPL_loss=value.detach(), HPL_rejected_U_CE=rescue.detach(),
                                HPL_rejected_U_used=self.used, HPL_rejected_U_probed=self.probed)
            return value
        finally:
            self.raw_logits = []
            self.recovered = self.candidates = None

    def state_dict(self):
        s = super().state_dict()
        expected = self.echo_pass if self.recipe['require_echo'] else self.branch_pass
        if (self.labeled is not None or self.recovered is not None or self.recovery_calls != self.calls
                or not 0 <= self.echo_pass <= self.branch_pass <= self.probed <= self.seen - self.accepted
                or self.used != expected):
            raise ValueError('The extra CE must be confined to source-certified rejected U')
        s['U_recovery'] = dict(recipe=self.recipe, calls=self.recovery_calls, probe_steps=self.probe_steps,
            extra_eval_forwards=2 * self.probe_steps, probed=self.probed, branch_pass=self.branch_pass,
            echo_pass=self.echo_pass, used=self.used)
        return s


class GuardedRecovery(Guarded, RejectedURecovery):
    pass


@contextmanager
def installed(c, holder):
    from experiments.cric_dura_comparison import t1_model

    with original_tri(c, holder, enabled=True, objective_factory=GuardedRecovery):
        old_install = t1_model.install

        def install(model, cfg):
            model = old_install(model, cfg)
            objective = holder['objective']
            original_hook = model._source_labeled_feature_hook

            def hook(self, **kwargs):
                labels, meta = kwargs['labels'], kwargs['extra'][1]
                rx = meta['rx_i'].to(labels.device).long()
                if set(rx.tolist()) - set(cfg['source_rxs']) or not bool(meta['tx_label_visible'].all()):
                    raise ValueError('Receiver-excluded certificates require true source L labels')
                objective.labeled = kwargs['labeled_inputs'].detach(), labels.detach(), rx
                try:
                    return original_hook(**kwargs)
                finally:
                    objective.labeled = None

            hook._accepts_labeled_inputs = True
            model._source_labeled_feature_hook = MethodType(hook, model)
            return model

        t1_model.install = install
        try:
            yield
        finally:
            t1_model.install = old_install
