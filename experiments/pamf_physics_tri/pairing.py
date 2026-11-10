"""Change only the detached shared cross-receiver partner plan of original REPL."""
from contextlib import contextmanager

import torch

from experiments.pamf_physics_branch.geometry import soft_distances
from experiments.pamf_physics_branch.objective import physical_probes
from experiments.pamf_repl_view_shuffle.source import installed as original_tri
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from experiments.pamf_ssl_research.u_mixstyle_objective import partner_eligibility


BRANCHES = ('time', 'frequency', 'pa', 'path', 'reference')


def branch_partners(features, directions, eligible, beta):
    scores = [soft_distances(features[name], directions[name], beta) for name in BRANCHES]
    ordinary = torch.stack([a for a, _ in scores]).mean(0)
    adjusted = torch.stack([b for _, b in scores]).mean(0)
    nearest = adjusted.masked_fill(~eligible, torch.inf).argmin(1)
    uncorrected = ordinary.masked_fill(~eligible, torch.inf).argmin(1)
    return nearest, int((nearest != uncorrected).sum()), float((ordinary - adjusted)[eligible].sum())


class BranchPairedREPL(GuardedReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self.beta = c['receiver_pairing']['beta']
        self.pair_plan = None
        self.pair_calls = self.probed = self.probe_steps = self.changed_random = self.changed_physics = 0
        self.eligible_edges = 0
        self.discount_sum = 0.

    def validate_unlabeled_batch(self, batch):
        super().validate_unlabeled_batch(batch)
        visible = batch[3]['tx_label_visible']
        if visible.shape != batch[1].shape or visible.dtype != torch.bool or bool(visible.any()):
            raise ValueError('Only hidden-label source U may define receiver pairs')

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        if self.pair_plan is not None:
            raise ValueError('Previous U receiver pairing plan was not consumed')
        # This hook precedes the six training passes and their feature captures.
        eligible = partner_eligibility(pseudo, accepted, metadata['rx_i'].long())
        valid = eligible.any(1)
        selected = valid.nonzero(as_tuple=True)[0]
        partners = torch.arange(len(x), device=x.device)
        if selected.numel():
            seed = self.c['model_seed'] * 100_000 + step
            features, directions, _ = physical_probes(student, x[selected], seed)
            candidates = eligible[selected][:, selected]
            nearest, changed, discount = branch_partners(features, directions, candidates, self.beta)
            partners[selected] = selected[nearest]
            self.probe_steps += 1
            self.probed += len(selected)
            self.changed_physics += changed
            self.eligible_edges += int(candidates.sum())
            self.discount_sum += discount
        self.pair_plan = partners, valid
        return mixed

    def select_style_partners(self, original, original_valid):
        if self.pair_plan is None:
            raise ValueError('Receiver pairing must precede the first original U pass')
        partners, valid = self.pair_plan
        self.pair_plan = None
        if not torch.equal(valid, original_valid):
            raise ValueError('Original pseudo-label cross-receiver eligibility differs')
        self.pair_calls += 1
        self.changed_random += int((partners != original).sum())
        return partners

    def state_dict(self):
        state = super().state_dict()
        if (self.pair_plan is not None or self.pair_calls != self.calls
                or 6 * self.probed != self.style_eligible or 2 * self.probed != self.rxe_eligible
                or (self.beta == 0 and (self.changed_physics or self.discount_sum))):
            raise ValueError('Both original mechanisms must use the same one-per-step receiver pairing')
        state['U_branch_pairing'] = dict(beta=self.beta, branches=list(BRANCHES), calls=self.pair_calls,
            probed=self.probed, probe_steps=self.probe_steps, extra_eval_forwards=2 * self.probe_steps,
            changed_vs_random=self.changed_random, changed_by_physics=self.changed_physics,
            mean_eligible_discount=self.discount_sum / max(1, self.eligible_edges))
        return state


@contextmanager
def installed(c, holder):
    with original_tri(c, holder, enabled=True, objective_factory=BranchPairedREPL):
        yield
