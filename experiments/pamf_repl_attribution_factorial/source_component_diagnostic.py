"""Source-only decomposition of actual Tri L and U gradient components."""
from contextlib import contextmanager
import json
import sys
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_weight import source_adamw_diagnostic as original
from experiments.pamf_repl_weight import source_ce_anchor
from experiments.pamf_ssl_research.source_branch_gradients import _relation, _sum, component_closure

PARTS = ('pseudo_CE', 'pair_KL', 'shallow', 'joint', 'extrap_delta')
_active = {}
_facts = []


def decompose_mechanism(o0, o1, m0, m1, r0, r1, y, accepted, physical, aux, *, weight):
    from experiments.pamf_ssl_research.hard_pl_objective import masked_pair
    ordinary = masked_pair(o0, o1, y, accepted, 1.)
    mixed = masked_pair(m0[~physical], m1[~physical], y[~physical], accepted[~physical], 1.)
    channel = masked_pair(m0[physical], m1[physical], y[physical], accepted[physical], 0.)
    raw = masked_pair(r0, r1, y, accepted, 1.)
    scale = weight/1.25
    return dict(pseudo_CE=scale*(.5*ordinary[1]+.25*mixed[1]+.25*channel[1]+.25*raw[1]),
                pair_KL=scale*(.5*ordinary[2]+.25*mixed[2]+.25*raw[2]),
                shallow=scale*.1*aux)


@contextmanager
def capture_components():
    from experiments.pamf_ssl_research import hard_pl_objective as hard, hard_pl_joint_objective as joint
    from experiments.pamf_ssl_research import hard_pl_receiver_extrapolation_objective as receiver
    old = hard.mechanism_loss, hard.HardPLObjective.loss, joint.JointNoAsMixStyle.loss, receiver.ReceiverExtrapolation.loss
    def mechanism(*args):
        objective = _active['objective']
        objective.components.update(decompose_mechanism(*args, weight=objective.settings['weight']))
        return old[0](*args)
    def base_loss(self, student, device):
        value = old[1](self, student, device)
        self._component_base = value
        return value
    def joint_loss(self, student, device):
        value = old[2](self, student, device)
        self.components['joint'] = value - self._component_base
        self._component_joint = value
        return value
    def receiver_loss(self, student, device):
        value = old[3](self, student, device)
        self.components['extrap_delta'] = value - self._component_joint
        return value
    hard.mechanism_loss, hard.HardPLObjective.loss = mechanism, base_loss
    joint.JointNoAsMixStyle.loss, receiver.ReceiverExtrapolation.loss = joint_loss, receiver_loss
    try:
        yield
    finally:
        hard.mechanism_loss, hard.HardPLObjective.loss, joint.JointNoAsMixStyle.loss, receiver.ReceiverExtrapolation.loss = old


class ComponentObjective(source_ce_anchor.CEAnchorObjective):
    def loss(self, student, device):
        if _active:
            raise ValueError('One component capture at a time')
        self.components = {}
        _active['objective'] = self
        try:
            return super().loss(student, device)
        finally:
            _active.clear()

    def guide(self, labeled, student):
        if set(self.components) != set(PARTS) or self.pending_gradient_loss is None:
            raise ValueError('All five actual U components are required')
        torch.testing.assert_close(sum(self.components.values()), self.pending_gradient_loss)
        named = [(n,p) for n,p in student.named_parameters() if p.requires_grad]
        params = [p for _,p in named]
        ce = getattr(student, '_diagnostic_identity_loss', None)
        if not torch.is_tensor(ce) or not ce.requires_grad:
            raise ValueError('Actual labeled CE required')
        def gradient(loss):
            return tuple(None if x is None else x.detach().clone() for x in
                torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True))
        lg, cg, ug = gradient(labeled), gradient(ce), gradient(self.pending_gradient_loss)
        components = {name: gradient(self.components[name]) for name in PARTS}
        closure = component_closure(named, _sum(named, *components.values()), ug)
        result = super().guide(labeled, student)
        for group in original.GROUPS:
            self.diagnostic['groups'][group]['U_components'] = {
                name: dict(vs_L_full=_relation(named, lg, value, group),
                           vs_labeled_CE=_relation(named, cg, value, group),
                           vs_U_full=_relation(named, ug, value, group))
                for name, value in components.items()}
        _facts.append(dict(component_gradient_closure=closure,
                           component_losses={k: float(v.detach()) for k,v in self.components.items()}))
        return result


def attach_facts(rows, facts):
    if len(rows) != len(facts):
        raise ValueError('Every source batch must have one component closure record')
    for row, value in zip(rows, facts):
        row.update(value)


def collect(runtime):
    if _facts:
        raise ValueError('A single component diagnosis per process is required')
    with source_ce_anchor.installed(), capture_components():
        original.DiagnosticObjective = ComponentObjective
        result = original.collect(runtime)
    attach_facts(result['rows'], _facts)
    _facts.clear()
    result.update(status='SOURCE_COMPONENT_DIAGNOSTIC_COMPLETE',
        interpretation='Three fixed E20 source batches per setting; all terms reuse the actual original '
            'U graph and close to the native total gradient. Local source relations do not prove '
            'long-run AdamW behavior or target performance.')
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.pamf_repl_attribution_factorial.source_component_diagnostic RUNTIME_JSON')
    print(json.dumps(collect(read(sys.argv[1])), ensure_ascii=False, allow_nan=False), flush=True)
