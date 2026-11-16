"""Cross the actual supervised loss parts with actual U loss parts at E20."""
from contextlib import contextmanager
import json
import sys
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_weight import source_ce_anchor
from experiments.pamf_ssl_research.source_branch_gradients import _PAIRS, _relation, _sum, component_closure, labeled_pairs
from . import source_component_diagnostic as base

_cross_facts = []


def labeled_components(pairs, total):
    if pairs is None or len(pairs) != 4:
        raise ValueError('Exactly four original labeled pair calls required')
    ce = .5*pairs[0][1] + .25*pairs[1][1] + .25*pairs[2][1] + .25*pairs[3][1]
    kl = .5*pairs[0][2] + .25*pairs[1][2] + .25*pairs[3][2]
    return dict(L_all_CE=ce, L_pair_KL=kl, L_residual=total-ce-kl)


@contextmanager
def capture_labeled_pairs():
    original_capture = source_ce_anchor.capture_identity_loss
    def install(model):
        model = original_capture(model)
        hook = model._source_labeled_feature_hook
        def observed(*args, **kwargs):
            with labeled_pairs():
                return hook(*args, **kwargs)
        observed._accepts_labeled_inputs = getattr(hook, '_accepts_labeled_inputs', False)
        model._source_labeled_feature_hook = observed
        return model
    source_ce_anchor.capture_identity_loss = install
    try:
        yield
    finally:
        source_ce_anchor.capture_identity_loss = original_capture


class CrossLossObjective(base.ComponentObjective):
    def guide(self, labeled, student):
        if self.c.get('ablation') != 'F21' or self.c.get('satellite_concat'):
            raise ValueError('Fixed F21 labeled slots and no appended satellite batch required')
        parts = labeled_components(_PAIRS.get(), labeled)
        named = [(n,p) for n,p in student.named_parameters() if p.requires_grad]
        params = [p for _,p in named]
        def gradient(loss):
            return tuple(None if x is None else x.detach().clone() for x in
                torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True))
        labeled_gradients = {name: gradient(value) for name,value in parts.items()}
        full_gradient = gradient(labeled)
        closure = component_closure(named, _sum(named, *labeled_gradients.values()), full_gradient)
        u_gradients = {name: gradient(self.components[name]) for name in base.PARTS}
        groups = {group: {left: {right: _relation(named, lg, ug, group)
                    for right, ug in u_gradients.items()}
                    for left, lg in labeled_gradients.items()}
                  for group in base.original.GROUPS}
        result = super().guide(labeled, student)
        _cross_facts.append(dict(L_component_gradient_closure=closure,
            L_component_losses={k: float(v.detach()) for k,v in parts.items()}, L_U_component_relations=groups))
        return result


def collect(runtime):
    if _cross_facts:
        raise ValueError('One source cross-loss diagnosis per process')
    previous = base.ComponentObjective
    base.ComponentObjective = CrossLossObjective
    try:
        with capture_labeled_pairs():
            result = base.collect(runtime)
    finally:
        base.ComponentObjective = previous
    if len(result['rows']) != len(_cross_facts):
        raise ValueError('Every source batch needs one L/U cross-component record')
    for row, facts in zip(result['rows'], _cross_facts):
        row.update(facts)
    _cross_facts.clear()
    result.update(status='SOURCE_CROSS_LOSS_DIAGNOSTIC_COMPLETE',
        interpretation='Three fixed E20 source batches per setting; the residual of the full labeled loss '
            'is recorded as L_residual, not assumed to be exclusively contrast. Both L and U component '
            'gradient closures are verified. Pairwise local source relations do not prove target effects.')
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.pamf_repl_attribution_factorial.source_cross_loss_diagnostic RUNTIME_JSON')
    print(json.dumps(collect(read(sys.argv[1])), ensure_ascii=False, allow_nan=False), flush=True)
