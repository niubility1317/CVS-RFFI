"""Source-only CE versus full-L gradient and first-step counterfactuals.

All model copies and optimizer shadows are disposable. U truth, source
validation, and target data are not read. The first-step shadows start with
fresh AdamW moments and do not establish later-trajectory behavior.
"""
from contextlib import contextmanager
import json
import sys

import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.source_branch_gradients import _relation
from experiments.pamf_ssl_research.source_optimizer_counterfactual import _first_step_options
from experiments.pamf_ssl_research.source_optimizer_shadow import shadow_step
from . import source_adamw_diagnostic as original


def capture_identity_loss(model):
    """Temporarily expose the existing CE tensor while the native hook runs."""
    hook = model._source_labeled_feature_hook
    def observed(*args, **kwargs):
        if 'identity_loss' not in kwargs or hasattr(model, '_diagnostic_identity_loss'):
            raise ValueError('Exactly one existing labeled CE tensor is required')
        model._diagnostic_identity_loss = kwargs['identity_loss']
        try:
            return hook(*args, **kwargs)
        finally:
            del model._diagnostic_identity_loss
    observed._accepts_labeled_inputs = getattr(hook, '_accepts_labeled_inputs', False)
    model._source_labeled_feature_hook = observed
    return model


def anchor_facts(named, ce_gradient, full_gradient, raw_U, guarded_gradient,
                 baseline_descent, joint_descent):
    protected_U = original.difference(guarded_gradient, full_gradient, named)
    extra_descent = original.difference(joint_descent, baseline_descent, named)
    return {g: dict(CE_vs_full_L=_relation(named, ce_gradient, full_gradient, g),
                    CE_vs_raw_U=_relation(named, ce_gradient, raw_U, g),
                    CE_vs_guarded_U=_relation(named, ce_gradient, protected_U, g),
                    CE_vs_extra_adamw_descent=_relation(named, ce_gradient, extra_descent, g))
            for g in original.GROUPS}


class CEAnchorObjective(original.DiagnosticObjective):
    def guide(self, labeled, student):
        named = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
        ce = getattr(student, '_diagnostic_identity_loss', None)
        if not torch.is_tensor(ce) or ce.ndim or not ce.requires_grad:
            raise ValueError('Existing differentiable labeled CE is required')
        ce_gradient = tuple(None if value is None else value.detach().clone() for value in
            torch.autograd.grad(ce, [p for _, p in named], retain_graph=True, allow_unused=True))
        with original.capture_guard_derivatives() as gradients:
            result = super().guide(labeled, student)
        if len(gradients) != 2:
            raise ValueError('The original full L/U derivatives must be captured exactly once')
        full, raw_U = gradients
        guarded = self.diagnostic['expected_guarded_gradient']
        options = _first_step_options(self.c)
        baseline, _ = shadow_step(student, named, full, **options)
        joint, _ = shadow_step(student, named, guarded, **options)
        facts = anchor_facts(named, ce_gradient, full, raw_U, guarded, baseline, joint)
        for group, values in facts.items():
            self.diagnostic['groups'][group]['labeled_CE_anchor'] = values
        return result


@contextmanager
def installed():
    from experiments.cric_dura_comparison import t1_model
    from experiments.pamf_ssl_research import hard_pl_train
    old_scope, old_objective = hard_pl_train.installed, original.DiagnosticObjective

    @contextmanager
    def guarded_scope(config, holder, *, objective_cls):
        with old_scope(config, holder, objective_cls=objective_cls):
            final_install = t1_model.install
            def install(model, cfg):
                return capture_identity_loss(final_install(model, cfg))
            t1_model.install = install
            try:
                yield
            finally:
                t1_model.install = final_install

    hard_pl_train.installed, original.DiagnosticObjective = guarded_scope, CEAnchorObjective
    try:
        yield
    finally:
        hard_pl_train.installed, original.DiagnosticObjective = old_scope, old_objective


def collect(runtime):
    with installed():
        result = original.collect(runtime)
    result.update(status='SOURCE_CE_ANCHOR_DIAGNOSTIC_COMPLETE',
        disposable_optimizer_steps=4*len(result['rows']),
        extra_CE_autograd_calls=len(result['rows']),
        interpretation='Three original-E20 source batches per setting; '
            'actual disposable AdamW first steps use fresh moments. '
            'No inference about later optimizer moments or target performance.')
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.pamf_repl_weight.source_ce_anchor RUNTIME_JSON')
    print(json.dumps(collect(read(sys.argv[1])), allow_nan=False), flush=True)
