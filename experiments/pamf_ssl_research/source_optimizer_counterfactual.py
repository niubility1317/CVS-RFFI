"""Observe actual fresh physical-AdamW steps on a fixed source-only diagnostic."""
from contextlib import contextmanager
import torch

from . import source_branch_gradients as source
from .branch_gradient_projection import branch_projected_gradients, GROUPS, ANCHORS
from .source_cap2_projection_reference import projected_gradients
from .source_optimizer_shadow import shadow_step

KIND = 'fixed_original_E20_source_optimizer_counterfactual_v1'
COMPARISONS = ['L_ONLY', 'NATIVE_CAP1', 'NATIVE_CAP2', 'BRANCH_CAP2_QR']


def validate_plan(plan):
    if (plan.get('kind') != KIND or plan.get('comparisons') != COMPARISONS
            or plan.get('seeds') != [2026092701, 2026092702, 2026092703]
            or plan.get('source_epoch') != 20 or plan.get('steps_per_seed') != 1
            or plan.get('L_batch') != 128 or plan.get('U_batch') != 256
            or plan.get('optimizer_state') != 'fresh_moments_first_new_native10_step'
            or any(plan.get(k) != 0 for k in ('V_reads', 'target_reads', 'U_truth_reads', 'G_training_reads'))):
        raise ValueError('Only the fixed source-only original-E20 fresh-first-step comparison is allowed')


def _difference(left, right, named):
    return tuple(None if a is None and b is None else
                 (torch.zeros_like(p) if a is None else a) - (torch.zeros_like(p) if b is None else b)
                 for (_, p), a, b in zip(named, left, right))


def _first_step_options(config):
    from experiments.cric_dura_comparison.f_train import arguments
    from experiments.cvs_phase1_repair.design import cosine_lr
    args = arguments(config, 'cpu')
    # f_train.train selects native_scope(owner, 'CENTER_UNITS') directly.
    # These current configs do not carry the historical combination field;
    # shadow_step validates the actual model's centered physical Sinc metadata.
    if (args.epochs != 10 or args.max_grad_norm != 5.
            or getattr(args, 'phase1_lr_schedule', None) != 'cosine'):
        raise ValueError('The actual first native10 cosine/physical-clip path differs')
    return dict(base_lr=float(cosine_lr(float(args.lr), 1, args.epochs, float(args.phase1_lr_min))),
                weight_decay=float(args.weight_decay))


@contextmanager
def installed_counterfactual():
    original_facts, original_class = source.gradient_facts, source.BranchDiagnostic
    current = {}

    class OptimizerDiagnostic(original_class):
        def guide(self, labeled, student):
            if current:
                raise ValueError('One active disposable original-source model required')
            current.update(model=student, config=self.c)
            try:
                return super().guide(labeled, student)
            finally:
                current.clear()

    def observe(named, losses, *, reference=None):
        if reference is None or not current:
            raise ValueError('Exact original gradients and active source model required')
        derivatives = []
        original_grad = torch.autograd.grad

        def capture(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            derivatives.append(result)
            return result

        torch.autograd.grad = capture
        try:
            facts, cap1 = original_facts(named, losses, reference=reference)
        finally:
            torch.autograd.grad = original_grad
        if len(derivatives) != 6:
            raise ValueError('Reuse exactly the six existing component derivatives')
        full_l, full_u = reference
        cap2, cap2_details = projected_gradients(named, full_l, full_u, max_u_ratio=2.)
        branch, branch_details = branch_projected_gradients(named, full_l, full_u,
                                                            derivatives[0], derivatives[1], max_u_ratio=2.)
        anchors = dict(zip(ANCHORS, (full_l, derivatives[0], derivatives[1])))
        options = _first_step_options(current['config'])
        baseline, baseline_metadata = shadow_step(current['model'], named, full_l, **options)
        observations = {}
        for method, combined in (('NATIVE_CAP1', cap1), ('NATIVE_CAP2', cap2), ('BRANCH_CAP2_QR', branch)):
            descent, metadata = shadow_step(current['model'], named, combined, **options)
            extra_step = _difference(descent, baseline, named)
            extra_gradient = _difference(combined, full_l, named)
            relations = {}
            for group in GROUPS:
                relations[group] = {anchor: dict(
                    gradient=source._relation(named, vector, extra_gradient, group),
                    optimizer_extra_descent=source._relation(named, vector, extra_step, group),
                    optimizer_total_descent=source._relation(named, vector, descent, group))
                    for anchor, vector in anchors.items()}
            observations[method] = dict(optimizer=metadata, relations=relations)
        facts['optimizer_counterfactual'] = dict(
            kind=KIND, optimizer_state='fresh_moments_first_new_native10_step',
            labeled_only_optimizer=baseline_metadata, comparisons=observations,
            raw_cap2_gradient_details=cap2_details, raw_branch_gradient_details=branch_details,
            shadow_optimizer_steps=4, original_parameter_or_gradient_writes=0,
            extra_autograd_calls=0, extra_model_forward_calls=0,
            sign_convention='descent=theta_before-theta_after;extra_U_descent=descent(L+protectedU)-descent(L).Negative anchor dot means a worse first-order anchor change relative to the actual L-only step,not a measured loss increase.',
            limitations='One fresh first-step snapshot;no inherited moments,trajectory,post-step loss evaluation or target-performance claim;all parameter-only shadows discarded.')
        return facts, cap1

    source.gradient_facts, source.BranchDiagnostic = observe, OptimizerDiagnostic
    try:
        yield
    finally:
        source.gradient_facts, source.BranchDiagnostic = original_facts, original_class
        current.clear()


def collect(runtime, plan, analysis_commit):
    validate_plan(plan)
    with installed_counterfactual():
        result = source.collect(runtime, plan, analysis_commit)
    conflicts = []
    for row in result['rows']:
        facts = row['source_branch_facts']['optimizer_counterfactual']
        for method, values in facts['comparisons'].items():
            for group, anchors in values['relations'].items():
                for anchor, relation in anchors.items():
                    before = relation['gradient']['cosine']
                    after = relation['optimizer_extra_descent']['cosine']
                    if before is not None and after is not None and before >= -1e-6 and after < -1e-6:
                        conflicts.append(dict(model_seed=row['model_seed'], method=method,
                                              group=group, anchor=anchor,
                                              protected_gradient_cosine=before,
                                              optimizer_extra_descent_cosine=after))
    original_counts = {key: result[key] for key in ('optimizer_steps', 'parameter_updates', 'parameter_grad_writes', 'persisted_buffer_updates')}
    shadow_gradient_writes = sum(facts['labeled_only_optimizer']['shadow_parameter_gradient_tensors']
                                + sum(item['optimizer']['shadow_parameter_gradient_tensors'] for item in facts['comparisons'].values())
                                for row in result['rows'] for facts in [row['source_branch_facts']['optimizer_counterfactual']])
    result.update(status='VERIFIED_FIXED_SOURCE_OPTIMIZER_COUNTERFACTUAL', kind=KIND,
                  comparisons=COMPARISONS, shadow_optimizer_steps=12,
                  optimizer_steps=12, parameter_updates=12, parameter_grad_writes=shadow_gradient_writes,
                  original_model_update_counts=original_counts,
                  shadow_parameter_updates_discarded=True, new_formal_training_updates=0,
                  nonnegative_gradient_but_negative_optimizer_extra_descent=conflicts,
                  scientific_promotion_threshold=None, automatic_candidate_launch=False,
                  interpretation='Original-E20 source gradients and original cap1 bridge unchanged;four actual CPU FP32 fresh CENTER_UNITS/physicalclip5/AdamW steps per seed on independent parameter-only shadows.Original model optimizer/parameter/gradient writes remain zero;12 disposable optimizer steps are additional diagnostic compute.Gradient signs do not prove optimizer-step or target monotonicity.')
    return result
