"""Fixed source-only feasibility of protecting actual quantized optimizer increments."""
from contextlib import contextmanager
import math
import torch

from . import source_branch_gradients as source
from . import source_optimizer_counterfactual as optimizer_source
from . import source_optimizer_shadow as shadow
from .parameter_step_projection import project_parameters

KIND = 'fixed_original_E20_source_step_projection_v1'
RULES = dict(minimum_whole_increment_norm_retention_per_seed=.5,
             minimum_removed_material_conflicts=1, material_cosine_threshold=-1e-6,
             all_actual_parameter_constraints_and_nonexpansion_required=True,
             all_three_fixed_seeds_required=True)


def validate_plan(plan):
    if (plan.get('kind') != KIND or plan.get('source_rules') != RULES
            or plan.get('seeds') != [2026092701, 2026092702, 2026092703]
            or plan.get('source_epoch') != 20 or plan.get('steps_per_seed') != 1
            or plan.get('L_batch') != 128 or plan.get('U_batch') != 256
            or any(plan.get(k) != 0 for k in ('V_reads', 'target_reads', 'U_truth_reads', 'G_training_reads'))):
        raise ValueError('Only the fixed source step-projection matrix and scientific rules are allowed')


@contextmanager
def installed_counterfactual():
    with optimizer_source.installed_counterfactual():
        original_facts = source.gradient_facts
        original_step = optimizer_source.shadow_step
        original_build = shadow.build
        snapshots = []

        def observed_step(*args, **kwargs):
            observed = {}

            def build(*build_args, **build_kwargs):
                optimizer = original_build(*build_args, **build_kwargs)
                parameters = list(build_args[1])
                def after_step(*unused):
                    observed['parameters'] = tuple(p.detach().clone() for p in parameters)
                optimizer.register_step_post_hook(after_step)
                return optimizer

            shadow.build = build
            try:
                result = original_step(*args, **kwargs)
            finally:
                shadow.build = original_build
            if 'parameters' not in observed:
                raise ValueError('Actual post-AdamW shadow parameters required')
            snapshots.append(observed['parameters'])
            return result

        def observe(named, losses, *, reference=None):
            snapshots.clear()
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
            if reference is None or len(derivatives) != 6 or len(snapshots) != 4:
                raise ValueError('Reuse exact original full/component derivatives and four actual optimizer shadows')
            baseline, raw_joint = snapshots[0], snapshots[3]
            protected, details = project_parameters(named, baseline, raw_joint, reference[0], derivatives[0], derivatives[1])
            raw = tuple(a.double() - b.double() for a, b in zip(baseline, raw_joint))
            actual = tuple(a.double() - b.double() for a, b in zip(baseline, protected))
            raw_norm = math.sqrt(sum(float(v.square().sum()) for v in raw))
            actual_norm = math.sqrt(sum(float(v.square().sum()) for v in actual))
            if raw_norm <= 0:
                raise ValueError('Nonzero actual BG incremental source update required')
            removed = []
            for group, values in details.items():
                if not values['actual_parameter_check'] or not values['norm_nonexpansion']:
                    raise ValueError('Actual quantized parameter constraints/nonexpansion required')
                for anchor, before in values['anchor_cosines_before'].items():
                    after = values['anchor_cosines_after'][anchor]
                    if before is not None and before < -1e-6 and (after is None or after >= -1e-6):
                        removed.append(dict(group=group, anchor=anchor, before_cosine=before, after_cosine=after,
                                            zero_increment=values['applied_increment_norm'] == 0,
                                            quantization_fallback=values['quantization_fallback']))
            facts['parameter_step_projection'] = dict(kind=KIND, groups=details,
                raw_whole_increment_norm=raw_norm, protected_whole_increment_norm=actual_norm,
                whole_increment_norm_retention=actual_norm / raw_norm,
                removed_material_conflicts=removed,
                quantization_fallback_groups=[g for g, d in details.items() if d['quantization_fallback']],
                exact_parameter_values_observed=True, protected_values_applied_to_original_model=False,
                extra_optimizer_steps=0, extra_autograd_calls=0, extra_model_forward_calls=0,
                interpretation='Compare exact FP64 differences of actual FP32 baseline/raw/protected parameters;protected values are computed and discarded,not written to any original model.Three anchors/parameter groups are correlated.')
            snapshots.clear()
            return facts, cap1

        optimizer_source.shadow_step = observed_step
        source.gradient_facts = observe
        try:
            yield
        finally:
            source.gradient_facts = original_facts
            optimizer_source.shadow_step = original_step
            shadow.build = original_build
            snapshots.clear()


def collect(runtime, plan, analysis_commit):
    validate_plan(plan)
    with installed_counterfactual():
        result = source.collect(runtime, plan, analysis_commit)
    observations = [row['source_branch_facts']['parameter_step_projection'] for row in result['rows']]
    minimum = min(row['whole_increment_norm_retention'] for row in observations)
    removed = sum(len(row['removed_material_conflicts']) for row in observations)
    passed = minimum >= RULES['minimum_whole_increment_norm_retention_per_seed'] and removed >= RULES['minimum_removed_material_conflicts']
    original_counts = {key: result[key] for key in ('optimizer_steps', 'parameter_updates', 'parameter_grad_writes', 'persisted_buffer_updates')}
    writes = sum(f['labeled_only_optimizer']['shadow_parameter_gradient_tensors']
                 + sum(v['optimizer']['shadow_parameter_gradient_tensors'] for v in f['comparisons'].values())
                 for row in result['rows'] for f in [row['source_branch_facts']['optimizer_counterfactual']])
    result.update(status='VERIFIED_FIXED_SOURCE_STEP_PROJECTION', kind=KIND, source_rules=RULES,
                  source_rule_pass=passed, scientific_status='SOURCE_COUNTERFACTUAL_PASSED' if passed else 'SCIENTIFIC_FALSIFIED_SOURCE',
                  minimum_whole_increment_norm_retention=minimum, removed_material_conflicts=removed,
                  quantization_fallback_groups=sum(len(row['quantization_fallback_groups']) for row in observations),
                  optimizer_steps=12, shadow_optimizer_steps=12, parameter_updates=12, parameter_grad_writes=writes,
                  original_model_update_counts=original_counts, shadow_parameter_updates_discarded=True,
                  new_formal_training_updates=0, protected_parameters_applied_to_original_model=False,
                  automatic_candidate_launch=False, goal_complete=False,
                  interpretation='One original-E20 L/U source step per seed.12 actual disposable optimizer shadows plus pure quantized increment projection,no extra forward/autograd,no original parameter/gradient/buffer updates,V/Utruth/target/Gtrain reads or healthy-job changes.Only a separately implemented/frozen/reviewed/smoked training candidate can follow a source pass;not target evidence or optimizer-trajectory monotonicity.')
    return result
