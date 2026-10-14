"""Focused metadata/recipe/checkpoint negatives; no datasets or trained weights."""
import copy
import json
from pathlib import Path

from . import design as d
from .provenance import (RECIPE_FIELDS, PREFIX, validate_evidence,
                         validate_checkpoint)


def fixture(c):
    p = Path(c['output_root'])
    ssl = c['execution_overrides']['use_ema_teacher']
    report = dict(arm='SIG_TIME', classifier_changed=False, default_head_only=True,
                  separate_head_training=False, layer='t2', output_embedding=160,
                  local_channels=64, extra_parameters=6 * 64 + 18400,
                  sinc_coordinates='CENTER_25K', sinc_scale_hz=25000.,
                  sinc_primary_path=PREFIX, extra_mechanism_parameters=0,
                  additional_classifier_parameters=0)
    done = dict(status='SOURCE_TRAINED', config=c, epoch=20, logged_steps=4440,
                optimizer_steps=4440, L_context_steps=4440, target_access=False,
                target_evaluated=False, checkpoint_sources=[], ancestors=[],
                initialization='scratch', baseline_recipe=c['training_recipe'],
                label_epochs=c['label_epochs'], pseudo_epochs=c['pseudo_epochs'],
                identity_input_policy=dict(explicit_CFO_identity_input=False),
                native_fitted_probes_disabled=True, native_disabled_probe_calls=1,
                head_fit_updates=0, feature_extractor=report, mask_counts={'unused': 0})
    initial = dict(scratch_only=True, checkpoint_sources=[], ancestors=[],
                   source_roles='EXACT_MATCH', target_contact=False,
                   teacher_origin='own scratch student only' if ssl else 'disabled',
                   seed=c['model_seed'])
    freeze = dict(status='SOURCE_FROZEN', config=c, checkpoint=str(p / 'final_ssdg.pth'),
                  epoch=20, selection='final_only', target_access=False)
    resolved = dict(config=c, identity_input_policy=dict(explicit_CFO_identity_input=False),
                    native_fitted_probes_disabled=True, actual_feature_extractor=report,
                    baseline_recipe=c['training_recipe'], label_epochs=c['label_epochs'],
                    pseudo_epochs=c['pseudo_epochs'])
    expected = d.make_args(c, 'cpu')
    args = {key: getattr(expected, key) for key in RECIPE_FIELDS}
    mechanism = dict(epoch=20, L_context_steps=4440, optimizer_steps=4440,
                     auxiliary_calls=250, optimizer_probe_steps=list(d.PROBE_STEPS),
                     default_head_joint_training=True, original_U_sampler_unchanged=True,
                     head_fit_updates=0, extra_actual_optimizer_steps=0,
                     U_truth_read=False, target_access=False)
    mechanism['actual_probes'] = [dict(
        step=step, epoch=(step - 1) // d.STEPS + 1,
        phase='label' if (step - 1) // d.STEPS + 1 <= c['label_epochs'] else 'pseudo',
        selected_W_gain_gradients_None=True, main_base_includes_actual_U=True,
        temporary_AdamW_steps=2, actual_update=dict(actual_optimizer_steps=1,
            actual_shadow_max_abs_error=0., actual_shadow_FP32_tolerance=1e-6))
        for step in d.PROBE_STEPS]
    source = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
                  capsule_id='synthetic-capsule', split_id='synthetic-split',
                  role_ids={'L_s': [[0, 1, 1, 0, 0]], 'U_s': [[0, 1, 1, 1, 1]]})
    contract = dict(source, native_role_comparison='EXACT_MATCH')
    ck = dict(epoch=20, run_id=d.RUN, candidate_id=c['row_id'],
              checkpoint_selection='final_only', a1_ema_successful_updates=4440,
              args=args, ema_model={'synthetic_tensor': 1} if ssl else None)
    return dict(done=done, initial=initial, freeze=freeze, resolved=resolved,
                actual_args=args, mechanism=mechanism, source_contract=contract,
                expected_source=source), ck


def rejects(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError('Invalid source/checkpoint evidence was accepted')


def check():
    negatives = 0
    for arm in d.ARMS:
        c = d.config(next(r for r in d.rows() if r['arm'] == arm and r['model_seed'] == d.SEEDS[0]))
        evidence, ck = fixture(c)
        assert validate_evidence(c, **evidence) == Path(c['output_root'])
        validate_checkpoint(ck, c)
        changes = (
            ('done', 'epoch', 50), ('done', 'optimizer_steps', 11100),
            ('done', 'L_context_steps', 11100), ('done', 'target_access', True),
            ('done', 'ancestors', ['old-model']), ('done', 'head_fit_updates', 1),
            ('done', 'native_disabled_probe_calls', 0),
            ('initial', 'teacher_origin', 'external checkpoint'),
            ('freeze', 'epoch', 50), ('freeze', 'selection', 'target_best'),
            ('actual_args', 'use_unlabeled', not c['execution_overrides']['use_unlabeled']),
            ('actual_args', 'use_ema_teacher', not c['execution_overrides']['use_ema_teacher']),
            ('actual_args', 'use_aug', True), ('actual_args', 'lambda_u', .99),
            ('actual_args', 'candidate_id', 'other-row'),
            ('mechanism', 'auxiliary_calls', 624),
            ('mechanism', 'optimizer_probe_steps', [1120, 4448, 7776, 11088]),
            ('mechanism', 'U_truth_read', True),
            ('mechanism', 'actual_probes', []),
            ('source_contract', 'capsule_id', 'different-capsule'),
            ('source_contract', 'role_ids', {'L_s': [], 'U_s': []}),
        )
        for section, key, value in changes:
            altered = copy.deepcopy(evidence)
            altered[section][key] = value
            rejects(lambda: validate_evidence(c, **altered))
            negatives += 1
        altered = copy.deepcopy(evidence)
        altered['done']['feature_extractor']['sinc_coordinates'] = 'RAW_HZ'
        rejects(lambda: validate_evidence(c, **altered))
        negatives += 1
        altered = copy.deepcopy(evidence)
        altered['done']['feature_extractor']['classifier_changed'] = True
        rejects(lambda: validate_evidence(c, **altered))
        negatives += 1
        altered = copy.deepcopy(evidence)
        altered['done']['identity_input_policy']['explicit_CFO_identity_input'] = True
        rejects(lambda: validate_evidence(c, **altered))
        negatives += 1
        for key, value in (('epoch', 50), ('run_id', 'old-run'), ('candidate_id', 'wrong-row'),
                           ('a1_ema_successful_updates', 11100), ('checkpoint_selection', 'best')):
            wrong = copy.deepcopy(ck)
            wrong[key] = value
            rejects(lambda: validate_checkpoint(wrong, c))
            negatives += 1
        wrong = copy.deepcopy(ck)
        wrong['ema_model'] = None if c['execution_overrides']['use_ema_teacher'] else {'old_teacher': 1}
        rejects(lambda: validate_checkpoint(wrong, c))
        negatives += 1
    # Paired error arithmetic is truth-side only; this fixture contains no labels.
    import numpy as np
    from .score import paired_errors
    control = np.zeros(168000, dtype=np.bool_)
    treatment = control.copy()
    control[:10] = True
    treatment[:7] = True
    treatment[10:15] = True
    paired = paired_errors(control, treatment)
    assert paired['corrected_errors'] == 5 and paired['introduced_errors'] == 3
    assert paired['net_correct_change'] == 2
    assert sum(paired[k] for k in ('corrected_errors', 'introduced_errors', 'both_correct', 'both_wrong')) == 168000
    rejects(lambda: paired_errors(control[:-1], treatment))
    # A rejected source proof must stop before checkpoint deserialization.
    import torch
    from . import provenance
    original_proof, original_load = provenance.source_proof, torch.load
    def forbidden_load(*args, **kwargs):
        raise AssertionError('Checkpoint was opened before a valid source proof')
    def invalid_source(config):
        raise ValueError('synthetic invalid source contract')
    provenance.source_proof, torch.load = invalid_source, forbidden_load
    try:
        rejects(lambda: provenance.own_model(c, 'cpu'))
    finally:
        provenance.source_proof, torch.load = original_proof, original_load
    return dict(status='PASS', source_metadata_negatives=negatives,
                all_five_E20_recipes=True, old_E50_and_wrong_CP_rejected=True,
                U_EMA_head_data_and_ancestry_mismatch_rejected=True,
                paired_error_census=True, proof_precedes_checkpoint_load=True,
                actual_data_or_checkpoint_reads=0,
                actual_optimizer_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False, indent=2))
