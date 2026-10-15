"""Own responsibility scratch20 freeze and native default-head provenance."""
import math
from pathlib import Path

from . import design as d
from experiments.cvs_sigtime_heldcredit.coordinate import PREFIX


RECIPE_FIELDS = (
    'run_id', 'candidate_id', 'output_dir', 'seed', 'epochs', 'label_epochs',
    'pseudo_epochs', 'use_unlabeled', 'use_ema_teacher', 'ema_decay',
    'pseudo_temporal_gate', 'pseudo_temporal_mode', 'pseudo_temporal_window',
    'pseudo_temporal_min_conf', 'pseudo_strong_agreement', 'pseudo_domain_gate',
    'pseudo_threshold_mode', 'pseudo_quantile', 'label_smoothing', 'lambda_u',
    'lambda_ent', 'phase1_lr_schedule', 'phase1_lr_min', 'use_aug',
    'use_concat_sat_channel_aug', 'use_sat_consistency', 'lambda_sat_cls',
    'lambda_sat_cons', 'sat_training_mode', 'from_scratch', 'baseline_ckpt',
    'teacher_ckpt', 'lr', 'weight_decay', 'batch_size', 'use_muse_ssdg',
    'wisig_out_len', 'wisig_domain', 'wisig_train_rxs', 'wisig_train_days',
    'wisig_test_rxs', 'wisig_test_days', 'split_mode', 'labeled_ratio',
    'unlabeled_ratio', 'source_val_ratio', 'phase1_source_role_protocol',
    'a1_source_screen_only', 'checkpoint_selection',
)


def validate_native_args(actual, c):
    """Compare this E20 row directly, without historical E50 recipe validators."""
    if not isinstance(actual, dict):
        raise ValueError('Actual native arguments are required')
    actual = actual.get('args', actual)
    expected = d.make_args(c, 'cpu')
    for key in RECIPE_FIELDS:
        if actual.get(key) != getattr(expected, key):
            raise ValueError('Actual E20 native recipe differs: ' + key)


def report_validator(report, c):
    if (report.get('arm') != 'SIG_TIME' or report.get('classifier_changed') is not False
            or report.get('default_head_only') is not True
            or report.get('separate_head_training') is not False
            or report.get('layer') != 't2' or report.get('output_embedding') != 160
            or report.get('sinc_coordinates') != 'CENTER_25K'
            or report.get('sinc_scale_hz') != 25000.
            or report.get('sinc_primary_path') != PREFIX
            or report.get('extra_mechanism_parameters') != 0
            or report.get('additional_classifier_parameters') != 0):
        raise ValueError('Actual CENTER25K/SIG_TIME/default-head structure differs')
    channels = report.get('local_channels')
    if not isinstance(channels, int) or report.get('extra_parameters') != 6 * channels + 18400:
        raise ValueError('Original SIG_TIME parameter count differs')


def validate_evidence(c, done, initial, freeze, resolved, actual_args,
                      mechanism, source_contract, expected_source):
    """Validate saved source metadata; this function loads no model or dataset."""
    d.validate(c)
    p = Path(c['output_root'])
    if (done.get('status') != 'SOURCE_TRAINED' or done.get('config') != c
            or done.get('epoch') != d.EPOCHS or done.get('logged_steps') != d.UPDATES
            or done.get('optimizer_steps') != d.UPDATES
            or done.get('L_context_steps') != d.UPDATES
            or done.get('target_access') is not False
            or done.get('target_evaluated') is not False
            or done.get('checkpoint_sources') != [] or done.get('ancestors') != []
            or done.get('initialization') != 'scratch'
            or done.get('baseline_recipe') != c['training_recipe']
            or done.get('label_epochs') != c['label_epochs']
            or done.get('pseudo_epochs') != c['pseudo_epochs']):
        raise ValueError('Own scratch20 source provenance or update budget differs')
    teacher_origin = ('own scratch student only'
                      if c['execution_overrides']['use_ema_teacher'] else 'disabled')
    expected_initial = dict(scratch_only=True, checkpoint_sources=[], ancestors=[],
                            source_roles='EXACT_MATCH', target_contact=False,
                            teacher_origin=teacher_origin, seed=c['model_seed'])
    expected_freeze = dict(status='SOURCE_FROZEN', config=c,
                           checkpoint=str(p / 'final_ssdg.pth'), epoch=d.EPOCHS,
                           selection='final_only', target_access=False)
    if initial != expected_initial or freeze != expected_freeze or resolved.get('config') != c:
        raise ValueError('Initialization, source freeze or resolved E20 row differs')
    if (done.get('identity_input_policy', {}).get('explicit_CFO_identity_input') is not False
            or resolved.get('identity_input_policy', {}).get('explicit_CFO_identity_input') is not False):
        raise ValueError('NoExplicitCFO identity input policy differs')
    if (done.get('native_fitted_probes_disabled') is not True
            or done.get('native_disabled_probe_calls') != 1 or done.get('head_fit_updates') != 0
            or resolved.get('native_fitted_probes_disabled') is not True):
        raise ValueError('Separate fitted native heads were not disabled')
    report = done.get('feature_extractor', {})
    report_validator(report, c)
    if (resolved.get('actual_feature_extractor') != report
            or done.get('mask_counts') is None or any(done['mask_counts'].values())
            or resolved.get('baseline_recipe') != c['training_recipe']
            or resolved.get('label_epochs') != c['label_epochs']
            or resolved.get('pseudo_epochs') != c['pseudo_epochs']):
        raise ValueError('Actual E20 mechanism or resolved recipe differs')
    validate_native_args(actual_args, c)
    if (mechanism.get('epoch') != d.EPOCHS
            or mechanism.get('L_context_steps') != d.UPDATES
            or mechanism.get('optimizer_steps') != d.UPDATES
            or mechanism.get('auxiliary_calls') != d.scheduled(c)
            or mechanism.get('optimizer_probe_steps') != list(d.PROBE_STEPS)
            or mechanism.get('default_head_joint_training') is not True
            or mechanism.get('original_U_sampler_unchanged') is not True
            or mechanism.get('head_fit_updates') != 0
            or mechanism.get('extra_actual_optimizer_steps') != 0
            or mechanism.get('U_truth_read') is not False
            or mechanism.get('target_access') is not False):
        raise ValueError('Actual scratch20 callbacks or optimization evidence incomplete')
    probes = mechanism.get('actual_probes', [])
    if [p.get('step') for p in probes] != list(d.PROBE_STEPS):
        raise ValueError('All four actual E20 optimization probe records are required')
    for probe in probes:
        epoch = (probe['step'] - 1) // d.STEPS + 1
        phase = 'label' if epoch <= c['label_epochs'] else 'pseudo'
        update = probe.get('actual_update', {})
        error = update.get('actual_shadow_max_abs_error')
        tolerance = update.get('actual_shadow_FP32_tolerance')
        if (probe.get('epoch') != epoch or probe.get('phase') != phase
                or probe.get('selected_W_gain_gradients_None') is not True
                or probe.get('main_base_includes_actual_U') is not True
                or probe.get('temporary_AdamW_steps') != d.TEMPORARY_STEPS_PER_PROBE
                or update.get('actual_optimizer_steps') != 1
                or update.get('actual_shadow_reference') != d.ACTUAL_SHADOW_REFERENCE
                or update.get('actual_shadow_per_element_allclose') is not True
                or update.get('actual_shadow_per_element_atol') != 2e-7
                or update.get('actual_shadow_per_element_rtol') != 2e-6
                or not isinstance(error, (int, float)) or not math.isfinite(error)
                or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance)
                or error < 0 or tolerance <= 0 or error > tolerance):
            raise ValueError('Actual once-only AdamW/default-head probe evidence differs')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(source_contract, expected_source)
    return p


def source_proof(c):
    """Check roles/ancestry/selection before any checkpoint deserialization."""
    d.validate(c)
    p = Path(c['output_root'])
    return validate_evidence(
        c, d.read(p / 'completion.json'), d.read(p / 'initialization.json'),
        d.read(p / 'source_frozen.json'), d.read(p / 'resolved_config.json'),
        d.read(p / 'actual_native_args.json'), d.read(p / 'source_training_mechanism.json'),
        d.read(p / 'source_contract.json'), d.read(d.SOURCE))


def validate_checkpoint(ck, c):
    if (ck.get('epoch') != d.EPOCHS or ck.get('run_id') != d.RUN
            or ck.get('candidate_id') != c['row_id']
            or ck.get('checkpoint_selection') != 'final_only'
            or ck.get('a1_ema_successful_updates') != d.UPDATES):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: own final20 row required')
    validate_native_args(ck.get('args'), c)
    ema = ck.get('ema_model')
    if c['execution_overrides']['use_ema_teacher']:
        if not isinstance(ema, dict) or not ema:
            raise ValueError('Own scratch EMA state is missing')
    elif ema is not None:
        raise ValueError('Pure CE control unexpectedly inherited an EMA state')


def own_model(c, device):
    from . import model
    import torch
    p = source_proof(c)
    ck = torch.load(p / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    validate_checkpoint(ck, c)
    result = model.build(c, device)
    result.load_state_dict(ck['model'], strict=True)
    return result.eval().requires_grad_(False)
