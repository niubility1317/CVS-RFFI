"""Strict prepared-only native DR/EG profiles. No target-driven selection."""
from copy import deepcopy
import math

DEFAULTS = dict(model_seed=392005, data_seed=392005, augmentation_seed=392005,
    evaluation_seed=392005, epochs=200, steps_per_epoch=222, labeled_batch=128,
    unlabeled_batch=256, initialization='scratch', precision='fp32', lr=2e-4,
    weight_decay=1e-4, max_grad_norm=5., solver_mode='simultaneous',
    controller_mode='off', curriculum_mode='fixed', predictor_lr_ratio=1.,
    normalization_scales='legacy_reestimate', native_dr=True,
    daot_feature=.5, daot_logit=.2, labeled_encoder_grl_multiplier=1.,
    unlabeled_encoder_grl_multiplier=0., outer_adv_weight=.35,
    disable_adversarial_head_loss=False, hp_ramp=False, diagnostic_interval=1000,
    probe_interval=1000, launch=False)

def resolve(config, overrides=None):
    values=deepcopy(DEFAULTS)
    for layer in (config, overrides or {}):
        unknown=set(layer)-set(values)
        if unknown: raise ValueError('Unknown joint settings: '+','.join(sorted(unknown)))
        values.update(layer)
    fixed=dict(epochs=200,steps_per_epoch=222,labeled_batch=128,unlabeled_batch=256,
        initialization='scratch',precision='fp32',lr=2e-4,weight_decay=1e-4,
        max_grad_norm=5.,controller_mode='off',curriculum_mode='fixed',
        unlabeled_encoder_grl_multiplier=0.,launch=False)
    for key,expected in fixed.items():
        if values[key]!=expected: raise ValueError(f'{key} must be {expected!r} in this prepared release')
    if values['solver_mode'] not in ('simultaneous','full_EG'): raise ValueError('fixed solver required; adaptive is deferred')
    if values['normalization_scales'] not in ('legacy_reestimate','reuse_origin_used_scales'): raise ValueError('normalization mode')
    for key in ('native_dr','hp_ramp','disable_adversarial_head_loss','launch'):
        if type(values[key]) is not bool: raise ValueError(key+' must be a boolean')
    for key in ('model_seed','data_seed','augmentation_seed','evaluation_seed','diagnostic_interval','probe_interval'):
        if type(values[key]) is not int or values[key]<0: raise ValueError(key+' must be a nonnegative integer')
    for key in ('predictor_lr_ratio','daot_feature','daot_logit','labeled_encoder_grl_multiplier','outer_adv_weight'):
        if not isinstance(values[key],(float,int)) or not math.isfinite(values[key]) or values[key]<0: raise ValueError(key)
    if not .25<=values['predictor_lr_ratio']<=1.: raise ValueError('predictor ratio must be in [0.25,1]')
    if values['disable_adversarial_head_loss'] != (values['outer_adv_weight']==0):
        raise ValueError('adv=0 disables both L adversarial CE and U adversarial-head CE, retaining z_dom CE')
    return values

def make_row(row_id, settings):
    c=resolve(settings)
    return dict(id=row_id,joint=c,carrier='native_random',daot_rc4=c['native_dr'],
        dr_full_extensions=False,x_enabled=False,u_enabled=False,control='off',
        curriculum='fixed',source_audit_policy='off',cstar_action_enabled=False,
        ticket_curriculum_enabled=False,epoch_budget='full_unlabeled',unlabeled_batch=256)

def epoch_for_accepted_step(accepted_before, steps_per_epoch=222):
    if accepted_before<0 or steps_per_epoch<1: raise ValueError('invalid accepted clock')
    return 1+accepted_before//steps_per_epoch

def hp_multiplier(epoch, enabled):
    # Inclusive endpoints: E21=0, E40=1. Identity-only, not domain/self.
    return min(1.,max(0.,(epoch-21)/19.)) if enabled else 1.

def dr_overrides(settings):
    c=resolve(settings)
    return dict(daot_teacher_view_count=2,daot_aggregation='mean',
        daot_lambda_orbit_z=c['daot_feature'],daot_lambda_orbit_logit=c['daot_logit'],
        daot_lambda_orbit_proto=0.,daot_lambda_orbit_relation=0.,daot_lambda_tangent=0.,
        daot_lambda_nuisance=0.,daot_lambda_fingerprint=0.,daot_enable_relation=False,
        rc4_enable_hard=True,rc4_enable_partial=True,rc4_enable_negative=False,
        rc4_use_anchor=False,rc4_satellite_hard_only=False,rc4_lambda_satellite=0.,
        rc4_lambda_feature_anchor=0.,rc4_total_identity_effective_budget=.15,
        rc4_lambda_hard=.6,rc4_lambda_partial=.4,rc4_identity_tail_partial_conditional_final=0.)
