"""Independent, prepared-only response-game experiments."""
from copy import deepcopy
import math
from .joint_config import make_row as native_row

DEFAULTS=dict(method='CF_EG', encoder_adversary=True, encoder_multiplier=1., unlabeled_head=True, beta_candidates=[1.,.5,0.],
    margin_tolerance=.01, cf_interval=1, fixed_beta=None, random_beta=False,
    transport_gamma=.5, transport_rho=.01, unmatched_rotation=False,
    xt_mu=.1, xt_interval=20, xt_inner_lr_ratio=1., xt_exposure_control=False,
    dric_interval=4, identity_budget=.1, trust_ratio=.25, curvature_damping=.1,
    min_monotonicity=1e-4, residual_tolerance=1e-5, max_iterations=5,
    drift=True, identity_constraint=True, shuffle_drift=False, decompose_interval=1000, launch=False,
    implementation='cached',ir_start_epoch=21,ir_kappa=.25,ir_cg_max_iterations=2,ir_cg_rtol=.001,
    ir_origin_scales='reuse_origin_used_scales',br_refresh_interval=8,
    ir_curvature='ggn',ir_feature_anchor='predictor')

IR_DEFAULT_KEYS=('implementation','ir_start_epoch','ir_kappa','ir_cg_max_iterations',
    'ir_cg_rtol','ir_origin_scales','br_refresh_interval','ir_curvature','ir_feature_anchor')

def resolve(values):
    if set(values)-set(DEFAULTS):raise ValueError('unknown response settings')
    c=deepcopy(DEFAULTS);c.update(values)
    if c['method'] not in ('CF_EG','TR_EG','XT_DANN','DRIC','EG','SIM','RK2','FR','CGD','TASK_PROJECT','IR_EG','BR_IR_EG'):
        raise ValueError('unknown response method')
    if c['launch'] is not False:raise ValueError('prepared only')
    for k in ('cf_interval','xt_interval','dric_interval','max_iterations','decompose_interval'):
        if type(c[k]) is not int or c[k]<1:raise ValueError(k)
    if c['max_iterations']>5:raise ValueError('at most five local iterations')
    for k in ('encoder_adversary','unlabeled_head','random_beta','unmatched_rotation','xt_exposure_control','drift','identity_constraint','shuffle_drift'):
        if type(c[k]) is not bool:raise ValueError(k+' must be boolean')
    for k in ('encoder_multiplier','margin_tolerance','transport_gamma','transport_rho','xt_mu','identity_budget','trust_ratio','curvature_damping','min_monotonicity','residual_tolerance','xt_inner_lr_ratio'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k]<0:raise ValueError(k)
    if c['min_monotonicity']==0 or c['residual_tolerance']==0:raise ValueError('strictly positive local solver tolerances required')
    if not 0<=c['transport_gamma']<=1:raise ValueError('gamma')
    candidates=c['beta_candidates']
    if not isinstance(candidates,list) or not candidates or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in candidates):raise ValueError('finite numeric beta list required')
    if candidates!=sorted(set(candidates),reverse=True) or candidates[-1]!=0:raise ValueError('descending beta candidates ending in zero')
    if c['fixed_beta'] is not None and (type(c['fixed_beta']) not in (int,float) or not math.isfinite(c['fixed_beta'])):raise ValueError('fixed beta')
    if c['fixed_beta'] is not None and c['fixed_beta'] not in c['beta_candidates']:raise ValueError('fixed beta')
    if c['fixed_beta'] is not None and c['random_beta']:raise ValueError('fixed and random beta controls are mutually exclusive')
    if c['xt_inner_lr_ratio']!=1.:raise ValueError('one-step head LR ratio is fixed at one')
    for key in ('ir_start_epoch','ir_cg_max_iterations','br_refresh_interval'):
        if type(c[key]) is not int or c[key]<1:raise ValueError(key)
    if c['ir_start_epoch']!=21 or c['ir_cg_max_iterations']>2:raise ValueError('IR v1 uses E21 and at most two CG iterations')
    if type(c['ir_kappa']) not in (int,float) or not math.isfinite(c['ir_kappa']) or not 0<=c['ir_kappa']<=1:raise ValueError('ir_kappa')
    if type(c['ir_cg_rtol']) not in (int,float) or not math.isfinite(c['ir_cg_rtol']) or c['ir_cg_rtol']<=0:raise ValueError('ir_cg_rtol')
    if c['implementation'] not in ('reference','cached'):raise ValueError('implementation')
    if c['ir_origin_scales']!='reuse_origin_used_scales':raise ValueError('IR origin scales')
    if c['ir_curvature'] not in ('ggn','zero') or c['ir_feature_anchor'] not in ('predictor','origin'):raise ValueError('IR ablation')
    if c['method'] in ('IR_EG','BR_IR_EG') and not c['unlabeled_head']:raise ValueError('IR requires U head supervision')
    return c

def make_row(name, response, seed=392005, *, pure_game=False):
    if type(pure_game) is not bool:raise ValueError('pure_game must be boolean')
    c=resolve(response)
    base='simultaneous' if c['method'] in ('DRIC','SIM','FR','CGD','TASK_PROJECT') else 'full_EG'
    row=native_row(name,dict(solver_mode=base,model_seed=seed,normalization_scales='reuse_origin_used_scales',
        labeled_encoder_grl_multiplier=float(c['encoder_adversary'])*c['encoder_multiplier']))
    row['response']=c
    if pure_game:
        row['pure_game']=True
        row['joint']['native_dr']=False
    return row


def validate_prepared_row(saved):
    """Read old non-IR rows without changing their recorded baseline fields."""
    expected=deepcopy(saved)
    if expected['response']['method'] not in ('IR_EG','BR_IR_EG'):
        for key in IR_DEFAULT_KEYS:
            expected['response'].setdefault(key,deepcopy(DEFAULTS[key]))
    row=make_row(expected['id'],expected['response'],expected['joint']['model_seed'],
                 pure_game=expected.get('pure_game',False))
    if row!=expected:raise ValueError('configuration differs from canonical prepared row')
    return row

def cadence(c):
    return c['dric_interval'] if c['method'] in ('DRIC','FR','CGD','TASK_PROJECT') else c['cf_interval'] if c['method']=='CF_EG' else c['xt_interval'] if c['method']=='XT_DANN' else 1

def schedule(c, epoch, accepted_before):
    if c['method'] in ('IR_EG','BR_IR_EG'):
        return dict(active=epoch>=c['ir_start_epoch'],strength=c['ir_kappa'],base='extragradient')
    method=c['method'];local=method in ('DRIC','FR','CGD','TASK_PROJECT')
    interval=cadence(c)
    active=epoch>=21 and (accepted_before+1)%interval==0
    ramp=min(1.,max(0.,(epoch-21)/(19. if local else 39.)))
    return dict(active=active, strength=ramp if local or method=='TR_EG' else 1.,
        base='simultaneous' if local or method=='SIM' else 'heun' if method=='RK2' else 'extragradient')

def stage_table(c):
    rows=[]
    for epoch in (1,20,21,40,41,60,61,79,80,90,91,160,161,200):
        start=(epoch-1)*222
        slots=[i+1 for i in range(222) if c['method'] not in ('EG','SIM','RK2')
            and schedule(c,epoch,start+i)['active'] and schedule(c,epoch,start+i)['strength']>0]
        initial=schedule(c,epoch,start)
        rows.append(dict(epoch=epoch,base=initial['base'],strength=initial['strength'],
            accepted_steps=[start+1,start+222],correction_steps_in_epoch=slots,
            correction_count=len(slots)))
    return rows
