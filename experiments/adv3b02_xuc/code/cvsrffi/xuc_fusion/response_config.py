"""Independent, prepared-only response-game experiments."""
from copy import deepcopy
from .joint_config import make_row as native_row

DEFAULTS=dict(method='CF_EG', encoder_adversary=True, encoder_multiplier=1., unlabeled_head=True, beta_candidates=[1.,.5,0.],
    margin_tolerance=.01, cf_interval=1, fixed_beta=None, random_beta=False,
    transport_gamma=.5, transport_rho=.01, unmatched_rotation=False,
    xt_mu=.1, xt_interval=20, xt_inner_lr_ratio=1., xt_exposure_control=False,
    dric_interval=4, identity_budget=.1, trust_ratio=.25, curvature_damping=.1,
    min_monotonicity=1e-4, residual_tolerance=1e-5, max_iterations=5,
    drift=True, identity_constraint=True, shuffle_drift=False, decompose_interval=1000, launch=False)

def resolve(values):
    if set(values)-set(DEFAULTS):raise ValueError('unknown response settings')
    c=deepcopy(DEFAULTS);c.update(values)
    if c['method'] not in ('CF_EG','TR_EG','XT_DANN','DRIC','EG','SIM','RK2','FR','CGD','TASK_PROJECT'):
        raise ValueError('unknown response method')
    if c['launch'] is not False:raise ValueError('prepared only')
    for k in ('cf_interval','xt_interval','dric_interval','max_iterations','decompose_interval'):
        if type(c[k]) is not int or c[k]<1:raise ValueError(k)
    if c['max_iterations']>5:raise ValueError('at most five local iterations')
    for k in ('encoder_adversary','unlabeled_head','random_beta','unmatched_rotation','xt_exposure_control','drift','identity_constraint','shuffle_drift'):
        if type(c[k]) is not bool:raise ValueError(k+' must be boolean')
    for k in ('encoder_multiplier','margin_tolerance','transport_rho','xt_mu','identity_budget','trust_ratio','curvature_damping','min_monotonicity','residual_tolerance'):
        if not isinstance(c[k],(int,float)) or not 0<=c[k]<float('inf'):raise ValueError(k)
    if not 0<=c['transport_gamma']<=1:raise ValueError('gamma')
    if c['beta_candidates']!=sorted(set(c['beta_candidates']),reverse=True) or c['beta_candidates'][-1]!=0 or any(not 0<=v<=1 for v in c['beta_candidates']):raise ValueError('descending beta candidates ending in zero')
    if c['fixed_beta'] is not None and c['fixed_beta'] not in c['beta_candidates']:raise ValueError('fixed beta')
    if c['xt_inner_lr_ratio']!=1.:raise ValueError('one-step head LR ratio is fixed at one')
    return c

def make_row(name, response, seed=392005):
    c=resolve(response)
    base='simultaneous' if c['method'] in ('DRIC','SIM','FR','CGD','TASK_PROJECT') else 'full_EG'
    row=native_row(name,dict(solver_mode=base,model_seed=seed,normalization_scales='reuse_origin_used_scales',
        labeled_encoder_grl_multiplier=float(c['encoder_adversary'])*c['encoder_multiplier']))
    row['response']=c
    return row

def schedule(c, epoch, accepted_before):
    method=c['method'];local=method in ('DRIC','FR','CGD','TASK_PROJECT')
    interval=c['dric_interval'] if local else c['cf_interval'] if method=='CF_EG' else c['xt_interval'] if method=='XT_DANN' else 1
    active=epoch>=21 and (accepted_before+1)%interval==0
    ramp=min(1.,max(0.,(epoch-21)/(19. if local else 39.)))
    return dict(active=active, strength=ramp if local or method=='TR_EG' else 1.,
        base='simultaneous' if local or method=='SIM' else 'heun' if method=='RK2' else 'extragradient')
