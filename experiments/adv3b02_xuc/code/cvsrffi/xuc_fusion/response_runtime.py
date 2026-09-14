"""One integration point, shared by formal runtime and bounded acceptance."""
import torch
from .response_config import schedule,cadence
from .response_context import monitor_views,risk_vector,passive,source_observation

def response_step(solver,objective,ctx,args,dr,monitor,completed):
    c=args.response;s=schedule(c,ctx.epoch,completed)
    monitor_index=completed//cadence(c)
    batch=monitor.batch(monitor_index,ctx.x.device)
    with passive(solver.model):views=monitor_views(batch,ctx,args)
    ctx.response_xt_diagnostics=[]
    def closure(on):
        xt=c['method']=='XT_DANN' and s['active']
        ctx.response_encoder_grl=0. if xt and not c['xt_exposure_control'] else float(on)*c['encoder_multiplier']
        ctx.response_xt=None
        if xt:
            head_ids={id(p) for p in solver.model.adv_head.parameters()}
            head_lr=next(g['lr'] for g in solver.optimizer.param_groups if any(id(p) in head_ids for p in g['params']))
            ctx.response_xt=dict(batch,step=completed//c['xt_interval'],lr=head_lr,mu=c['xt_mu'],
                lambda_encoder=float(ctx.weights['adv'])*float(on)*c['encoder_multiplier'],exposure_control=c['xt_exposure_control'])
        return objective(ctx)
    # TR uses the training batch for fit and physically disjoint L for monitor.
    selected=[i for i,identity in enumerate(batch['ids']) if identity not in set(ctx.sample_ids)]
    if c['method']=='TR_EG' and not selected:raise ValueError('no disjoint transport monitor samples')
    transport=(ctx.x,batch['x'][selected],batch['domain'][selected])
    local_views={'clean':ctx.x}
    if ctx.epoch>=args.sat_cons_start_epoch and ctx.satellite_mask_count:
        local_views[ctx.satellite_scenario]=ctx.satellite
    from SSDG.train_ssdg import rc4_tail_transition_scale
    u_weight=float(dr.args.rc4_lambda_domain)*rc4_tail_transition_scale(ctx.epoch,
        start_epoch=dr.args.rc4_tail_transition_start_epoch,ramp_epochs=dr.args.rc4_tail_transition_epochs,
        floor=dr.args.rc4_tail_transition_floor)
    if not c['unlabeled_head']:u_weight=0.
    def risk():return risk_vector(solver.model,views,batch['y'])
    def hp_monitor():
        probability=solver.model(ctx.dr_strong,return_aux=True)['tx_logits'].softmax(-1)
        route=ctx.dr_route
        true=probability.gather(1,route.pseudo[:,None]).squeeze(1)
        mass=(probability*route.candidate_mask).sum(1)
        def aggregate(values,mask):
            weights=route.weights[mask]
            return float((values[mask]*weights).sum()/weights.sum().clamp_min(1e-12)) if mask.any() else None
        return dict(H_pseudo_probability=aggregate(true,route.hard),P_set_mass=aggregate(mass,route.partial),
            hard_constraints=False,pseudo_labels_are_not_truth=True)
    risk.auxiliary=hp_monitor
    diagnostic_interval=args.joint['diagnostic_interval']
    observe=diagnostic_interval>0 and completed%diagnostic_interval==0
    observation_before=source_observation(solver.model,batch,views) if observe else None
    result=solver.response_step(closure,schedule=s,risk=risk,
        transport=transport,dric=(ctx,local_views,u_weight))
    result.telemetry=result.telemetry or {}
    if observe:
        risks,features,head_gradient=source_observation(solver.model,batch,views)
        r0,z0,h0=observation_before
        result.telemetry['passive_source_observation']=dict(risk_before=r0.tolist(),risk_after=risks.tolist(),
            representation_step_drift_norm=float((features-z0).norm()),
            L_head_gradient_change_norm=float((head_gradient-h0).norm()),
            L_head_gradient_norm_before=float(h0.norm()),L_head_gradient_norm_after=float(head_gradient.norm()),
            scope='whole_accepted_step_on_fixed_balanced_L_not_task_only_drift_or_recovery_gap',controls_training=False)
    result.telemetry.update(response_method=c['method'],schedule=s,
        labeled_monitor_ids=batch['ids'],labeled_monitor_samples=90,monitor_rotation_index=monitor_index,active_monitor_scenes=list(views),
        xt_fields=ctx.response_xt_diagnostics,encoder_adversary=c['encoder_adversary'],
        effective_adversarial_coefficients=dict(L_encoder=float(ctx.weights['adv'])*float(c['encoder_adversary'])*c['encoder_multiplier'],
            U_encoder=0.,L_head=float(ctx.weights['adv']),U_head=u_weight),
        unlabeled_encoder_grl=0.,head_LU_supervision=c['unlabeled_head'])
    return result
