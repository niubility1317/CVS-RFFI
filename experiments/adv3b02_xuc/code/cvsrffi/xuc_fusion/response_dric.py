"""DRIC-Lite in explicit signed, low-dimensional parameter coordinates."""
import itertools
import time
from contextvars import ContextVar
import torch
import torch.nn.functional as F
from cvsrffi.game_tracking.solvers import StepResult
from cvsrffi.game_tracking.state import TrainingState,RNGState,clone_buffers,restore_buffers
from .response_context import passive,risk_vector

_derivative_calls=ContextVar('response_derivative_calls',default=0)

def flat_grad(loss,params,graph=False):
    _derivative_calls.set(_derivative_calls.get()+1)
    gs=torch.autograd.grad(loss,params,create_graph=graph,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])

def basis(directions,maxdim=4):
    cols=[]
    for direction in directions:
        q=direction.detach().clone()
        for _ in range(2):
            for old in cols:q=q-torch.dot(old,q)*old
        if q.norm()>1e-10:cols.append(q/q.norm())
        if len(cols)==maxdim:break
    if not cols:return directions[0].new_zeros((directions[0].numel(),0))
    return torch.stack(cols,1)

def constrained_game(jac,g,n,A,eps,radius,*,mu=1e-4,tol=1e-5,max_iterations=5):
    """Small monotone VI via active-set enumeration and <=5 secular Newton steps.

    Constraint multipliers enter the encoder equation while the head response
    remains coupled. Nonconvergence is explicit; no independent post projection.
    """
    jac=jac.double();g=g.double();A=A.double();eps=eps.double()
    eye=torch.eye(len(g),device=g.device,dtype=g.dtype)
    eig=torch.linalg.eigvalsh((jac+jac.T)/2).min()
    damping=(mu-eig).clamp_min(0);J=jac+damping*eye
    if not torch.isfinite(J).all():raise FloatingPointError('nonfinite local game')
    unconstrained=torch.linalg.solve(J,-g)
    pad=F.pad(A,(0,len(g)-n))
    # At most n independent active halfspaces are needed in this <=4D space.
    for size in range(min(n,len(A))+1):
        for subset in itertools.combinations(range(len(A)),size):
            H=pad[list(subset)];bound=eps[list(subset)]
            for ball in (False,True):
                lam=0.;solution=None
                for iteration in range(max_iterations if ball else 1):
                    matrix=J.clone();matrix[:n,:n]+=lam*eye[:n,:n]
                    K=torch.cat((torch.cat((matrix,H.T),1),torch.cat((H,H.new_zeros(size,size)),1)),0)
                    rhs=torch.cat((-g,bound))
                    try:sol=torch.linalg.solve(K,rhs)
                    except torch.linalg.LinAlgError:break
                    z=sol[:len(g)];multipliers=sol[len(g):]
                    error=float(z[:n].norm())-radius
                    if not ball or abs(error)<=tol*max(radius,1e-8):solution=(z,multipliers,iteration+1);break
                    derivative_rhs=torch.cat((-torch.cat((z[:n],z.new_zeros(len(g)-n))),z.new_zeros(size)))
                    derivative=torch.linalg.solve(K,derivative_rhs)[:n]
                    slope=float(torch.dot(z[:n],derivative)/z[:n].norm().clamp_min(1e-15))
                    if slope>=0:break
                    lam=max(0.,lam-error/slope)
                if solution is None:continue
                z,multipliers,iterations=solution
                stationarity=J@z+g+H.T@multipliers
                stationarity[:n]+=lam*z[:n]
                residual=float(stationarity.norm()/g.norm().clamp_min(1e-12))
                feasible=bool((A@z[:n]<=eps+tol*max(radius,1e-8)).all()) and float(z[:n].norm())<=radius*(1+tol)+1e-12
                if feasible and bool((multipliers>=-tol).all()) and residual<=tol:
                    return z,dict(spectral_damping=float(damping),minimum_symmetric_eigenvalue=float(eig+damping),
                        residual=residual,iterations=iterations,active_constraints=list(subset),ball_multiplier=lam,
                        unconstrained_encoder_norm=float(unconstrained[:n].norm()))
    raise RuntimeError('DRIC_LOCAL_RESIDUAL_FAILURE: no feasible converged local solution')

def selected_parameters(model):
    selected=[(name,p) for name,p in model.named_parameters() if
        name.startswith(('id_backbone.t_proj.','id_backbone.f_proj.'))]
    if not selected:raise ValueError('DRIC requires explicit identity end projection blocks')
    return selected

def add_vector(params,delta):
    offset=0
    with torch.no_grad():
        for p in params:p.add_(delta[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()

def local_comparator(solver,ordinary,origin,theta,phi,domain_fields,lambda_e,strength,loss,buffers,rng):
    """Matched-tail AdamW-displacement CGD / one-sided response reference.

    At the ORIGINAL point, use ordinary optimizer displacement as the forcing
    and the two independently differentiated interaction blocks. No task
    counterfactual, risk constraint or drift forcing is used by CGD.
    """
    model=solver.model;optimizer=solver.optimizer;c=solver.config
    gradient_norm=sum(float(g.double().square().sum()) for g in ordinary if g is not None)**.5
    solver._install(ordinary);optimizer.step();baseline=TrainingState(model,optimizer)
    old={id(p):v for p,v,_ in origin.parameters}
    bt=torch.cat([(p.detach()-old[id(p)]).flatten() for p in theta])
    bh=torch.cat([(p.detach()-old[id(p)]).flatten() for p in phi])
    origin.restore()
    with passive(model):
        dl,dh=domain_fields();adv=flat_grad(-lambda_e*dl,theta,True);head=flat_grad(dh,phi,True)
        ti={id(p) for p in theta};hi={id(p) for p in phi}
        lt=next(g['lr'] for g in optimizer.param_groups if any(id(p) in ti for p in g['params']))
        lh=next(g['lr'] for g in optimizer.param_groups if any(id(p) in hi for p in g['params']))
        U=basis([bt,adv.detach()])*lt**.5;V=basis([bh,head.detach()])*lh**.5
        if not U.shape[1] or not V.shape[1]:
            baseline.restore();solver.steps+=1
            return StepResult(loss,None,gradient_norm,c['method'],True,1,'degenerate_ordinary',telemetry={'degenerate':True})
        B=torch.stack([V.T@flat_grad(a,phi) for a in U.T@adv],0)
        C=torch.stack([U.T@flat_grad(h,theta) for h in V.T@head],0)
        ut=U.T@bt/lt;vh=V.T@bh/lh
        if c['method']=='CGD':
            J=torch.cat((torch.cat((torch.eye(len(ut),device=bt.device),B),1),torch.cat((C,torch.eye(len(vh),device=bt.device)),1)),0)
            z=torch.linalg.solve(J.double(),torch.cat((ut,vh)).double()).to(bt)
            u,v=z[:len(ut)],z[len(ut):]
            residual=float((J@z-torch.cat((ut,vh))).norm())
        else:
            # Response tracking is explicitly one-sided and includes C times
            # the encoder displacement, so it detects the zero-head-gap toy.
            u=ut;v=vh-C@ut;residual=0.
        dt=(U@u-bt)*strength;dp=(V@v-bh)*strength
    baseline.restore();add_vector(theta,dt);add_vector(phi,dp)
    solver._finite_state('local_comparator');restore_buffers(model,buffers);rng.restore();solver.steps+=1
    return StepResult(loss,None,gradient_norm,c['method'],True,1,'ordinary_point_local_competitive' if c['method']=='CGD' else 'one_sided_response_tracking',telemetry=dict(
        local_scope='matched_identity_tail_and_domain_head',task_counterfactual=False,risk_constraint=False,
        B_norm=float(B.norm()),C_norm=float(C.norm()),residual=residual,optimizer_commits=1,
        moment_policy='one_ordinary_AdamW_commit_plus_direct_local_correction',literature_full_dimensional_reproduction=False))

def dric_step(solver,closure,context,strength):
    model=solver.model;optimizer=solver.optimizer;c=solver.config;origin=solver.setup();started=time.perf_counter()
    derivative_before=_derivative_calls.get()
    if context is None:raise ValueError('DRIC signed field context required')
    ctx,views,alpha_u=context
    named=selected_parameters(model);theta=[p for _,p in named];phi=list(model.adv_head.parameters())
    theta_ids={id(p) for p in theta};phi_ids={id(p) for p in phi}
    alpha_l=float(ctx.weights['adv']);lambda_e=alpha_l*float(c['encoder_adversary'])*c['encoder_multiplier']
    train_flags={m:m.training for m in model.modules()}
    from .response_replay import LocalReplay
    replay=LocalReplay(model,theta+phi)
    def local_risk():
        with replay.session('risk'):return risk_vector(model,views,ctx.y)
    def domain_fields():
        flags={m:m.training for m in model.modules()};rng_here=RNGState.capture()
        old_explicit=getattr(model,'response_explicit_domain_field',False)
        try:
            for m,flag in train_flags.items():m.training=flag
            origin.rng.restore();model.response_explicit_domain_field=True
            with replay.session('domain'):
                labeled=model(ctx.x,y_tx=ctx.y,return_aux=True,domain_labels=ctx.domain)
                unlabeled=model(ctx.dr_strong,return_aux=True,domain_labels=ctx.dr_d)
            dl=F.cross_entropy(labeled['adv_dom_logits'],ctx.domain)
            du=F.cross_entropy(unlabeled['adv_dom_logits'],ctx.dr_d)
            return dl,alpha_l*dl+alpha_u*du
        finally:
            model.response_explicit_domain_field=old_explicit;rng_here.restore()
            for m,flag in flags.items():m.training=flag
    try:
        loss,ordinary=solver._evaluate(lambda:closure(c['encoder_adversary']),'dric_origin')
        buffers,rng=clone_buffers(model),RNGState.capture()
        if c['method'] in ('FR','CGD'):
            return local_comparator(solver,ordinary,origin,theta,phi,domain_fields,
                alpha_l*float(c['encoder_adversary'])*c['encoder_multiplier'],strength,loss,buffers,rng)
        solver._install(ordinary);optimizer.step();ordinary_state=TrainingState(model,optimizer)
        origin.restore()
        decompose=(solver.steps+1)%c['decompose_interval']==0
        _,taskgradient=solver._evaluate(lambda:closure(False),'dric_task',retain_graph=decompose)
        component_gradients={}
        if decompose:
            for name,value in ctx.response_task_components.items():
                component_gradients[name]=[None if g is None else g.detach().clone() for g in
                    torch.autograd.grad(value,solver.parameters,retain_graph=True,allow_unused=True)]
        solver._install(taskgradient);optimizer.step();taskstate=TrainingState(model,optimizer)
        old={id(p):v for p,v,_ in origin.parameters}
        b=torch.cat([(p.detach()-old[id(p)]).reshape(-1) for p in theta])
        task_values={id(p):v for p,v,_ in taskstate.parameters}
        # Lite fixes all other parameters at their ordinary updated values.
        # Both old-tail risk and task-tail risk use THAT SAME background. This
        # preserves the early backbone's normal DANN and gives a local budget.
        ordinary_state.restore()
        with torch.no_grad():
            for p in theta:p.copy_(old[id(p)])
            for p in phi:p.copy_(old[id(p)])
        with passive(model):
            oldrisk=local_risk().detach()
            _,dh=domain_fields();head0=flat_grad(dh,phi).detach()
        with torch.no_grad():
            for p in theta:p.copy_(task_values[id(p)])
        with passive(model):
            risks=local_risk();reference=risks.detach()
            riskgrads=[flat_grad(r,theta).detach() for r in risks]
            dl,dh=domain_fields()
            adv=flat_grad(-lambda_e*dl,theta,True);head=flat_grad(dh,phi,True)
            drift=head.detach()-head0
            if not c['drift'] or c['method']=='CGD':drift=drift*0
            if c['shuffle_drift']:drift=drift.roll(1)
            forcing=head0+drift
            head_adv=flat_grad(-lambda_e*dl,phi,True)
            response_direction=flat_grad(torch.dot(head_adv,-forcing.detach()),theta).detach()
            task_coupling=flat_grad(torch.dot(flat_grad(dh,theta,True),b),phi).detach()
            risk_order=reference.argsort(descending=True).tolist()
            U=basis([adv.detach(),response_direction,b]+[riskgrads[i] for i in risk_order])
            V=basis([forcing,drift,task_coupling,head0])
            degenerate=U.shape[1]==0 or V.shape[1]==0 or lambda_e==0
            if degenerate:
                u=b*0;v=head0*0;stats=dict(degenerate=True,residual=0.,iterations=0,unconstrained_encoder_norm=0.)
            else:
                # Isotropic positive metric in whitened coordinates. This is
                # explicit; it is not an undocumented diagonal Adam preconditioner.
                lr_theta=next(g['lr'] for g in optimizer.param_groups if any(id(p) in theta_ids for p in g['params']))
                lr_head=next(g['lr'] for g in optimizer.param_groups if any(id(p) in phi_ids for p in g['params']))
                U=U*lr_theta**.5;V=V*lr_head**.5
                a=U.T@adv;h=V.T@forcing
                B=torch.stack([V.T@flat_grad(value,phi) for value in a],0)
                C=torch.stack([U.T@flat_grad(value,theta) for value in V.T@head],0)
                # PSD empirical outer-product curvature plus positive proximal
                # metric. No inversion of an indefinite network Hessian.
                projected=V.T@head.detach();fisher=torch.outer(projected,projected)
                Q=torch.eye(V.shape[1],device=b.device)+fisher+c['curvature_damping']*fisher.diag().mean().clamp_min(1e-8)*torch.eye(V.shape[1],device=b.device)
                J=torch.cat((torch.cat((torch.eye(U.shape[1],device=b.device),B),1),torch.cat((C,Q),1)),0).detach()
                budget=c['identity_budget']*(oldrisk-reference).clamp_min(0)
                A=torch.stack(riskgrads)@U
                if not c['identity_constraint']:A=A[:0];budget=budget[:0]
                radius=c['trust_ratio']*float(b.norm())/lr_theta**.5
                if c['method']=='FR':
                    # Response tracking comparator retains current adversarial
                    # step; head response includes task drift, no bilateral Bv.
                    J[:U.shape[1],U.shape[1]:]=0
                if c['method']=='TASK_PROJECT':
                    J[:U.shape[1],U.shape[1]:]=0;J[U.shape[1]:,:U.shape[1]]=0
                    task_by_id={id(p):g for p,g in zip(solver.parameters,taskgradient)}
                    task_theta=torch.cat([(torch.zeros_like(p) if task_by_id[id(p)] is None else task_by_id[id(p)]).reshape(-1) for p in theta])
                    A=(task_theta@U)[None,:];budget=b.new_zeros(1)
                z,stats=constrained_game(J,torch.cat((a,h)).detach(),U.shape[1],A,budget,radius,
                    mu=c['min_monotonicity'],tol=c['residual_tolerance'],max_iterations=c['max_iterations'])
                u=U@z[:U.shape[1]].to(U);v=V@z[U.shape[1]:].to(V)
                stats['unconstrained_encoder_norm']*=lr_theta**.5
                stats.update(theta_basis_dim=U.shape[1],head_basis_dim=V.shape[1],B_norm=float(B.norm()),C_norm=float(C.norm()),
                    asymmetric_defect=float((C+B.T).norm()),metric='isotropic_current_lr_whitened',curvature='PSD_projected_gradient_outer_product')
        ordinary_state.restore()
        # One ordinary AdamW moment/decay commit defines the background. The
        # counterfactual moments were read virtually, never mixed or committed.
        # Warmup interpolates actual displacements to the fully solved candidate;
        # only strength=1 claims the candidate's linear local budget.
        with torch.no_grad():
            if not degenerate or lambda_e==0:
                for p in theta:p.lerp_(task_values[id(p)],strength)
            if not degenerate:
                for p in phi:p.lerp_(old[id(p)],strength)
        add_vector(theta,u*strength)
        if not degenerate:add_vector(phi,v*strength)
        with passive(model):
            actual=local_risk().detach()
            _,response_head=domain_fields();tracking=flat_grad(response_head,phi).detach()
        decomposition={}
        if decompose:
            finalstate=TrainingState(model,optimizer)
            for name,gradients in component_gradients.items():
                origin.restore();solver._install(gradients);optimizer.step()
                component_theta=[p.detach().clone() for p in theta]
                ordinary_state.restore()
                with torch.no_grad():
                    for p,value in zip(theta,component_theta):p.copy_(value)
                    for p in phi:p.copy_(old[id(p)])
                with passive(model):
                    _,component_head=domain_fields();delta=flat_grad(component_head,phi).detach()-head0
                decomposition[name]=float(delta.norm())
            finalstate.restore()
        committed_u=torch.cat([(p.detach()-task_values[id(p)]).flatten() for p in theta])
        ordinary_values={id(p):value for p,value,_ in ordinary_state.parameters}
        ordinary_difference=sum(float((p.detach()-ordinary_values[id(p)]).double().square().sum()) for p in model.parameters())**.5
        nonselected_error=max((float((p.detach()-ordinary_values[id(p)]).abs().max()) for p in model.parameters() if id(p) not in theta_ids|phi_ids),default=0.)
        predicted=torch.stack(riskgrads)@committed_u;damage=actual-reference
        budget=c['identity_budget']*(oldrisk-reference).clamp_min(0)
        stats.update(task_displacement=float(b.norm()),extra_identity_displacement=float((u*strength).norm()),
            deviation_from_ordinary=ordinary_difference,nonselected_parameter_error=nonselected_error,
            committed_displacement_from_task=float(committed_u.norm()),
            local_budget_background='ordinary_updated_nonselected_parameters',
            full_linear_budget_applies=strength==1 and not degenerate,
            drift_norm=float(drift.norm()),head_response_displacement=float((v*strength).norm()),
            tracking_gradient_norm=float(tracking.norm()),tracking_relative_residual=float(tracking.norm()/head0.norm().clamp_min(1e-12)),
            component_drift_norms=decomposition,component_drift_definition='separate_same_state_virtual_AdamW_not_additive',
            identity_risk_damage=damage.tolist(),identity_budget=budget.tolist(),
            risk_violation_rate=float((damage>budget+1e-7).float().mean()),
            risk_violation_worst=float((damage-budget).clamp_min(0).max()),
            risk_linearization_error=(damage-predicted).tolist(),
            retained_adversarial_ratio=float(u.norm())/max(stats['unconstrained_encoder_norm'],1e-12),
            selected_parameters=[n for n,_ in named],optimizer_commits=1,
            local_autograd_calls=_derivative_calls.get()-derivative_before,
            local_leaf_executions=replay.executed,local_leaf_replays=replay.replayed,
            main_field_backward_calls=2,component_backward_calls=len(component_gradients),
            explicit_signed_training_field=True,local_encoder_strength=strength,
            moment_policy='one_ordinary_AdamW_state_virtual_task_state_never_committed',elapsed_seconds=time.perf_counter()-started)
        solver._finite_state('dric_formal');restore_buffers(model,buffers);rng.restore();solver.steps+=1
        gradient_norm=sum(float(g.double().square().sum()) for g in ordinary if g is not None)**.5
        return StepResult(loss,None,gradient_norm,c['method'],True,2,'direct_local_game_displacement',telemetry=stats)
    except Exception:
        origin.restore();raise
