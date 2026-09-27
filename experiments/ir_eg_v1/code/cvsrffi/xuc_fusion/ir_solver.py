"""Native joint implicit-response EG with complete accepted-step atomicity."""
from copy import deepcopy
from contextlib import contextmanager
import torch
from cvsrffi.game_tracking.solvers import GameSolver, StepResult, NonFiniteStep
from cvsrffi.game_tracking.state import TrainingState, RNGState, clone_buffers, restore_buffers
from .response_config import resolve
from .ir_types import build_layout
from .ir_head import capture_head_calls, head_gradient, ggn_matvec
from .ir_metric import build_response_metric
from .ir_cg import solve_response
from .ir_objective import build_objective_packet, graph_activity
from .br_history import BRHistory


class ResponseNumericalFailure(FloatingPointError):
    pass


def _validate_loss(loss,stage):
    if not torch.is_tensor(loss) or loss.numel()!=1 or not loss.requires_grad:
        raise ValueError(stage+': expected one differentiable scalar')
    if not torch.isfinite(loss).all():raise NonFiniteStep(stage+':loss')


def _copy(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value,dict): return {k:_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(_copy(v) for v in value)
    return deepcopy(value)


def _context(ctx):
    return {k:_copy(v) for k,v in vars(ctx).items() if not k.startswith('ir_') or k in ('ir_stage_signature','ir_diagnostics')}


def _restore_context(ctx,state):
    vars(ctx).clear();vars(ctx).update(_copy(state))


def _flat(values, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(values,parameters)])


class AttributeState:
    """Explicit adapter for the native legacy PrototypeMemoryBank."""
    def __init__(self,obj):self.obj=obj
    def state_dict(self):return _copy(vars(self.obj))
    def load_state_dict(self,state):vars(self.obj).clear();vars(self.obj).update(_copy(state))


class IREGSolver(GameSolver):
    def __init__(self,model,optimizer,config,**kwargs):
        config=resolve(config)
        if config['method'] not in ('IR_EG','BR_IR_EG'): raise ValueError('IR method required')
        kwargs.pop('mode',None)
        super().__init__(model,optimizer,mode='extragradient',**kwargs)
        if type(optimizer) is not torch.optim.AdamW or self.scaler is not None:
            raise ValueError('IR v1 requires FP32 AdamW without AMP')
        if self.predictor_lr_ratio!=1.:raise ValueError('IR predictor ratio must equal one')
        if self.nonfinite!='raise':raise ValueError('IR never skips a failed source batch')
        if any(p.dtype!=torch.float32 for p in self.parameters):raise ValueError('IR requires FP32')
        for group in optimizer.param_groups:
            if any(group.get(k,False) for k in ('amsgrad','maximize','differentiable','capturable','fused','foreach')):
                raise ValueError('unsupported AdamW variant')
        for state in optimizer.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if key in state and state[key].dtype!=torch.float32:raise ValueError('IR requires FP32 AdamW state')
        self.config=config
        self.layout=build_layout(model,optimizer)
        by_name=dict(zip(self.names,self.parameters))
        self.layout_parameters=tuple(by_name[name] for name in self.layout.names)
        self.phi=tuple(self.layout_parameters[i] for i in self.layout.phi_indices)
        self.psi=tuple(self.layout_parameters[i] for i in self.layout.psi_indices)
        self.history=BRHistory(self.layout,config['br_refresh_interval'])
        self.failure_injector=None

    def state_dict(self):
        state=super().state_dict()
        state['ir_config']=deepcopy(self.config);state['br_history']=self.history.state_dict()
        return state

    def load_state_dict(self,state):
        if state.get('ir_config')!=self.config:raise ValueError('IR checkpoint configuration mismatch')
        super().load_state_dict(state);self.history.load_state_dict(state['br_history'])
        if self.history.gradients is not None:
            self.history.gradients=tuple(None if g is None else g.to(p.device) for p,g in zip(self.layout_parameters,self.history.gradients))

    def _inject(self,stage):
        if self.failure_injector:self.failure_injector(stage)

    def _native_order(self, gradients):
        by_name=dict(zip(self.layout.names,gradients))
        return [by_name[name] for name in self.names]

    def _layout_order(self, gradients):
        by_name=dict(zip(self.names,gradients))
        return tuple(by_name[name] for name in self.layout.names)

    def _signature(self,ctx):
        explicit=getattr(ctx,'ir_stage_signature',None)
        if explicit is None:
            # Only discrete native phase boundaries, never route IDs or EMA values.
            explicit=(max(e for e in (1,21,41,61,80,91,161) if e<=ctx.epoch),
                      getattr(ctx,'calibration_version',None))
        return (self.layout.signature,explicit,tuple(p.requires_grad for p in self.layout_parameters))

    @contextmanager
    def _capture(self,ctx):
        with capture_head_calls(self.model.adv_head,self.layout.signature,str(self._signature(ctx))) as tape:
            ctx.ir_tape=tape;ctx.ir_call_indices={};ctx.ir_phi_predictor=self.phi
            try:yield tape
            finally:ctx.__dict__.pop('ir_tape',None)

    def reference_step(self,closure,ctx,**kwargs):return self.step(closure,ctx=ctx,implementation='reference',**kwargs)
    def cached_step(self,closure,ctx,**kwargs):return self.step(closure,ctx=ctx,implementation='cached',**kwargs)

    def step(self,closure,*,ctx=None,commit=None,external_stateful=(),implementation=None,**kwargs):
        if ctx is None:raise ValueError('IR requires an explicit materialized step context')
        if kwargs:raise ValueError('unsupported IR step options')
        external=tuple(obj if hasattr(obj,'state_dict') and hasattr(obj,'load_state_dict') else AttributeState(obj)
                       for obj in external_stateful)
        origin=TrainingState(self.model,self.optimizer,stateful=self.stateful+external)
        ctx0=_context(ctx);solver0=deepcopy(self.state_dict())
        external_flags=[(m,m.training) for obj in external_stateful if isinstance(obj,torch.nn.Module) for m in obj.modules()]
        count=[0];handle=self.model.register_forward_hook(lambda *unused:count.__setitem__(0,count[0]+1))
        try:
            active=ctx.epoch>=self.config['ir_start_epoch']
            br=self.config['method']=='BR_IR_EG' and active
            # This branch does not instantiate hooks on the head, masks, packet,
            # response state, or CG. It calls the original complete EG verbatim.
            if not active or (self.config['ir_kappa']==0 and not br):
                result=super().step(closure)
                result.telemetry=result.telemetry or {}
                result.telemetry.update(configured=True,eligible=active,executed=False,applied=False,
                    nonzero=False,fallback=False,effective_kappa=0.,full_backward_calls=2,
                    head_only_backward_calls=0,jvp_calls=0,vjp_calls=0,field_forward_calls=2,
                    cg_iterations=0,cg_status=None,implementation='native_full_EG')
            else:
                result=self._ir_step(closure,ctx,origin,br,implementation or self.config['implementation'])
            self._inject('after_formal')
            if commit:commit()
            self._inject('after_external_commit')
            if self.config['method']=='BR_IR_EG' and active:
                self.history.commit(self._layout_order(self.last_trace['corrector']),self._signature(ctx),self.steps,
                                    refreshed=result.telemetry['br_refreshed'])
            result.telemetry=result.telemetry or {}
            result.telemetry.update(model_forward_calls=count[0],formal_optimizer_commits=1,
                                    history_bytes=self.history.bytes,accepted_index=self.steps)
            return result
        except Exception:
            origin.restore();_restore_context(ctx,ctx0);self.load_state_dict(solver0)
            for module,flag in external_flags:module.training=flag
            raise
        finally:handle.remove()

    def _ir_step(self,closure,ctx,origin,br,implementation):
        if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
            raise ValueError('IR requires FP32 without autocast')
        if build_layout(self.model,self.optimizer).signature!=self.layout.signature:
            raise ValueError('IR parameter structure changed')
        self._measure_components=False;self._components={}
        self.optimizer.zero_grad(set_to_none=True)
        signature=self._signature(ctx);accepted_index=self.steps+1
        anchor_batch=None
        with self._capture(ctx) as origin_tape:
            loss=closure()
            _validate_loss(loss,'origin')
            activity=graph_activity(loss,self.psi)
            reason=self.history.refresh_reason(signature,accepted_index,activity) if br else 'full_IR'
            refreshed=reason is not None
            targets=self.layout_parameters if refreshed else self.phi
            active_targets=[p for p in targets if p.requires_grad]
            grads=torch.autograd.grad(loss,active_targets,allow_unused=True) if active_targets else ()
            by_id={id(p):g for p,g in zip(active_targets,grads)}
            raw=tuple(None if by_id.get(id(p)) is None else by_id[id(p)].detach().clone() for p in self.layout_parameters)
            h0=tuple(raw[i] for i in self.layout.phi_indices)
            if self.config['ir_feature_anchor']=='origin':anchor_batch=origin_tape.batch()
            loss0=float(loss.detach())
        if not torch.isfinite(loss.detach()) or any(g is not None and not torch.isfinite(g).all() for g in raw):
            raise NonFiniteStep('origin')
        del loss,grads,origin_tape
        self._finite_state('origin');self._inject('after_origin')
        origin_ctx=_context(ctx)
        buffers=clone_buffers(self.model);rng_after=RNGState.capture()
        first=raw if refreshed else self.history.predictor(activity,h0,signature,accepted_index)
        first=self._native_order(first)
        self._install(first)
        predictor_clip=self.last_clip_coefficient
        self.optimizer.step();self._finite_state('predictor');self._inject('after_virtual')
        phip=tuple(p.detach().clone() for p in self.phi)
        metric=build_response_metric(origin.optimizer_state,dict(clip_coefficient=predictor_clip,
                                   optimizer_after=deepcopy(self.optimizer.state_dict())),h0,self.layout)
        if metric.diagonal.device!=self.phi[0].device and metric.diagonal.numel()==0:
            from dataclasses import replace
            metric=replace(metric,diagonal=metric.diagonal.to(self.phi[0].device),
                           sqrt_diagonal=metric.sqrt_diagonal.to(self.phi[0].device),active_indices=metric.active_indices.to(self.phi[0].device))
        restore_buffers(self.model,origin.buffers);origin.rng.restore()
        _restore_context(ctx,origin_ctx)
        packet=None;cg=None;fallback=None;delta=torch.zeros_like(_flat(phip,self.phi));hp=None;eg_raw=None
        curvature_calls=0;energy=None
        fields=2;full_backwards=1+int(refreshed)
        try:
            with self._capture(ctx) as tape:
                native_loss=closure()
                _validate_loss(native_loss,'predictor')
                packet=build_objective_packet(ctx)
                if getattr(ctx,'ir_diagnostics',False):
                    active=[p for p in self.parameters if p.requires_grad]
                    diagnostic_grad=torch.autograd.grad(native_loss,active,allow_unused=True,retain_graph=True)
                    diagnostic_by_id={id(p):g for p,g in zip(active,diagnostic_grad)}
                    eg_raw=[None if diagnostic_by_id.get(id(p)) is None else diagnostic_by_id[id(p)].detach().clone()
                            for p in self.parameters]
                self._inject('after_capture')
                # The packet must have removed every real adv_head dependency.
                probe=tuple(p.detach().clone().requires_grad_(True) for p in phip)
                assembled=packet.assemble(probe)
                trainable_phi=tuple(p for p in self.phi if p.requires_grad)
                leaked=torch.autograd.grad(assembled,trainable_phi,allow_unused=True,retain_graph=True) if trainable_phi else ()
                if any(g is not None and bool(g.count_nonzero()) for g in leaked):
                    raise ValueError('unregistered native phi gradient leak')
                batch=anchor_batch or packet.head_batch
                hp=head_gradient(phip,batch)
                # Ownership is checked at predictor features regardless of the OR ablation.
                expected=hp if anchor_batch is None else head_gradient(phip,packet.head_batch)
                native_head=torch.autograd.grad(native_loss,trainable_phi,allow_unused=True,retain_graph=True) if trainable_phi else ()
                actual_by_id={id(p):g for p,g in zip(trainable_phi,native_head)}
                active_flat=torch.cat([torch.full((p.numel(),),p.requires_grad,device=p.device,dtype=torch.bool) for p in self.phi])
                actual_flat=_flat(tuple(actual_by_id.get(id(p)) for p in self.phi),self.phi)
                if not torch.allclose(actual_flat[active_flat],expected[active_flat],atol=1e-6,rtol=1e-5):
                    raise ValueError('native head field differs from registered CE')
                e=hp-_flat(h0,self.phi)
                self._inject('inside_cg')
                if self.config['ir_kappa']:
                    def operator(v):
                        nonlocal curvature_calls
                        if self.config['ir_curvature']=='zero':return torch.zeros_like(v)
                        curvature_calls+=1
                        return ggn_matvec(phip,batch,v)
                    try:
                        cg=solve_response(e,metric,operator,max_iterations=self.config['ir_cg_max_iterations'],rtol=self.config['ir_cg_rtol'])
                    except FloatingPointError as exc:
                        raise ResponseNumericalFailure(str(exc)) from exc
                    if cg.status in ('failed','numerical_failure','breakdown') or not torch.isfinite(cg.solution).all():
                        raise ResponseNumericalFailure(cg.reason or cg.status)
                    delta[metric.active_indices]=metric.sqrt_diagonal*cg.solution
                    au=cg.solution+metric.sqrt_diagonal*operator(delta)[metric.active_indices]
                    rhs=-metric.sqrt_diagonal*e[metric.active_indices]
                    energy=float(.5*torch.dot(cg.solution.double(),au.double())-torch.dot(rhs.double(),cg.solution.double()))
                response=_flat(phip,self.phi)+self.config['ir_kappa']*delta
                phi_response=[];offset=0
                for p in self.phi:
                    phi_response.append(response[offset:offset+p.numel()].reshape_as(p).detach().clone())
                    offset+=p.numel()
                if implementation=='cached':
                    leaves=tuple(p.requires_grad_(True) for p in phi_response)
                    final_loss=packet.assemble(leaves)
                    _validate_loss(final_loss,'cached_corrector')
                    targets=[p for p in self.psi if p.requires_grad]+list(leaves)
                    final=torch.autograd.grad(final_loss,targets,allow_unused=True)
                    mapped={id(p):g for p,g in zip(targets,final)}
                    mapped.update({id(p):g if p.requires_grad else None for p,g in zip(self.phi,final[-len(leaves):])})
                    gradients=[None if mapped.get(id(p)) is None else mapped[id(p)].detach().clone() for p in self.parameters]
                    loss1=float(final_loss.detach())
                    del final_loss,assembled,native_loss
                else:
                    del assembled,native_loss
                    packet.release();packet=None
                    # All graph-owning context callbacks and tape records go away
                    # before installing the response in the live parameters.
                    _restore_context(ctx,origin_ctx);tape.records.clear()
                    with torch.no_grad():
                        for p,value in zip(self.phi,phi_response):p.copy_(value)
                    restore_buffers(self.model,origin.buffers);origin.rng.restore()
                    loss1,gradients=self._evaluate(closure,'ir_corrector')
                    fields=3
            self._inject('after_corrector')
        except ResponseNumericalFailure as exc:
            fallback=str(exc)
            if packet:packet.release();packet=None
            # Same materialized batch/context and RNG, rebuilt ordinary EG.
            origin.restore();_restore_context(ctx,origin_ctx)
            self._install(first);self.optimizer.step()
            restore_buffers(self.model,origin.buffers);origin.rng.restore()
            loss1,gradients=self._evaluate(closure,'fallback_corrector')
            fields+=1
        finally:
            if packet:packet.release()
            for key in list(vars(ctx)):
                if key.startswith('ir_') and key not in ('ir_stage_signature','ir_diagnostics'):vars(ctx).pop(key,None)
        if any(g is not None and not torch.isfinite(g).all() for g in gradients):raise NonFiniteStep('corrector')
        replay_ctx=_context(ctx)
        origin.restore();_restore_context(ctx,origin_ctx)
        # Persist origin semantics, while retaining observable native traversal audit.
        for key in ('forward_calls','field_components','dr_field_divisors'):
            if key in replay_ctx:setattr(ctx,key,replay_ctx[key])
        attribution=None
        if eg_raw is not None:
            from .ir_telemetry import clip_attribution
            self._install(eg_raw)
            common=self.last_clip_coefficient
            origin.restore()
            attribution=clip_attribution(self.model,self.optimizer,self.parameters,eg_raw,gradients,
                max_grad_norm=self.max_grad_norm,common_coefficient=common)
        self._install(gradients);formal_clip=self.last_clip_coefficient
        self.optimizer.step();self._finite_state('formal')
        restore_buffers(self.model,buffers);rng_after.restore();self.steps+=1
        self.last_trace=dict(origin=first,corrector=gradients,phip=phip,h0=h0,hp=hp,delta=delta.detach(),
            metric=metric,head_batch=batch,formal_update=[p.detach()-v for p,v,_ in origin.parameters])
        norm=sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
        telemetry=dict(configured=True,eligible=True,executed=bool(self.config['ir_kappa']),
            applied=bool(self.config['ir_kappa']) and fallback is None,nonzero=bool(delta.count_nonzero()),
            fallback=fallback is not None,accepted=True,effective_kappa=self.config['ir_kappa'],implementation=implementation,
            cg_status=cg.status if cg else None,cg_iterations=cg.iterations if cg else 0,
            linear_relative_residual=cg.relative_residual if cg else None,
            quadratic_model_energy=energy,h0_norm=float(_flat(h0,self.phi).double().norm()),
            hp_norm=float(hp.double().norm()) if hp is not None else None,
            e_norm=float((hp-_flat(h0,self.phi)).double().norm()) if hp is not None else None,
            response_norm=float((self.config['ir_kappa']*delta).double().norm()),
            parameter_displacement_norm=sum(float((p.detach()-v).double().square().sum()) for p,v,_ in origin.parameters)**.5,
            predictor_clip_coefficient=predictor_clip,formal_clip_coefficient=formal_clip,
            full_backward_calls=full_backwards,field_forward_calls=fields,
            diagnostic_full_backward_calls=int(eg_raw is not None),clip_attribution=attribution,
            head_only_backward_calls=3+int(anchor_batch is not None)+int(not refreshed),
            jvp_calls=curvature_calls*len(batch.calls),vjp_calls=curvature_calls*len(batch.calls),
            curvature_operator_calls=curvature_calls,br_refreshed=refreshed,br_refresh_reason=reason,
            history_age=1 if br and not refreshed else None)
        return StepResult(loss0,loss1,norm,'extragradient',True,fields,self.config['method'],
                          failure_stage=fallback,telemetry=telemetry)
