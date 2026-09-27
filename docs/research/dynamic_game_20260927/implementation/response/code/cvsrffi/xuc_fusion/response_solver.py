"""Transactional response solvers: raw gradients first, one AdamW commit."""
import time
import torch
import torch.nn.functional as F
from cvsrffi.game_tracking.solvers import GameSolver,StepResult,NonFiniteStep
from cvsrffi.game_tracking.state import TrainingState,RNGState,clone_buffers,restore_buffers
from .response_fields import procrustes
from .response_context import passive,actual_head_input

class ResponseSolver(GameSolver):
    def __init__(self,model,optimizer,config,**kw):
        super().__init__(model,optimizer,**kw);self.config=config

    def response_step(self,closure,*,schedule,risk=None,transport=None,dric=None):
        self.mode=schedule['base'];method=self.config['method']
        if not schedule['active'] or schedule['strength']==0 or method in ('EG','SIM','RK2','XT_DANN'):
            result=super().step(lambda:closure(self.config['encoder_adversary']))
            if method in ('DRIC','FR','CGD','TASK_PROJECT'):
                result.telemetry=result.telemetry or {}
                result.telemetry.update(extra_identity_displacement=0.,correction_active=False)
            return result
        if method in ('DRIC','FR','CGD','TASK_PROJECT'):
            from .response_dric import dric_step
            return dric_step(self,closure,dric,schedule['strength'])
        if method=='CF_EG':return self.cf_step(closure,risk)
        if method=='TR_EG':return self.tr_step(closure,transport,schedule['strength'])
        raise ValueError(method)

    def setup(self):
        self._measure_components=False;self._components={}
        return TrainingState(self.model,self.optimizer, stateful=self.stateful)

    def predictor(self,gradients):
        self._install(gradients);lrs=[g['lr'] for g in self.optimizer.param_groups]
        try:
            for group,lr in zip(self.optimizer.param_groups,lrs):group['lr']=lr*self.predictor_lr_ratio
            self.optimizer.step();self._finite_state('virtual')
        finally:
            for group,lr in zip(self.optimizer.param_groups,lrs):group['lr']=lr

    def cf_step(self,closure,risk):
        start=time.perf_counter();origin=self.setup();c=self.config
        if risk is None:raise ValueError('CF requires legal L monitoring')
        try:
            fields=[];losses=[];origin_buffers=origin_rng=None
            for on in (c['encoder_adversary'],False):
                origin.restore()
                loss,first=self._evaluate(lambda:closure(on),'cf_origin')
                if origin_buffers is None:origin_buffers=clone_buffers(self.model);origin_rng=RNGState.capture()
                self.predictor(first);restore_buffers(self.model,origin.buffers);origin.rng.restore()
                loss2,second=self._evaluate(lambda:closure(on),'cf_corrector')
                fields.append(second);losses.append((loss,loss2))
            g1,g0=fields
            def candidate(beta):
                origin.restore()
                gradients=[None if a is None and b is None else
                    (torch.zeros_like(p) if b is None else b)+beta*((torch.zeros_like(p) if a is None else a)-(torch.zeros_like(p) if b is None else b))
                    for p,a,b in zip(self.parameters,g1,g0)]
                self._install(gradients);self.optimizer.step();self._finite_state('cf_candidate')
                # Compare the exact model state that will be committed, including
                # the single origin forward's running statistics.
                restore_buffers(self.model,origin_buffers)
                with passive(self.model),torch.no_grad():
                    r=risk().detach().clone()
                    auxiliary=risk.auxiliary() if hasattr(risk,'auxiliary') else None
                if not torch.isfinite(r).all():raise NonFiniteStep('cf:risk')
                return gradients,r,auxiliary
            _,r0,aux0=candidate(0.);task=[p.detach().clone() for p in self.parameters]
            tested=[];selected=None
            choices=c['beta_candidates']
            if c['fixed_beta'] is not None:choices=[c['fixed_beta']]
            elif c['random_beta']:
                gen=torch.Generator().manual_seed(60103+self.steps)
                choices=[choices[int(torch.randint(len(choices),(),generator=gen))]]
            for beta in choices:
                gradients,r,aux=candidate(beta);damage=r-r0
                feasible=bool((damage<=c['margin_tolerance']+1e-7).all())
                tested.append(dict(beta=beta,feasible=feasible,group_damage=damage.tolist(),HP_monitor=aux))
                if feasible or c['fixed_beta'] is not None or c['random_beta']:
                    selected=beta;break
            if selected is None:raise RuntimeError('task counterfactual must be feasible')
            increment=sum(float((p.detach()-q).double().square().sum()) for p,q in zip(self.parameters,task))**.5
            restore_buffers(self.model,origin_buffers);origin_rng.restore();self.steps+=1
            norm=sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
            delta=sum(float(((torch.zeros_like(p) if a is None else a)-(torch.zeros_like(p) if b is None else b)).double().square().sum()) for p,a,b in zip(self.parameters,g1,g0))**.5
            return StepResult(losses[0][0],losses[0][1],norm,'CF_EG',True,4,'counterfactual_adamw',telemetry=dict(
                beta=selected,delta_gradient_norm=delta,actual_game_increment=increment,candidates=tested,
                risk_reference=r0.tolist(),HP_reference=aux0,optimizer_commits=1,virtual_adamw_updates=3+len(tested),elapsed_seconds=time.perf_counter()-start))
        except Exception:
            origin.restore();raise

    def tr_step(self,closure,transport,strength):
        origin=self.setup();start=time.perf_counter()
        if transport is None:raise ValueError('paired fit/monitor samples required')
        xfit,xmonitor,domain=transport
        try:
            with passive(self.model),torch.no_grad():
                z0=actual_head_input(self.model,xfit).detach();m0=actual_head_input(self.model,xmonitor).detach()
            loss,first=self._evaluate(lambda:closure(self.config['encoder_adversary']),'tr_origin')
            buffers,rng=clone_buffers(self.model),RNGState.capture()
            self.predictor(first);restore_buffers(self.model,origin.buffers);origin.rng.restore()
            with passive(self.model),torch.no_grad():
                zp=actual_head_input(self.model,xfit).detach();mp=actual_head_input(self.model,xmonitor).detach()
                if self.config['unmatched_rotation']:zp=zp.roll(1,0)
                rotation=procrustes(z0,zp,self.config['transport_rho'])
                first_linear=next(m for m in self.model.adv_head.modules() if isinstance(m,torch.nn.Linear))
                old=first_linear.weight.detach().clone();gamma=self.config['transport_gamma']*strength
                before=F.cross_entropy(self.model.adv_head(mp),domain)
                first_linear.weight.copy_((1-gamma)*old+gamma*old@rotation.T)
                after=F.cross_entropy(self.model.adv_head(mp),domain)
                explained=1-(mp-m0@rotation.T).square().sum()/(mp-m0).square().sum().clamp_min(1e-12)
            loss2,second=self._evaluate(lambda:closure(self.config['encoder_adversary']),'tr_corrector')
            origin.restore();self._install(second);self.optimizer.step();self._finite_state('tr_formal')
            restore_buffers(self.model,buffers);rng.restore();self.steps+=1
            norm=sum(float(g.double().square().sum()) for g in second if g is not None)**.5
            return StepResult(loss,loss2,norm,'TR_EG',True,2,'predicted_head_transport',telemetry=dict(
                gamma=gamma,rotation_explained_monitor=float(explained),transport_ce_gain_monitor=float(before-after),
                rotation_determinant=float(torch.linalg.det(rotation)),monitor_disjoint=True,
                optimizer_commits=1,svd_calls=1,extra_model_forwards=4,elapsed_seconds=time.perf_counter()-start))
        except Exception:
            origin.restore();raise
