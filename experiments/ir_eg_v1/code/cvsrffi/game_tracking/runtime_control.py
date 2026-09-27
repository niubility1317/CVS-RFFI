"""Independent source game/capability clocks and resumable V2 calibration."""
from copy import deepcopy
from dataclasses import asdict
import math
import time

import numpy as np
import torch

from .audit_evidence import make_evidence_v2, unavailable
from .capability import CapabilityConfig, evaluate_capability_v2
from .controller import ControllerConfig, GameControllerV2
from .curriculum import CurriculumConfigV2, CapabilityCurriculumV2
from .source_audit import isolated_rng
from .step_context import satellite_policy, fixed_satellite_policy


def game_audit_v2(*args, **kwargs):
    from .runtime_audit_v2 import game_audit_v2 as run
    return run(*args, **kwargs)


def capability_audit_v2(model,source,indexes,args,*,step,version,policy_level,epoch=1):
    """Fit once on L_s fit containers; test same monitor on two policies."""
    from .runtime import extract
    from cvsrffi.eval import apply_sat_channel_for_scenario
    started=time.perf_counter()
    adaptive=args.game_curriculum=='capability'
    current_policy=satellite_policy(policy_level) if adaptive else fixed_satellite_policy(epoch)
    next_policy=satellite_policy(min(1.,policy_level+.1)) if adaptive else deepcopy(current_policy)
    with isolated_rng(2718):
        owned=deepcopy(model).eval()
        fit=extract(owned,source.train,indexes['capability_fit'],args.eval_batch_size)
        mon=extract(owned,source.train,indexes['capability_monitor'],args.eval_batch_size)
        views={}
        device=mon['x'].device
        # Common random numbers isolate the policy difference. Channels retain
        # their original physical ranges; only frequency and mixture change.
        gen=torch.Generator(device=device).manual_seed(82718)
        selection=torch.rand(len(mon['x']),device=device,generator=gen)
        mixture=torch.rand(len(mon['x']),device=device,generator=gen)
        channel_views=[]
        with torch.no_grad():
            for i,scene in enumerate(satellite_policy(0.)['scenarios']):
                channel_gen=torch.Generator(device=device).manual_seed(92718+i)
                channel_views.append(apply_sat_channel_for_scenario(mon['x'],scene,args,
                                     gen=channel_gen,return_meta=False)[0])
            for name,policy in [('current',current_policy),('next',next_policy)]:
                boundary=torch.tensor(policy['weights'],device=device).cumsum(0)
                category=torch.searchsorted(boundary,mixture).clamp_max(2)
                x=mon['x'].clone()
                for i,channel in enumerate(channel_views):
                    mask=(selection<policy['probability']) & (category==i)
                    x[mask]=channel[mask]
                zs=[]
                for part in x.split(args.eval_batch_size):
                    zs.append(owned(part,return_aux=True)['z_id'].float().detach())
                views[name]=torch.cat(zs)
        result=evaluate_capability_v2(fit['z'],fit['y'],fit['rx'],mon['z'],mon['y'],mon['rx'],
            fit_groups=fit['groups'],monitor_groups=mon['groups'],monitor_views=views,
            policy_level=policy_level,config=CapabilityConfig(steps=args.game_probe_steps,
                lr=args.game_capability_lr),step=step,encoder_version=version,
            independence_verified=True)
        result.update(current_policy=current_policy,next_policy=next_policy,
                      policy_kind='adaptive' if adaptive else 'fixed',
                      next_policy_role='adaptive_level_plus_0.1' if adaptive else 'fixed_policy_no_adaptive_candidate',
                      source_role='L_s',readout_head_shared=True,
                      elapsed_seconds=time.perf_counter()-started)
    return result


def _valid_capabilities(observations):
    fields=('identity','margin','next_identity','next_margin','next_worst_tx')
    return [m for m in observations if m.get('valid') and m.get('identity_valid') and
            not m.get('collapsed',True) and all(isinstance(m.get(k),(int,float)) and
            math.isfinite(m[k]) for k in fields)]


def calibrate_capability_v2(observations,args):
    valid=_valid_capabilities(observations)
    if len(valid)<3:return None
    q=lambda key:float(np.quantile([m[key] for m in valid],.25))
    chance=1./args.num_classes
    identity=max(chance+.05,min(.95,q('identity')*.9))
    next_identity=max(chance+.05,min(.95,q('next_identity')*.9))
    freshness=max(1,min(10,args.game_capability_interval//10))
    config=CurriculumConfigV2(identity_enter=identity,identity_exit=identity-.05,
        margin_enter=q('margin')-.01,margin_exit=q('margin')-.06,
        next_identity_enter=next_identity,next_identity_exit=next_identity-.05,
        next_margin_enter=q('next_margin')-.01,
        next_worst_tx_enter=max(chance,q('next_worst_tx')*.9),
        cooldown_steps=args.game_capability_interval,max_age_steps=freshness,max_version_lag=freshness,
        calibration_id='source_capability_three_valid_observations_v2')
    return CapabilityCurriculumV2(config)


def calibrate_game_v2(observations,capabilities,args):
    valid=[m['lag'] for m in observations if m.get('data_valid') and m.get('coverage_valid') and
        m.get('lag',{}).get('quality_pass') and m['lag'].get('control_ready') and
        m['lag'].get('status') in ('RELIABLE_HIGH_GAP','RELIABLE_LOW_GAP') and
        isinstance(m['lag'].get('gap_normalized'),(float,int)) and math.isfinite(m['lag']['gap_normalized'])]
    caps=_valid_capabilities(capabilities)
    if len(valid)<3 or len(caps)<3:return None
    values=[m['gap_normalized'] for m in valid]
    threshold=max(.01,float(np.quantile(values,.75)+np.std(values)))
    freshness=max(1,min(10,args.game_audit_interval//10))
    config=ControllerConfig(lag_enter=threshold,lag_exit=threshold*.5,
        identity_min=max(1./args.num_classes,float(np.quantile([m['identity'] for m in caps],.25))*.9),
        margin_min=float(np.quantile([m['margin'] for m in caps],.25))-.05,
        catchup_steps=args.game_max_extra_head,cooldown_steps=args.game_audit_interval,
        audit_interval=args.game_audit_interval,sparse_audit_interval=args.game_audit_interval,
        max_age_steps=freshness,max_version_lag=freshness,
        calibration_id='source_trusted_lag_three_valid_observations_v2')
    return GameControllerV2(config)


class V2Coordinator:
    def __init__(self,model,source,args,proto,auditor,budget,indexes):
        self.model,self.source,self.args,self.proto=model,source,args,proto
        self.auditor,self.budget,self.indexes=auditor,budget,indexes
        self.controller=self.curriculum=None
        self.next_game_audit_step=self.next_capability_step=0
        self.game_observations=[];self.capability_observations=[]
        self.last_metrics={};self.capability_metrics={};self.sets=None
        self.capability_observed=False;self.game_observed=False

    def observe(self,ctx,*,step,version,epoch,augmentor=None):
        self.sets=None;self.capability_observed=self.game_observed=False
        # Capability owns its own cadence, even when lag fitting failed.
        if step>=self.next_capability_step:
            self.next_capability_step=step+self.args.game_capability_interval
            forecast=max(.001,self.capability_metrics.get('elapsed_seconds',1.) or 1.)
            if self.budget.can_afford(step,audit_seconds=forecast)[0]:
                self.capability_metrics=capability_audit_v2(self.model,self.source,self.indexes,self.args,
                    step=step,version=version,epoch=epoch,
                    policy_level=self.curriculum.level if self.curriculum else 0.)
                self.budget.commit(step,audit_seconds=self.capability_metrics['elapsed_seconds'])
            else:
                self.capability_metrics=dict(schema='game_capability_v2',step=step,encoder_version=version,
                    valid=False,identity_valid=False,reason='capability_audit_budget_exhausted',elapsed_seconds=0.)
            self.capability_observed=True
            self.capability_observations.append(deepcopy(self.capability_metrics))
            if self.curriculum is None and step>=self.args.game_source_calibration_steps:
                self.curriculum=calibrate_capability_v2(self.capability_observations,self.args)
        if step>=self.next_game_audit_step:
            self.next_game_audit_step=step+self.args.game_audit_interval
            forecast=max(.001,self.last_metrics.get('elapsed_seconds',1.) or 1.)
            if self.budget.can_afford(step,audit_seconds=forecast)[0]:
                self.last_metrics,self.sets=game_audit_v2(self.model,self.source,self.indexes,self.auditor,
                    self.args,ctx,step=step,version=version,epoch=epoch,proto=self.proto,augmentor=augmentor)
                self.budget.commit(step,audit_seconds=self.last_metrics['elapsed_seconds'])
            else:
                self.last_metrics=make_evidence_v2(observation_id=f'budget:{step}:{version}',step=step,
                    encoder_version=version,data_valid=True,coverage_valid=False,
                    lag=unavailable('game_audit_budget_exhausted'))
                self.last_metrics['elapsed_seconds']=0.
            self.game_observed=True
            self.last_metrics['capability']=deepcopy(self.capability_metrics)
            self.game_observations.append(deepcopy(self.last_metrics))
        # Attach only the actual capability timestamp; never relabel old data.
        if self.last_metrics:self.last_metrics['capability']=deepcopy(self.capability_metrics)
        if self.controller is None and step>=self.args.game_source_calibration_steps:
            self.controller=calibrate_game_v2(self.game_observations,self.capability_observations,self.args)
        return self.last_metrics

    def invalidate_game_evidence(self,reason='effective_satellite_policy_changed'):
        if self.last_metrics:
            self.last_metrics['lag']=unavailable(reason)
            self.last_metrics['gradient']=unavailable(reason)
        self.sets=None

    def state_dict(self):
        return deepcopy(dict(schema='game_coordinator_v2',next_game_audit_step=self.next_game_audit_step,
            next_capability_step=self.next_capability_step,game_observations=self.game_observations,
            capability_observations=self.capability_observations,last_metrics=self.last_metrics,
            capability_metrics=self.capability_metrics,
            controller=self.controller.state_dict() if self.controller else None,
            curriculum=self.curriculum.state_dict() if self.curriculum else None))

    def load_state_dict(self,state):
        if state.get('schema')!='game_coordinator_v2':raise ValueError('v2 coordinator state required')
        for key in ('next_game_audit_step','next_capability_step','game_observations',
                    'capability_observations','last_metrics','capability_metrics'):
            setattr(self,key,deepcopy(state[key]))
        for name,cls,cfg in [('controller',GameControllerV2,ControllerConfig),
                             ('curriculum',CapabilityCurriculumV2,CurriculumConfigV2)]:
            value=state.get(name)
            component=cls(cfg(**value['config'])) if value else None
            if component:component.load_state_dict(value)
            setattr(self,name,component)
