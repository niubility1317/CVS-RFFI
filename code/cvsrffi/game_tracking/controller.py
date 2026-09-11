"""Source-only finite action controller. Threshold defaults are examples for
local tests, requiring source calibration and a frozen config for formal runs.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ControllerConfig:
    lag_enter: float = 0.2
    lag_exit: float = 0.1
    readable_min: float = 0.1
    imbalance_enter: float = 0.5
    imbalance_exit: float = 0.3
    identity_min: float = 0.5
    margin_min: float = 0.0
    confirmation_windows: int = 2
    cooldown_steps: int = 5
    max_age_steps: int = 10
    max_version_lag: int = 0
    catchup_steps: int = 2
    audit_interval: int = 10
    sparse_audit_interval: int = 50
    calibration_id: str = 'LOCAL_TEST_ONLY_UNCALIBRATED'

    def __post_init__(self):
        if self.lag_enter <= self.lag_exit or self.imbalance_enter <= self.imbalance_exit:
            raise ValueError('enter thresholds must exceed exit thresholds')
        if min(self.confirmation_windows, self.audit_interval) < 1 or self.catchup_steps < 0:
            raise ValueError('invalid controller window or action budget')
        if min(self.cooldown_steps, self.max_age_steps, self.max_version_lag) < 0:
            raise ValueError('invalid controller freshness')
        if self.sparse_audit_interval < self.audit_interval:
            raise ValueError('sparse cadence cannot be denser')


def signal_fresh(metrics, step, encoder_version, max_age_steps, max_version_lag=0):
    """Reject future observations, stale versions and absent/nonfinite evidence."""
    if not metrics or not metrics.get('valid', False):
        return False
    try:
        age, lag = step - metrics['step'], encoder_version - metrics['encoder_version']
        return 0 <= age <= max_age_steps and 0 <= lag <= max_version_lag
    except (KeyError, TypeError):
        return False


class GameController:
    def __init__(self, config=ControllerConfig()):
        self.config = config
        self.state, self.pending, self.streak = 'NORMAL', None, 0
        self.last_change, self.last_observation = -10**12, None

    def decide(self, metrics, *, step, encoder_version, budget=None):
        """Input scalar fields: G_lag, S_rx (or native S_domain), identity,
        margin, direction_imbalance. Identity/margin must be independently valid.
        Does not execute actions or charge budget; caller records actual work.
        """
        c = self.config
        evidence_valid = signal_fresh(metrics, step, encoder_version, c.max_age_steps, c.max_version_lag)
        numeric = ('G_lag', 'identity', 'margin')
        evidence_valid = evidence_valid and all(isinstance(metrics.get(k), (float, int)) and math.isfinite(metrics[k]) for k in numeric)
        readable = metrics.get('S_rx', metrics.get('S_domain')) if metrics else None
        evidence_valid = evidence_valid and isinstance(readable, (float, int)) and math.isfinite(readable)
        before = self.state
        if not evidence_valid:
            self.pending, self.streak = None, 0
            return self._result('HOLD_CURRICULUM', before, 'invalid_or_stale_source_audit', True, 0)
        if metrics['identity'] < c.identity_min or metrics['margin'] < c.margin_min or not metrics.get('identity_valid', True):
            self.pending, self.streak = None, 0
            return self._result('HOLD_CURRICULUM', before, 'identity_or_margin_below_source_threshold', True, 0)
        lag, imbalance = metrics['G_lag'], metrics.get('direction_imbalance')
        high_lag = lag >= (c.lag_exit if self.state == 'CATCHUP' else c.lag_enter)
        high_imbalance = (isinstance(imbalance, (int, float)) and math.isfinite(imbalance) and
                          metrics.get('gradient_valid', False) and
                          imbalance >= (c.imbalance_exit if self.state == 'CORRECT' else c.imbalance_enter))
        desired = ('CATCHUP' if c.catchup_steps > 0 and high_lag and readable >= c.readable_min else
                   'CORRECT' if lag <= c.lag_exit and high_imbalance else 'NORMAL')
        observation = (metrics['step'], metrics['encoder_version'])
        if observation != self.last_observation:
            self.streak = self.streak + 1 if self.pending == desired else 1
            self.pending, self.last_observation = desired, observation
        if desired != self.state and self.streak >= c.confirmation_windows and step - self.last_change >= c.cooldown_steps:
            self.state, self.last_change = desired, step
        # Hysteresis cannot authorize an action contradicted by current evidence.
        action = self.state if self.state == desired else 'NORMAL'
        reason = 'confirmed_' + desired.lower() if action == desired else 'confirmation_or_cooldown'
        k = c.catchup_steps if action == 'CATCHUP' else 0
        if budget is not None and action in ('CATCHUP', 'CORRECT'):
            costs = dict(head_steps=k) if action == 'CATCHUP' else dict(field_evaluations=1, corrections=1, base_steps=1)
            allowed, reasons = budget.can_afford(step, **costs)
            if not allowed:
                action, k, reason = 'NORMAL', 0, ','.join(reasons)
        result = self._result(action, before, reason, False, k)
        if desired == 'NORMAL' and lag <= c.lag_exit and readable < c.readable_min:
            result['next_audit_interval'] = c.sparse_audit_interval
        return result

    def _result(self, action, before, reason, hold, k):
        return dict(action=action, state_before=before, state_after=self.state, reason=reason,
                    hold_curriculum=hold, catchup_steps=k, main_solver='eg' if action == 'CORRECT' else 'normal',
                    next_audit_interval=self.config.audit_interval)

    def state_dict(self):
        return dict(config=asdict(self.config), state=self.state, pending=self.pending, streak=self.streak,
                    last_change=self.last_change, last_observation=self.last_observation)

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('controller configuration differs from checkpoint')
        for key in ('state', 'pending', 'streak', 'last_change', 'last_observation'):
            setattr(self, key, state[key])
        if self.last_observation is not None:
            self.last_observation = tuple(self.last_observation)


class GameControllerV2:
    """Consume typed, trusted evidence once; failures never certify health."""
    def __init__(self,config=ControllerConfig()):
        self.config=config;self.state='NORMAL';self.pending=None;self.streak=0
        self.last_observation=None;self.last_change=-10**12
        self.consumed_observations=set();self.requested_observations=set();self.accepted_events=0

    def _normal(self,reason):
        return dict(action='NORMAL',reason=reason,catchup_steps=0,hold_curriculum=False,
                    main_solver='normal',next_audit_interval=self.config.audit_interval)

    def decide(self,m,*,step,encoder_version,budget=None):
        c=self.config
        if not m or m.get('schema')!='game_audit_v2':return self._normal('v2_evidence_required')
        if not m.get('data_valid') or not m.get('coverage_valid'):return self._normal('invalid_data_or_coverage')
        age=step-m.get('step',step+1);version_lag=encoder_version-m.get('encoder_version',encoder_version+1)
        if not 0<=age<=c.max_age_steps or not 0<=version_lag<=c.max_version_lag:
            return self._normal('stale_or_future_evidence')
        obs=m.get('observation_id')
        if not isinstance(obs,str) or not obs:return self._normal('missing_observation_id')
        if obs in self.consumed_observations or obs in self.requested_observations:
            return self._normal('observation_already_consumed')
        lag=m.get('lag',{})
        value=lag.get('gap_normalized');readable=lag.get('readability')
        reliable=lag.get('status') in ('RELIABLE_HIGH_GAP','RELIABLE_LOW_GAP')
        if (not reliable or not lag.get('quality_pass') or not lag.get('control_ready') or
            not isinstance(value,(int,float)) or not math.isfinite(value) or
            not isinstance(readable,(int,float)) or not math.isfinite(readable)):
            self.pending=None;self.streak=0
            return self._normal('untrusted_lag_evidence')
        cap=m.get('capability',{})
        cap_age=step-cap.get('step',step+1);cap_version=encoder_version-cap.get('encoder_version',encoder_version+1)
        if (cap.get('schema')!='game_capability_v2' or not cap.get('valid') or not cap.get('identity_valid') or cap.get('collapsed',True) or
            not 0<=cap_age<=c.max_age_steps or not 0<=cap_version<=c.max_version_lag or
            not all(isinstance(cap.get(k),(int,float)) and math.isfinite(cap[k]) for k in ('identity','margin')) or
            cap['identity']<c.identity_min or cap['margin']<c.margin_min):
            self.pending=None;self.streak=0
            return self._normal('identity_protection_unavailable')
        gradient=m.get('gradient',{});imbalance=gradient.get('direction_imbalance')
        grad_ok=bool(gradient.get('valid') and gradient.get('representative') and
                     gradient.get('scope')=='full_current_training_objective' and
                     isinstance(imbalance,(int,float)) and math.isfinite(imbalance))
        # Source-calibrated numerical thresholds cannot change the estimand's
        # quality state: HIGH proves recoverable improvement, not convergence.
        high=(lag['status']=='RELIABLE_HIGH_GAP' and
              value >= (c.lag_exit if self.state=='CATCHUP' else c.lag_enter))
        desired=('CATCHUP' if c.catchup_steps > 0 and high and readable>=c.readable_min else 'CORRECT'
                 if lag['status']=='RELIABLE_LOW_GAP' and value<=c.lag_exit and grad_ok and imbalance >= (c.imbalance_exit if self.state=='CORRECT' else c.imbalance_enter)
                 else 'NORMAL')
        if obs!=self.last_observation:
            self.streak=self.streak+1 if self.pending==desired else 1
            self.pending=desired;self.last_observation=obs
        if desired=='NORMAL':return self._normal('no_trusted_action_condition')
        if self.streak<c.confirmation_windows or step-self.last_change<c.cooldown_steps:
            return self._normal('confirmation_or_cooldown')
        k=c.catchup_steps if desired=='CATCHUP' else 0
        costs=dict(head_steps=k) if k else dict(field_evaluations=1,corrections=1,base_steps=1)
        if budget is not None and not budget.can_afford(step,**costs)[0]:
            self.consumed_observations.add(obs)
            return self._normal('action_budget_exhausted')
        self.requested_observations.add(obs);self.state=desired;self.last_change=step
        return dict(action=desired,reason='trusted_'+desired.lower(),catchup_steps=k,hold_curriculum=False,
                    main_solver='eg' if desired=='CORRECT' else 'normal',next_audit_interval=c.audit_interval,
                    observation_id=obs)

    def record_outcome(self,observation_id,*,accepted,committed_head_steps=0):
        if observation_id not in self.requested_observations:
            raise ValueError('outcome requires one outstanding observation action')
        self.requested_observations.remove(observation_id);self.consumed_observations.add(observation_id)
        self.accepted_events+=int(bool(accepted) or committed_head_steps>0)

    def state_dict(self):
        return dict(schema='game_controller_v2',config=asdict(self.config),state=self.state,pending=self.pending,
                    streak=self.streak,last_observation=self.last_observation,last_change=self.last_change,
                    consumed_observations=sorted(self.consumed_observations),
                    requested_observations=sorted(self.requested_observations),accepted_events=self.accepted_events)

    def load_state_dict(self,state):
        if state.get('schema')!='game_controller_v2' or state.get('config')!=asdict(self.config):
            raise ValueError('v2 controller schema/config mismatch')
        for k in ('state','pending','streak','last_observation','last_change','accepted_events'):setattr(self,k,state[k])
        self.consumed_observations=set(state['consumed_observations'])
        self.requested_observations=set(state['requested_observations'])
