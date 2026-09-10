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
        if min(self.confirmation_windows, self.catchup_steps, self.audit_interval) < 1:
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
        desired = ('CATCHUP' if high_lag and readable >= c.readable_min else
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
