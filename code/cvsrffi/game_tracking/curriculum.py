"""One-axis capability curriculum, with no epoch-forced upgrades.

Default thresholds are local-test examples. Formal runs must freeze thresholds
calibrated from legal source measurements, including clean/LEO consistency.
"""
from dataclasses import asdict, dataclass
import math

from .controller import signal_fresh


@dataclass(frozen=True)
class CurriculumConfig:
    identity_enter: float = 0.7
    identity_exit: float = 0.5
    margin_enter: float = 0.1
    margin_exit: float = 0.0
    consistency_enter: float = 0.8
    consistency_exit: float = 0.6
    lag_max: float = 0.1
    confirmation_windows: int = 3
    cooldown_steps: int = 10
    max_age_steps: int = 10
    max_version_lag: int = 0
    max_increment: float = 0.1
    initial_level: float = 0.0
    maximum_level: float = 1.0
    calibration_id: str = 'LOCAL_TEST_ONLY_UNCALIBRATED'

    def __post_init__(self):
        if self.identity_enter <= self.identity_exit or self.margin_enter <= self.margin_exit:
            raise ValueError('curriculum requires distinct enter/exit thresholds')
        if not -1 <= self.consistency_exit < self.consistency_enter <= 1:
            raise ValueError('consistency thresholds require -1 <= exit < enter <= 1')
        if self.confirmation_windows < 1 or min(self.cooldown_steps, self.max_age_steps, self.max_version_lag) < 0:
            raise ValueError('invalid curriculum confirmation/freshness')
        if self.max_increment <= 0 or not 0 <= self.initial_level <= self.maximum_level <= 1:
            raise ValueError('invalid curriculum bounds')


class CapabilityCurriculum:
    def __init__(self, config=CurriculumConfig()):
        self.config, self.level = config, config.initial_level
        self.ready, self.streak, self.last_observation = False, 0, None
        self.last_change, self.version = -10**12, 0

    def update(self, metrics, *, step, encoder_version, hold=False):
        c, old = self.config, self.level
        fresh = signal_fresh(metrics, step, encoder_version, c.max_age_steps, c.max_version_lag)
        finite = all(isinstance(metrics.get(k), (float, int)) and math.isfinite(metrics[k])
                     for k in ('identity', 'margin', 'G_lag', 'consistency')) if metrics else False
        reason = 'capability_hold'
        if hold or not fresh or not finite or not metrics.get('identity_valid', True) or metrics.get('collapsed', False):
            self.ready, self.streak = False, 0
            reason = 'external_hold' if hold else 'invalid_stale_or_collapsed'
        else:
            if (metrics['identity'] < c.identity_exit or metrics['margin'] < c.margin_exit
                    or metrics['consistency'] < c.consistency_exit or metrics['G_lag'] > c.lag_max):
                self.ready, self.streak = False, 0
            enter = (metrics['identity'] >= c.identity_enter and metrics['margin'] >= c.margin_enter
                     and metrics['consistency'] >= c.consistency_enter and metrics['G_lag'] <= c.lag_max)
            observation = (metrics['step'], metrics['encoder_version'])
            if observation != self.last_observation:
                self.streak = self.streak + 1 if enter else 0
                self.last_observation = observation
            if self.streak >= c.confirmation_windows:
                self.ready = True
            if self.ready and enter and step - self.last_change >= c.cooldown_steps:
                self.level = min(c.maximum_level, self.level + c.max_increment)
                if self.level != old:
                    self.last_change, self.version, self.streak = step, self.version + 1, 0
                    self.ready = False
                    reason = 'confirmed_source_capability'
        changed = self.level != old
        return dict(level=self.level, previous_level=old, changed=changed, ready=self.ready, reason=reason,
                    curriculum_version=self.version, invalidate_feature_cache=changed,
                    reset_optimistic=changed, reset_reason='curriculum_change' if changed else None)

    def state_dict(self):
        return dict(config=asdict(self.config), level=self.level, ready=self.ready, streak=self.streak,
                    last_observation=self.last_observation, last_change=self.last_change, version=self.version)

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('curriculum configuration differs from checkpoint')
        for key in ('level', 'ready', 'streak', 'last_observation', 'last_change', 'version'):
            setattr(self, key, state[key])
        if self.last_observation is not None:
            self.last_observation = tuple(self.last_observation)
