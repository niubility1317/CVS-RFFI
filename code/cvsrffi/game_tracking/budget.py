"""Rolling-window budgets charged only for work actually performed.

Charge successful head steps separately from a later rejected coupled update.
Field evaluations and audit time are actual computation, even on rejected steps.
`base_steps` and `corrections` count committed main updates only.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class BudgetConfig:
    window_steps: int = 100
    max_head_steps: int = 20
    max_field_evaluations: int = 20
    max_audit_seconds: float = 60.0
    max_correction_fraction: float = 0.2

    def __post_init__(self):
        if self.window_steps < 1 or min(self.max_head_steps, self.max_field_evaluations, self.max_audit_seconds) < 0:
            raise ValueError('invalid compute budget')
        if not 0 <= self.max_correction_fraction <= 1:
            raise ValueError('invalid correction fraction')


class ComputeBudget:
    KEYS = ('head_steps', 'field_evaluations', 'audit_seconds', 'corrections', 'base_steps')

    def __init__(self, config=BudgetConfig()):
        self.config, self.events, self.last_step = config, [], -1
        self.lifetime = dict.fromkeys(self.KEYS, 0)

    def _prune(self, step):
        if step < self.last_step:
            raise ValueError('budget step cannot move backwards')
        self.last_step = int(step)
        self.events = [event for event in self.events if event['step'] > step - self.config.window_steps]

    def usage(self, step):
        self._prune(step)
        return {key: sum(e[key] for e in self.events) for key in self.KEYS}

    def can_afford(self, step, **costs):
        self._validate(costs)
        current = self.usage(step)
        proposed = {key: current[key] + costs.get(key, 0) for key in self.KEYS}
        reasons = []
        for key, cap in (('head_steps', self.config.max_head_steps),
                         ('field_evaluations', self.config.max_field_evaluations),
                         ('audit_seconds', self.config.max_audit_seconds)):
            if costs.get(key, 0) and proposed[key] > cap:
                reasons.append(key + '_budget_exhausted')
        if costs.get('corrections', 0) and proposed['corrections'] > self.config.max_correction_fraction * proposed['base_steps']:
            reasons.append('correction_fraction_exhausted')
        return not reasons, reasons

    def commit(self, step, **actual_costs):
        """Record observed costs, including overruns; never hide actual work."""
        self._validate(actual_costs)
        self._prune(step)
        event = {key: actual_costs.get(key, 0) for key in self.KEYS}
        event['step'] = int(step)
        self.events.append(event)
        for key in self.KEYS:
            self.lifetime[key] += event[key]

    def _validate(self, costs):
        if set(costs) - set(self.KEYS):
            raise ValueError('unknown budget counter')
        for key, value in costs.items():
            if not math.isfinite(value) or value < 0 or (key != 'audit_seconds' and int(value) != value):
                raise ValueError('actual costs must be finite nonnegative counts/time')

    def state_dict(self):
        return dict(config=asdict(self.config), events=[dict(e) for e in self.events],
                    lifetime=dict(self.lifetime), last_step=self.last_step)

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('budget configuration differs from checkpoint')
        for event in state['events']:
            self._validate({k: event[k] for k in self.KEYS})
        self.events = [dict(e) for e in state['events']]
        self.lifetime, self.last_step = dict(state['lifetime']), int(state['last_step'])
