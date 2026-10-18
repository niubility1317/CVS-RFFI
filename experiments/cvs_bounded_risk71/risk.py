"""Bounded finite-pool risk weights over the unchanged P70 stratified reader."""
from copy import deepcopy
import math
from numbers import Real


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(name + ' must be a finite real number')
    return float(value)


def _distribution(values, name):
    try:
        result = [_real(value, name) for value in values]
    except TypeError as error:
        raise ValueError(name + ' must contain four group weights') from error
    if len(result) != 4 or any(value < 0 for value in result) or not math.isclose(math.fsum(result), 1., rel_tol=0., abs_tol=1e-10):
        raise ValueError(name + ' must be four nonnegative weights summing to one')
    return result


def mixture(p_g, gamma, v):
    """Return (1-gamma)*nominal+gamma*v without changing pool support."""
    nominal = _distribution(p_g, 'nominal p_g')
    target = _distribution(v, 'v_g')
    gamma = _real(gamma, 'gamma')
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must lie in [0,1]')
    pi = [(1. - gamma) * p + gamma * value for p, value in zip(nominal, target)]
    if any(p == 0. and weight > 0. for p, weight in zip(nominal, pi)):
        raise ValueError('Positive risk weight has no nominal-pool observation support')
    return pi


class ReadProxy:
    """Change the loss's p_g argument while returning the exact existing pack.

    The internal method remains QUALITY_STRAT for the P70 hook.  Reports retain
    nominal p_g and identify the bounded risk separately from pool allocation.
    """
    method = _kind = 'QUALITY_STRAT'

    def __init__(self, reader, gamma, v):
        if getattr(reader, 'method', None) != 'QUALITY_STRAT':
            raise ValueError('Bounded risk requires the existing QUALITY_STRAT allocation')
        self.nominal_p_g = _distribution(reader.summary()['p_g'], 'nominal p_g')
        self.gamma = _real(gamma, 'gamma')
        self.v_g = _distribution(v, 'v_g')
        self.pi_g = mixture(self.nominal_p_g, self.gamma, self.v_g)
        self.reader = reader

    @property
    def calls(self):
        return self.reader.calls

    @property
    def packets(self):
        return self.reader.packets

    def __deepcopy__(self, memo):
        value = type(self)(deepcopy(self.reader, memo), self.gamma, self.v_g)
        memo[id(self)] = value
        return value

    def consume(self, step, epoch, device):
        iq, labels, domains, metadata, original = self.reader.consume(step, epoch, device)
        if list(original['p_g']) != self.nominal_p_g:
            raise ValueError('Underlying nominal pool weights changed')
        record = dict(original)
        record.update(p_g=list(self.pi_g), nominal_p_g=list(self.nominal_p_g),
                      pi_g=list(self.pi_g), v_g=list(self.v_g), gamma=self.gamma,
                      risk_method='BOUNDED_RISK', allocation_method='QUALITY_STRAT')
        return iq, labels, domains, metadata, record

    def _report(self, original):
        result = dict(original)
        result.update(method='BOUNDED_RISK', allocation_method='QUALITY_STRAT',
                      p_g=list(self.nominal_p_g), nominal_p_g=list(self.nominal_p_g),
                      pi_g=list(self.pi_g), v_g=list(self.v_g), gamma=self.gamma,
                      new_channel_calls=0)
        return result

    def summary(self):
        return self._report(self.reader.summary())

    def audit(self):
        return self._report(self.reader.audit())


def load(directory, *, seed, gamma, v):
    """Load a completed P70 pool only; never prepare or generate a channel."""
    from experiments.cvs_quality_risk70 import pool
    return ReadProxy(pool.load(directory, 'QUALITY_STRAT', seed=seed), gamma, v)
