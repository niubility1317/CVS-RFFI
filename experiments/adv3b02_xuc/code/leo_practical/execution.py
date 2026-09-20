"""Execution-only switches. They do not change the physical Config or RNG."""
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import numpy as np


@dataclass(frozen=True)
class Execution:
    constants: bool = False
    recurrence: bool = False
    config_reuse: bool = False
    light_metadata: bool = False
    identity_filter: bool = False


FAST = Execution(True, True, True, True, True)
REFERENCE = Execution()


@lru_cache(maxsize=64)
def full_config(cfg):
    return replace(cfg, processing_route="full")


@lru_cache(maxsize=64)
def constants(cfg):
    diffuse = 10**(np.asarray(cfg.diffuse_power_db)/10)
    a = math.log(10)/20
    los = np.exp(2*a*np.asarray(cfg.los_mean_db) + 2*a*a*np.square(cfg.los_std_db))
    los[2] = 0
    reference = float(los[0] + diffuse[0])
    pdp = np.exp(-np.arange(3, dtype=float))
    pdp /= pdp.sum()
    base = np.array([0., 1., 3.])
    delays = []
    for s in range(3):
        weights = diffuse[s]*pdp
        weights = weights.copy()
        weights[0] += los[s]
        weights /= weights.sum()
        spread = np.sqrt(np.sum(weights*(base - weights @ base)**2))
        delays.append(base*cfg.rms_delay_ns[s]*1e-9*cfg.fs_hz/spread)
    arrays = (diffuse, los, pdp, np.asarray(delays))
    for array in arrays:
        array.flags.writeable = False
    return (*arrays, reference)


def tracking(z, rho, innovation, draws, amplitude, compiled):
    if compiled:
        # lfilter's C recurrence uses double precision without fast-math.
        # Keep the multiplication order of the original scalar recurrence.
        from scipy.signal import lfilter
        following, _ = lfilter([innovation], [1., -rho], draws, zi=[rho*z])
        states = np.concatenate(([z], following[:-1]))
        return amplitude*states, float(following[-1])
    result = np.empty(len(draws))
    for i, e in enumerate(draws):
        result[i] = amplitude*z
        z = rho*z + innovation*e
    return result, z
