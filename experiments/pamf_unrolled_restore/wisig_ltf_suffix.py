"""Estimate an L-LTF suffix on the current packet without editing received samples."""

import numpy as np

from .wisig_smooth_preamble import TIME, aligned, blend_weights
from .wisig_timing import NOMINAL, STARTS, long_template


FILTERS = {'scalar': ((0,), ()), 'short': ((-2, -1, 0, 1, 2), ()),
           'widely_linear': ((-2, -1, 0, 1, 2), (0, 1))}
MODES = ('public', *(name for name in FILTERS), *(name+'_held' for name in FILTERS))


def design(time, name):
    direct, mirror = FILTERS[name]
    base = [long_template(time-NOMINAL-lag) for lag in direct]
    base.extend(long_template(time-NOMINAL-lag).conj() for lag in mirror)
    return np.stack(base, axis=-1).astype(np.complex64)


def ridge_fit(basis, residual):
    adjoint = basis.conj().T
    gram = adjoint @ basis
    penalty = .1 * gram.diagonal().real.mean()
    return np.linalg.solve(gram + penalty*np.eye(len(gram), dtype=gram.dtype), adjoint @ residual.T)


def readouts(raw, delta, omega, response):
    """Correct the public suffix with scalar, short, or widely linear L-LTF fits."""
    public = aligned(raw, delta, omega, response, varying=True)
    results = {name: public.copy() for name in MODES if name != 'public'}
    used = {name: np.zeros(len(raw), dtype=bool) for name in FILTERS}
    for displacement in np.unique(delta[delta > 0]):
        rows = np.flatnonzero(delta == displacement)
        start = NOMINAL + 3  # Avoid the start of L-LTF and give the short taps full support.
        end = len(TIME)-displacement
        observed, missing = TIME[start:end], TIME[end:]
        hold = max(6, len(observed)//3)
        split = len(observed)-hold
        if split < 6:
            continue
        r_observed, r_missing = observed+displacement, missing+displacement
        phase_observed = np.exp(-1j*omega[rows, None]*r_observed)
        phase_missing = np.exp(1j*omega[rows, None]*r_missing)
        coeff = response[rows, displacement + NOMINAL - STARTS[0]]
        base = (coeff @ blend_weights(r_observed).T)*long_template(observed-NOMINAL)
        residual = (raw[np.ix_(rows, r_observed)]*phase_observed-base).astype(np.complex64)
        validation = residual[:, split:]
        error = np.abs(validation)**2
        for name in FILTERS:
            basis = design(observed, name)
            held = basis[split:] @ ridge_fit(basis[:split], residual[:, :split])
            held_error = np.abs(validation-held.T)**2
            allow = held_error.sum(-1) < .9*error.sum(-1)
            update = (design(missing, name) @ ridge_fit(basis, residual)).T*phase_missing
            results[name][np.ix_(rows, missing)] += update
            results[name+'_held'][np.ix_(rows[allow], missing)] += update[allow]
            used[name][rows] = allow
    return dict(public=public, **results), used
