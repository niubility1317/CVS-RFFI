"""Nested source fit expansion without prior or target records."""
import numpy as np
from experiments.phase1_spectral_synthesis.fit_sampling import select_fit


def select_large_fit(training,prior_ids,size):
    if size!=32768:raise ValueError('Only the registered 32768-record expansion is supported')
    original=select_fit(training,prior_ids,8192)
    eligible=np.setdiff1d(training,prior_ids,assume_unique=True)
    remaining=np.setdiff1d(eligible,original,assume_unique=True)
    needed=size-len(original)
    if len(remaining)<needed:raise ValueError('Insufficient eligible source records')
    extra=np.random.default_rng(2026110785).choice(remaining,size=needed,replace=False)
    return np.sort(np.r_[original,extra])
