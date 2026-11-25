"""Fill a missing WiSig L-STF prefix from the same received packet's repeats."""

import numpy as np

from .wisig_joint_preamble import STARTS, waveform
from .wisig_smooth_preamble import TIME, aligned, blend_weights
from .wisig_timing import NOMINAL, PERIOD


MODES = ('public', 'repeat', 'public_with_repeat_residual')


def readouts(raw, delta, omega, response):
    public = aligned(raw, delta, omega, response, varying=True)
    repeat = public.copy()
    residual = public.copy()
    first_present = -delta
    row, col = np.nonzero(TIME[None] < first_present[:, None])
    if len(row):
        # Skip one period after the capture edge. Three later copies still lie
        # within L-STF for the registered -40 to +40 sample window.
        gap = first_present[row] + PERIOD - col
        next_cycle = col + ((gap + PERIOD-1) // PERIOD)*PERIOD
        donor = next_cycle[:, None] + PERIOD*np.arange(3)
        donor_received = donor + delta[row, None]
        drift = np.exp(1j*omega[row, None]*(col[:, None] - donor))
        level = response[row, delta[row] - (STARTS[0] - NOMINAL)]
        nominal = waveform(donor-NOMINAL) * (blend_weights(donor_received)*level[:, None]).sum(-1)
        nominal *= np.exp(1j*omega[row, None]*donor_received)
        actual = raw[row[:, None], donor_received]
        repeat[row, col] = (drift*actual).mean(1)
        residual[row, col] += (drift*(actual-nominal)).mean(1)
    return dict(zip(MODES, (public, repeat, residual), strict=True))
