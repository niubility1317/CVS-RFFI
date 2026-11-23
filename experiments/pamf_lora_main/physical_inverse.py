"""Source-only inverse of the replayed continuous and snapshot operators."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr

from .packet_data import CHIRPS
from .packet_physics import (CONTEXT, RESIDUAL_LATENCY, channel, chirp_grid, chirp_samples,
                             hardware, pack_windows, residual_channel)


def convolution(rows, kernels, width):
    positions = rows[:, None] - np.arange(kernels.shape[-1])[None, :]
    valid = (positions >= 0) & (positions < width) & (kernels != 0)
    return csr_matrix((kernels[valid], (np.nonzero(valid)[0], positions[valid])),
                      shape=(len(rows), width))


class ReplayAction:
    def __init__(self, propagation, main, image):
        self.propagation = propagation
        self.adjoint = propagation.conj().T.tocsr()
        self.main = main
        self.image = image

    def __call__(self, iq):
        propagated = self.propagation @ iq
        return self.main * propagated + self.image * propagated.conj()

    def adjoint_iq(self, iq):
        return self.adjoint @ (self.main.conj()*iq + self.image*iq.conj())

    def operator(self):
        m, n = self.propagation.shape

        def apply(value):
            p = self(value[:n] + 1j*value[n:])
            return np.concatenate((p.real, p.imag))

        def transpose(value):
            p = self.adjoint_iq(value[:m] + 1j*value[m:])
            return np.concatenate((p.real, p.imag))

        return LinearOperator((2*m, 2*n), matvec=apply, rmatvec=transpose, dtype=np.float64)


def full_kernels(meta, trace):
    from leo_practical.channel import fractional_delay_kernel
    half = meta['numerical_filter_latency_samples']
    delays = np.asarray(meta['delays_samples_by_state'])
    kernels = np.zeros((len(trace['state']), int(delays.max()) + 2*half + 1), dtype=np.complex128)
    for state, taps in enumerate(delays):
        active = trace['state'] == state
        for tap, delay in enumerate(taps):
            lag = int(delay)
            kernel = fractional_delay_kernel(delay-lag, half)
            kernels[active, lag:lag+len(kernel)] += trace['coefficients'][active, tap, None]*kernel
    return kernels * trace['path_gain'][:, None]


def validate(action, clean, observed, meta, trace):
    signal, noise = trace['pre_agc_signal'], trace['pre_agc_noise']
    g = 10.**(meta['agc_gain_db']/20.)
    if not np.array_equal(np.asarray(g*(signal+noise), dtype=observed.dtype), observed):
        raise ValueError('Operator trace does not reproduce the registered AGC and output precision')
    error = np.linalg.norm(action(clean)-signal) / max(np.linalg.norm(signal), 1e-12)
    if not np.isfinite(error) or error > 2e-9:
        raise ValueError(f'The source operator does not reproduce the injected noiseless signal: {error}')
    return signal, noise, g, float(error)


def inverse(action, observed, clean_observation, noise):
    # Use the same regularization for the two arms of the noise attribution.
    solver = action.operator()
    regularization = float(np.clip(np.sqrt(3. * np.mean(np.abs(noise)**2)), 1e-5, 1.))
    solutions = []
    for y in (observed, clean_observation):
        rhs = np.concatenate((y.real, y.imag))
        result = lsmr(solver, rhs, damp=regularization, atol=1e-5, btol=1e-5, maxiter=30)
        estimate = result[0]
        a, b = np.split(estimate, 2)
        x = a + 1j*b
        if not np.isfinite(x).all():
            raise FloatingPointError('Nonfinite known-operator solution')
        solutions.append(x)
    return solutions


def full_replay(z, scene, rx, pid):
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed
    x, _ = normalize_input(z)
    cfg = channel(scene)
    seed = stable_seed(2026112207, 'source-V', scene, rx, str(pid))
    stream = ChannelStream(cfg, seed, hardware(scene, f'source-V/{rx}'))
    y, meta, trace = stream.process(x[CONTEXT:], preceding_context=x[:CONTEXT],
                                    return_traces=True, return_operator=True)
    start = CONTEXT
    kernel = full_kernels(meta, trace)
    propagation = convolution(start+np.arange(len(y)), kernel, len(x))
    operator = ReplayAction(propagation, trace['operator_main'], trace['operator_image'])
    signal, noise, gain, err = validate(operator, x, y, meta, trace)
    solved, known_noise_solved = inverse(operator, y.astype(np.complex128)/gain, signal, noise)
    offset = meta['numerical_filter_latency_samples'] - CONTEXT
    # Direct and pure denoising use the registered sampling/latency convention.
    # The inverse directly estimates the native source time grid.
    return np.stack([pack_windows(w) for w in (chirp_samples(y, offset),
                     chirp_samples(gain*signal, offset), chirp_samples(solved),
                     chirp_samples(known_noise_solved))]), err


def residual_replay(z, scene, rx, pid):
    from leo_practical.channel import normalize_input, sample_geometry, stable_seed
    from leo_practical.residual import ResidualChannel
    x, _ = normalize_input(z)
    cfg = residual_channel(scene)
    receiver = hardware(scene, f'source-V/{rx}')
    parts = ('source-V', scene, rx, str(pid))
    geometry = sample_geometry(cfg, np.random.default_rng(stable_seed(2026112207, *parts, 'geometry')))
    start = 32 + RESIDUAL_LATENCY
    rows, errors = [], []
    for chirp, grid in zip(CHIRPS, chirp_grid(x), strict=True):
        stream = ResidualChannel(cfg, stable_seed(2026112207, *parts, 'chirp', chirp), receiver, geometry)
        y, meta, trace = stream.process(grid[start:start+256], preceding_context=grid[:start],
                                        return_traces=True, return_operator=True)
        h = trace['residual_kernel']
        if not np.array_equal(h.conj(), trace['image_kernel']) or meta['channel_equalization_applied']:
            raise ValueError('Residual operator requires the registered no-EQ observation')
        propagation = convolution(start+np.arange(len(y)), np.broadcast_to(h, (len(y), len(h))), len(grid))
        operator = ReplayAction(propagation, trace['operator_main'], trace['operator_image'])
        signal, noise, gain, err = validate(operator, grid, y, meta, trace)
        solved, known_noise_solved = inverse(operator, y.astype(np.complex128)/gain, signal, noise)
        rows.append((y, gain*signal, solved[32:288], known_noise_solved[32:288]))
        errors.append(err)
    return np.stack([pack_windows(np.stack(a)) for a in zip(*rows, strict=True)]), max(errors)
