"""Bounded post-reception actions with explicit units and private randomness.

These are known digital actions, not estimates of pure transmitter/receiver
hardware. Curvature is an artificial pressure test, never orbital Doppler.
"""
import math
import torch
from experiments.cvs_multi_disentangle.physics import apply_linear, _iq

SAMPLE_RATE = 25_000_000.
LINEAR_PARAMETER_DIM = 4
TEMPORAL_PARAMETER_DIM = 4
PHASE_LIMIT_RAD = .25
CFO_LIMIT_HZ = 3120.
PHASE_NOISE_LIMIT_RAD = .025
CURVATURE_PRESSURE_HZ_S = 7.34e8


def _generator(generator):
    if generator is None or str(generator.device) != 'cpu':
        raise ValueError('An explicit private CPU generator is required')


def phase_basis(like):
    """Three orthogonal smooth phase modes; stochastic, not quadratic drift."""
    t = (torch.arange(256, device=like.device, dtype=like.dtype) + .5) / 256
    return torch.stack([math.sqrt(2.) * torch.sin(math.pi * (k + 1) * t) / (k + 1)
                        for k in range(3)])


def phase_process(batch, generator, like):
    """Finite clipped-Gaussian sine-series process with registered spectral decay.

    The same realization is passed to all endpoints and exact29 computations.
    Coefficients clip at +/-2.5 to retain a bounded proposal support. This is
    not a claim of calibrated physical phase noise.
    """
    _generator(generator)
    coefficients = torch.randn(batch, 3, generator=generator).clamp(-2.5, 2.5).to(like)
    return coefficients @ phase_basis(like) / math.sqrt(sum(1 / (k + 1)**2 for k in range(3)))


def sample_parameters(kind, batch, generator, like, pressure=False):
    _generator(generator)
    if kind not in ('linear', 'temporal'):
        raise ValueError('Unknown physical action')
    p = (2 * torch.rand(batch, 4, generator=generator) - 1).to(like)
    if kind == 'temporal':
        p[:, 0] *= PHASE_LIMIT_RAD
        p[:, 1] *= CFO_LIMIT_HZ
        p[:, 2] = (p[:, 2] + 1) * .5 * PHASE_NOISE_LIMIT_RAD
        p[:, 3] *= CURVATURE_PRESSURE_HZ_S if pressure else 0.
    return p


def apply_temporal(x, parameters, phase_noise=None):
    z = _iq(x)
    if parameters.shape != (len(x), 4):
        raise ValueError('T parameters must be [phase_rad,CFO_Hz,phase_std_rad,curvature_Hz/s]')
    p = parameters.to(x)
    if phase_noise is None:
        if bool((p[:, 2] != 0).any()):
            raise ValueError('Nonzero random phase requires an explicit seeded realization')
        phase_noise = torch.zeros_like(x[:, 0])
    if phase_noise.shape != x[:, 0].shape:
        raise ValueError('Random phase realization must be [B,256]')
    t = (torch.arange(256, device=x.device, dtype=x.dtype) - 127.5) / SAMPLE_RATE
    phase = (p[:, 0, None] + 2 * math.pi * p[:, 1, None] * t
             + math.pi * p[:, 3, None] * t.square() + p[:, 2, None] * phase_noise.to(x))
    out = z * torch.polar(torch.ones_like(phase), phase)
    return torch.stack((out.real, out.imag), 1)


def apply_action(x, p, kind, phase_noise=None):
    if kind == 'linear':
        return apply_linear(x, p)
    if kind == 'temporal':
        return apply_temporal(x, p, phase_noise)
    raise ValueError('Unknown physical action')


def factorial_batch(x, generator, step=0, pressure=False):
    _iq(x)
    pl = sample_parameters('linear', len(x), generator, x)
    pt = sample_parameters('temporal', len(x), generator, x, pressure)
    noise = phase_process(len(x), generator, x)
    xl, xt = apply_linear(x, pl), apply_temporal(x, pt, noise)
    order = 'LT' if step % 2 == 0 else 'TL'
    both = apply_temporal(xl, pt, noise) if order == 'LT' else apply_linear(xt, pl)
    return dict(x00=x, x10=xl, x01=xt, x11=both, linear_parameters=pl,
                temporal_parameters=pt, phase_noise=noise, order=order,
                curvature_is_artificial_pressure=bool(pressure))


def quality_proxies(x):
    """Repeat coherence, fitted-repeat residual, phase dispersion; never SNR.

    All survive unit RMS normalization. Low coherence itself reports that a
    packet does not support the assumed repeated-preamble proxy.
    """
    z = _iq(x)
    a, b = z[:, 80:140], z[:, 100:160]
    energy_a = a.abs().square().mean(1).clamp_min(1e-10)
    energy_b = b.abs().square().mean(1).clamp_min(1e-10)
    cross = (b * a.conj()).mean(1)
    coherence = cross.abs() / (energy_a * energy_b).sqrt()
    gain = cross / energy_a
    residual = (b - gain[:, None] * a).abs().square().mean(1) / energy_b
    unit_cross = b * a.conj()
    unit_cross = unit_cross / unit_cross.abs().clamp_min(1e-10)
    dispersion = 1 - unit_cross.mean(1).abs()
    return torch.stack((coherence.clamp(0, 1), residual.clamp_min(0), dispersion.clamp(0, 1)), 1)


def cfo_proxy(x):
    """Lag20 CFO estimate with +/-Fs/40 alias range, not pure TX CFO."""
    z = _iq(x)
    cross = (z[:, 100:160] * z[:, 80:140].conj()).mean(1)
    return torch.angle(cross) * SAMPLE_RATE / (2 * math.pi * 20)


def cfo_additive_diagnostic(x, tx, rx, day):
    """Source-only y + (RX,day) additive least squares and interaction residual."""
    keys_y = sorted(set(map(int, tx)))
    keys_rd = sorted(set(zip(map(int, rx), map(int, day))))
    if not len(x):
        return dict(status='N/A_EMPTY')
    yindex = {v: i for i, v in enumerate(keys_y)}
    rdindex = {v: i for i, v in enumerate(keys_rd)}
    design = torch.zeros(len(x), len(keys_y) + len(keys_rd), dtype=torch.float64)
    for i, (y, r, d) in enumerate(zip(tx, rx, day)):
        design[i, yindex[int(y)]] = 1
        design[i, len(keys_y) + rdindex[(int(r), int(d))]] = 1
    values = cfo_proxy(x).detach().cpu().double()
    coef = torch.linalg.pinv(design) @ values
    residual = values - design @ coef
    variance = (values - values.mean()).square().mean()
    return dict(samples=len(x), tx_count=len(keys_y), rx_day_count=len(keys_rd),
                design_rank=int(torch.linalg.matrix_rank(design)),
                residual_rmse_hz=float(residual.square().mean().sqrt()),
                residual_variance_fraction=float(residual.square().mean() / variance.clamp_min(1e-12)),
                interpretation='Observed additive association; residual includes TX-RX interaction, noise and aliasing; coefficients not uniquely physical.')


def contract():
    return dict(sample_rate_hz=SAMPLE_RATE, linear_tap_scale=.06,
                temporal_units=['rad','Hz','rad','Hz/s'], phase_limit_rad=PHASE_LIMIT_RAD,
                cfo_limit_hz=CFO_LIMIT_HZ, phase_noise_limit_rad=PHASE_NOISE_LIMIT_RAD,
                phase_process='3-mode Gaussian coefficients clipped +/-2.5; sine series 1/k amplitude; explicit realization',
                curvature_training_hz_s=0., curvature_pressure_hz_s=CURVATURE_PRESSURE_HZ_S,
                curvature_interpretation='artificial stress only, not orbital Doppler rate',
                order='alternating LT/TL by auxiliary step; identical across controls',
                rng='private CPU generator', rms_renormalization=False,
                quality='lag20 coherence, fitted residual ratio, phase dispersion; uncalibrated proxies')
