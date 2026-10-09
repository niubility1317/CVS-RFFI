"""Known, mild post-reception interventions; no TX/RX recovery claim.

All random draws use an explicitly supplied CPU generator. The interventions
never fit a signal, inspect labels, or mutate PyTorch's training RNG streams.
"""
import torch

LINEAR_PARAMETER_DIM = 4
TEMPORAL_PARAMETER_DIM = 3
LINEAR_TAP_SCALE = 0.06
TEMPORAL_SCALES = (0.08, 0.10, 0.06)


def _iq(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (2, 256):
        raise ValueError('Expected IQ [B,2,256]')
    if not x.is_floating_point():
        raise TypeError('IQ must use a real floating dtype')
    return torch.complex(x[:, 0], x[:, 1])


def _parameters(x, p, width):
    if tuple(p.shape) != (x.shape[0], width):
        raise ValueError('Intervention parameter shape mismatch')
    return p.to(device=x.device, dtype=x.dtype)


def apply_linear(x, parameters):
    """Identity plus two causal complex taps at delays 1 and 3 (zero boundary)."""
    z = _iq(x)
    p = _parameters(x, parameters, LINEAR_PARAMETER_DIM)
    y = z
    for delay, offset in ((1, 0), (3, 2)):
        tap = LINEAR_TAP_SCALE * torch.complex(p[:, offset], p[:, offset + 1])
        shifted = torch.cat((torch.zeros_like(z[:, :delay]), z[:, :-delay]), dim=1)
        y = y + tap[:, None] * shifted
    return torch.stack((y.real, y.imag), dim=1)


def apply_temporal(x, parameters):
    """Small constant, linear and quadratic receive-side phase perturbation."""
    z = _iq(x)
    p = _parameters(x, parameters, TEMPORAL_PARAMETER_DIM)
    n = torch.linspace(-1., 1., x.shape[-1], device=x.device, dtype=x.dtype)
    phase = (TEMPORAL_SCALES[0] * p[:, 0, None]
             + TEMPORAL_SCALES[1] * p[:, 1, None] * n
             + TEMPORAL_SCALES[2] * p[:, 2, None] * n.square())
    y = z * torch.complex(torch.cos(phase), torch.sin(phase))
    return torch.stack((y.real, y.imag), dim=1)


def factorial_batch(x, generator):
    """Return the exact ordered 2x2 factorial observations and known targets."""
    _iq(x)
    if generator is None or str(generator.device) != 'cpu':
        raise ValueError('A private CPU torch.Generator is required')
    p_l = (2 * torch.rand((len(x), LINEAR_PARAMETER_DIM), generator=generator) - 1).to(x)
    p_t = (2 * torch.rand((len(x), TEMPORAL_PARAMETER_DIM), generator=generator) - 1).to(x)
    x_l = apply_linear(x, p_l)
    x_t = apply_temporal(x, p_t)
    return dict(x00=x, x10=x_l, x01=x_t, x11=apply_temporal(x_l, p_t),
                linear_parameters=p_l, temporal_parameters=p_t)


def contract():
    return dict(scope='Known mild post-receive interventions only',
                shape=[2, 256], linear_delays=[1, 3], linear_tap_scale=LINEAR_TAP_SCALE,
                linear_parameters=LINEAR_PARAMETER_DIM,
                temporal_parameters=TEMPORAL_PARAMETER_DIM,
                temporal_basis=['constant', 'linear', 'quadratic'],
                temporal_scales=list(TEMPORAL_SCALES),
                parameter_distribution='independent uniform [-1,1]',
                boundary='causal zero padding', order='temporal(linear(x))',
                rms_renormalization=False, rng='explicit private CPU generator',
                exact_physical_factor_identification=False)
