"""A small received-IQ response change, without TX hardware synthesis.

This is a candidate observation transform, not proof of label preservation or
an emulator that replaces a real receiver. No data, model or truth is opened.
"""
import hashlib
import math
import random

import torch


MAX_RADIUS = 0.06


def named_seed(seed, namespace, *parts):
    # A random stream key, not an artifact integrity or permission mechanism.
    payload = repr((int(seed), str(namespace), parts)).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')


def session_response(seed, *, stream, epoch, rx, day, eq):
    """One coefficient per virtual RX/day/eq session and epoch, independent of TX.

    Repeated packets/views in that virtual session receive identical response
    parameters. A new epoch refreshes the virtual session. This does not claim
    that recorded receiver hardware changes between epochs.
    """
    if stream not in ('train', 'source_probe') or epoch < 1:
        raise ValueError('Only separated source-only streams with positive epochs')
    if any(type(v) is not int or v < 0 for v in (rx, day, eq)):
        raise ValueError('Expected nonnegative physical RX/day/eq metadata')
    rng = random.Random(named_seed(seed, 'mixed_iq20/response/' + stream,
                                   epoch, rx, day, eq))
    radius = MAX_RADIUS * math.sqrt(rng.random())
    phase = 2 * math.pi * rng.random()
    return complex(radius * math.cos(phase), radius * math.sin(phase))


def apply_response(x, coefficients, *, boundary='reflect'):
    """y[n] = x[n] + a*(x[n-1]+x[n+1])/2; restore input packet RMS.

    The length and reference-window position stay fixed. Only first/last output
    samples use invented boundary values; the interior does not wrap. On the
    interior LTI response H(w)=1+a*cos(w), |H| is in [.94,1.06] before scalar
    power restoration. No new noise, image term, nonlinearity, CFO or clipping.
    """
    if (x.ndim != 3 or x.shape[1:] != (2, 256) or x.shape[0] < 1
            or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite IQ[B,2,256] required')
    if boundary not in ('reflect', 'edge'):
        raise ValueError('No circular or unspecified boundary handling')
    a = torch.as_tensor(coefficients, dtype=torch.complex64, device=x.device)
    if a.shape != (len(x),) or not bool(torch.isfinite(a).all()) or bool((a.abs() > MAX_RADIUS + 1e-7).any()):
        raise ValueError('One finite bounded coefficient per packet required')
    z = torch.complex(x[:, 0].float(), x[:, 1].float())
    left_edge = z[:, 1:2] if boundary == 'reflect' else z[:, :1]
    right_edge = z[:, -2:-1] if boundary == 'reflect' else z[:, -1:]
    previous = torch.cat((left_edge, z[:, :-1]), dim=1)
    following = torch.cat((z[:, 1:], right_edge), dim=1)
    unscaled = z + a[:, None] * (previous + following) * .5
    original_power = z.abs().square().mean(1)
    response_power = unscaled.abs().square().mean(1)
    scale = torch.sqrt(original_power / response_power.clamp_min(1e-30))
    scale = torch.where(original_power > 0, scale, torch.ones_like(scale))
    result = unscaled * scale[:, None]
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Response arithmetic produced nonfinite IQ')
    return torch.stack((result.real, result.imag), dim=1).to(x.dtype), {
        'scalar_power_restoration': scale.detach(),
        'boundary_dependent_indices': [0, 255],
        'unscaled_valid_interval': [1, 255],
        'numerical_index_delay_samples': 0,
        'reference_window': [80, 160],
        'unknown_TX_inverse': False,
        'new_added_noise': False,
        'label_preservation_validated': False,
    }
