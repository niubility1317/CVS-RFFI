"""Analytic IQ fidelity measurements. They do not establish TX identity."""
import torch
from .design import LAGS


def complex_iq(value):
    if value.ndim != 3 or value.shape[1:] != (2, 256) or not value.is_floating_point():
        raise ValueError('Floating native IQ[B,2,256] required')
    return torch.complex(value[:, 0], value[:, 1])


def lag_correlations(value):
    z = complex_iq(value)
    out = []
    for lag in LAGS:
        left, right = z[:, :-lag], z[:, lag:]
        denominator = (left.abs().square().sum(1)*right.abs().square().sum(1)).clamp_min(1e-16).sqrt()
        out.append((left.conj()*right).sum(1)/denominator)
    return torch.stack(out, 1)


def relative_error(value, reference):
    return (value-reference).abs().square().mean(1)/reference.abs().square().mean(1).clamp_min(1e-8)


def fidelity(value, reference):
    if value.shape != reference.shape:
        raise ValueError('Matched paired clean IQ required')
    z, x = complex_iq(value), complex_iq(reference)
    result = dict(iq_nmse=relative_error(z, x),
        prefix232_nmse=relative_error(z[:, :232], x[:, :232]),
        tail24_nmse=relative_error(z[:, 232:], x[:, 232:]))
    xf, zf = torch.fft.fft(x, norm='ortho'), torch.fft.fft(z, norm='ortho')
    power_order = xf.abs().square().argsort(1)
    amplitude_order = x.abs().square().argsort(1)
    for q in range(4):
        band = power_order[:, q*64:(q+1)*64]
        result[f'fft_power_Q{q+1}_complex_nmse'] = relative_error(zf.gather(1, band), xf.gather(1, band))
        group = amplitude_order[:, q*64:(q+1)*64]
        local_x, local_z = x.gather(1, group), z.gather(1, group)
        result[f'amplitude_Q{q+1}_residual_nmse'] = relative_error(local_z, local_x)
        gain_residual = (local_x.conj()*(local_z-local_x)).sum(1)/local_x.abs().square().sum(1).clamp_min(1e-8)
        result[f'amplitude_Q{q+1}_complex_gain_error'] = gain_residual.abs().square()
    correlation_error = (lag_correlations(value)-lag_correlations(reference)).abs().square()
    result['lag_correlation_error'] = correlation_error.mean(1)
    for index, lag in enumerate(LAGS):
        result[f'lag{lag}_correlation_error'] = correlation_error[:, index]
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite source IQ fidelity metric')
    return result


def summarize(values):
    if not values or any(v.ndim != 1 or not len(v) for v in values.values()):
        raise ValueError('Nonempty per-example metrics required')
    return {name:dict(mean=float(v.mean()), median=float(torch.quantile(v, .5)),
        p90=float(torch.quantile(v, .9)), p95=float(torch.quantile(v, .95)), maximum=float(v.max()))
        for name, v in values.items()}
