"""Optional propagation-only FIR equalization and equivalent residual operators.

No TX/PA correction. Channel estimates are explicit truth-plus-error simulation
proxies, not estimates obtained from pilots in the supplied recording.
"""
import numpy as np


def linear_filter(x, kernel, boundary="reflect", context=None):
    """Causal linear FIR; no circular wrap. Return output and input history."""
    x, kernel = np.asarray(x, complex), np.asarray(kernel, complex)
    length = len(kernel)-1
    if length == 0:
        return x*kernel[0], np.empty(0, complex)
    if context is None:
        if boundary == "require_context":
            raise ValueError("FIR requires preceding context")
        prefix = np.pad(x, (length, 0), mode=boundary)[:length]
    else:
        context = np.asarray(context, complex)
        if len(context) < length:
            raise ValueError("FIR preceding context too short")
        prefix = context[-length:]
    extended = np.concatenate([prefix, x])
    y = np.convolve(extended, kernel, mode="full")[length:length+len(x)]
    return y, extended[-length:].copy()


def propagation_kernel(stream, state, coefficients, gain):
    from .channel import fractional_delay_kernel
    size = stream.history_size+1
    h = np.zeros(size, complex)
    for tap, delay in enumerate(stream.delays[state]):
        integer, fraction = int(np.floor(delay)), delay % 1
        kernel = fractional_delay_kernel(fraction, stream.cfg.fractional_delay_half_length)
        h[integer:integer+len(kernel)] += gain*coefficients[tap]*kernel
    return h


def design_equalizer(h, cfg, rng, noise_variance, input_power, residual_scale=1.0):
    """Finite causal MMSE or regularized/gain-capped ZF FIR approximation."""
    power = float(np.sum(abs(h)**2))
    variance = power*10**(cfg.channel_estimation_nmse_db/10)*residual_scale**2/max(1,len(h))
    error = np.sqrt(variance/2)*(rng.normal(size=len(h))+1j*rng.normal(size=len(h)))
    estimate = h+error
    nfft = 1 << int(np.ceil(np.log2(4*(len(h)+cfg.equalizer_length))))
    response = np.fft.fft(estimate, nfft)
    reg = (noise_variance/max(input_power,1e-30) if cfg.equalizer_method == "mmse"
           else cfg.zf_regularization)
    weights = response.conj()/(abs(response)**2+max(reg,1e-30))
    limit = 10**(cfg.equalizer_max_gain_db/20)
    weights *= np.minimum(1, limit/np.maximum(abs(weights),1e-30))
    weights *= np.exp(-2j*np.pi*np.arange(nfft)*cfg.equalizer_delay_samples/nfft)
    impulse = np.fft.ifft(weights)
    fir = impulse[:cfg.equalizer_length].copy()
    # Truncation can increase frequency peaks: enforce cap on the realized FIR.
    peak = float(np.max(abs(np.fft.fft(fir,nfft))))
    if peak > limit:
        fir *= limit/peak
    return fir, dict(method=cfg.equalizer_method,
        estimation_model="practical_channel_plus_error_proxy_not_pilot_estimator",
        estimated_nmse_db=cfg.channel_estimation_nmse_db+20*np.log10(residual_scale),
        regularization=float(reg), target_total_delay_samples=cfg.equalizer_delay_samples,
        fir_length=len(fir), max_gain_db=cfg.equalizer_max_gain_db,
        discarded_impulse_energy_fraction=float(np.sum(abs(impulse[len(fir):])**2)/max(np.sum(abs(impulse)**2),1e-30)))
