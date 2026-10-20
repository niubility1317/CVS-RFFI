"""NumPy receive-chain comparison, independent of neural inference and TX labels."""
import numpy as np

KEYS = ('coarse', 'kernel', 'coefficients', 'variance')


def validate_estimate(arrays, count):
    shapes = dict(coarse=(count, 2, 256), kernel=(count, 2, 64),
                  coefficients=(count, 4, 256), variance=(count,))
    if set(arrays) != set(shapes):
        raise ValueError('Every frozen context component is required')
    for key, shape in shapes.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Malformed/nonfinite context: '+key)
    if (arrays['variance'] <= 0).any():
        raise ValueError('Positive per-real-axis native variance required')


def complex_iq(value):
    return value[:, 0].astype(np.float64)+1j*value[:, 1].astype(np.float64)


def action(arrays, value):
    """Exact causal FIR with left reflect extension, then main/image mixing."""
    kernel = complex_iq(arrays['kernel'])
    coefficients = np.asarray(arrays['coefficients'], np.float64)
    main = coefficients[:, 0]+1j*coefficients[:, 1]
    image = coefficients[:, 2]+1j*coefficients[:, 3]
    if value.shape != main.shape or len(kernel) != len(value):
        raise ValueError('Batched current-window action only')
    length = kernel.shape[-1]
    nfft = 1 << (value.shape[-1]+2*length-3).bit_length()
    index = np.r_[np.arange(length-1, 0, -1), np.arange(value.shape[-1])]
    full = np.fft.ifft(np.fft.fft(value[:, index], nfft)*np.fft.fft(kernel, nfft))
    filtered = full[:, length-1:length-1+value.shape[-1]]
    return main*filtered+image*filtered.conj()


def diagnostics(estimate, oracle, clean, received, probes):
    count = len(clean)
    validate_estimate(estimate, count)
    validate_estimate(dict(oracle, coarse=clean), count)
    if (clean.shape != (count, 2, 256) or received.shape != clean.shape or
            not np.isfinite(clean).all() or not np.isfinite(received).all() or
            probes.shape != (4, count, 256) or not np.isfinite(probes).all()):
        raise ValueError('Four matched probes and finite paired source IQ required')
    x, y = complex_iq(clean), complex_iq(received)
    truth, estimated = action(oracle, x), action(estimate, x)
    energy = np.maximum(np.mean(abs(x)**2, axis=-1), 2e-4)
    oracle_energy = np.maximum(np.mean(abs(truth)**2, axis=-1), 2e-4)
    probe_nmse = []
    for probe in probes:
        target = action(oracle, probe)
        probe_nmse.append(np.mean(abs(action(estimate, probe)-target)**2, axis=-1)/
                          np.maximum(np.mean(abs(target)**2, axis=-1), 1e-4))
    native = np.asarray(oracle['variance'], np.float64)
    predicted = np.asarray(estimate['variance'], np.float64)
    # |complex residual|^2 / 2 is mean squared error per real IQ coordinate.
    residual = np.mean(abs(y-estimated)**2, axis=-1)/2
    true_residual = np.mean(abs(y-truth)**2, axis=-1)/2
    return dict(coarse_clean_nmse=np.mean(abs(complex_iq(estimate['coarse'])-x)**2, axis=-1)/energy,
                action_clean_nmse=np.mean(abs(estimated-truth)**2, axis=-1)/oracle_energy,
                action_probe_nmse=np.mean(probe_nmse, axis=0),
                log_variance_error=np.log(predicted/native),
                squared_log_variance_error=np.log(predicted/native)**2,
                native_real_variance=native, predicted_real_variance=predicted,
                estimated_action_residual_real_mse=residual,
                oracle_action_residual_real_mse=true_residual,
                estimated_action_residual_over_predicted_variance=residual/predicted,
                oracle_action_residual_over_native_variance=true_residual/native)


def summarize(values, row, view, rx, day):
    count = len(rx)
    if len(day) != count or any(v.shape != (count,) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Complete finite matched source metrics required')
    groups = [('overall', 'all', np.ones(count, bool))]
    groups += [('RX', int(v), rx == v) for v in sorted(set(rx.tolist()))]
    groups += [('day', int(v), day == v) for v in sorted(set(day.tolist()))]
    result = []
    for dimension, group, selected in groups:
        result.append(dict(candidate=row, view=view, dimension=dimension, group=group,
                           count=int(selected.sum()), metrics={k: dict(mean=float(v[selected].mean()),
                           median=float(np.median(v[selected])), p90=float(np.quantile(v[selected], .9)))
                           for k, v in values.items()}))
    return result
