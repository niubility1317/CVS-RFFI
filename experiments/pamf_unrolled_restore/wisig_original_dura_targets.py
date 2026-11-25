"""Source IQ-only target adapter for the original DURA probe objective."""

from pathlib import Path
import multiprocessing
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .wisig_paired_estimator import voltages
from .wisig_ten_channel_estimator import choose_and_apply


COUNT = 63000
WORKERS = 16
OPTIONS = dict(complexity=4., shrink=0., blend=.2)
SUPERVISION = ('waveform', 'operator', 'fast_mixture')
SHAPES = dict(kernel=(2, 64), coefficients=(4, 256), variance=())
T = np.arange(256)
FIR_INDEX = np.abs(T[:, None] - np.arange(64))
CENTERS = np.r_[T[::8], 255]
WINDOWS = np.exp(-.5 * ((T - CENTERS[:, None]) / 10.)**2)


def load(folder, key):
    return np.load(Path(folder) / f'{key}.npy', mmap_mode='r', allow_pickle=False)


def modulation(filtered, desired):
    conjugate = filtered.conj()
    power = WINDOWS @ abs(filtered)**2
    cross = WINDOWS @ conjugate**2
    power += max(3e-3 * power.mean(), 1e-12)
    direct, image = WINDOWS @ (conjugate * desired), WINDOWS @ (filtered * desired)
    determinant = power**2 - abs(cross)**2
    values = ((power * direct - cross * image) / determinant,
              (power * image - cross.conj() * direct) / determinant)
    return [np.interp(T, CENTERS, v.real) + 1j * np.interp(T, CENTERS, v.imag) for v in values]


def project(clean, desired):
    """Fit the native single shared causal FIR and two smooth modulation curves."""
    delayed = clean[FIR_INDEX]
    blocks = (delayed.conj() * desired[:, None]).reshape(16, 16, 64).sum(axis=1)
    images = (delayed * desired[:, None]).reshape(16, 16, 64).sum(axis=1)
    strength = (abs(blocks)**2 + abs(images)**2).sum(axis=0) / (abs(delayed)**2).sum(axis=0).clip(1e-12)
    support = np.sort(np.argpartition(strength, -12)[-12:])
    design = delayed[:, support]
    filtered = delayed[:, strength.argmax()]
    taps = None
    best = None
    for iteration in range(3):
        main, image = modulation(filtered, desired)
        error = np.linalg.norm(main * filtered + image * filtered.conj() - desired)**2
        if best is None or error < best[0]:
            best = error, main, image, taps
        if iteration == 2:
            break
        direct = main[:, None] * design
        conjugate = image[:, None] * design.conj()
        real_linear = np.column_stack((direct + conjugate, 1j * (direct - conjugate)))
        matrix = np.vstack((real_linear.real, real_linear.imag))
        gram = matrix.T @ matrix
        gram.flat[::len(gram)+1] += max(1e-3 * np.trace(gram) / len(gram), 1e-12)
        fitted = np.linalg.solve(gram, matrix.T @ np.r_[desired.real, desired.imag])
        taps = fitted[:len(support)] + 1j * fitted[len(support):]
        filtered = design @ taps
    error, main, image, best_taps = best
    kernel = np.zeros(64, np.complex128)
    if best_taps is None:
        kernel[strength.argmax()] = 1.
    else:
        # Fix the scale ambiguity between the shared FIR and both curves.
        gain = max(np.linalg.norm(best_taps), 1e-12)
        kernel[support] = best_taps / gain
        main, image = gain * main, gain * image
    return kernel, main, image, error / max(np.linalg.norm(desired)**2, 1e-18)


def project_operator(clean, signal, operator):
    lags, direct, image = operator
    causal = (lags >= 0) & (lags < SHAPES['kernel'][-1])
    lags = lags[causal]
    values = np.concatenate((direct[:, causal], image[:, causal].conj()))
    active = np.any(values != 0., axis=0)
    if not active.any():
        # The selected Adaptive action may contain only lags outside the original causal format.
        return (*project(clean, signal), True)
    matrix = values[:, active]
    _, vectors = np.linalg.eigh(matrix.conj().T @ matrix)
    axis = vectors[:, -1]
    kernel = np.zeros(SHAPES['kernel'][-1], complex)
    kernel[lags[active]] = axis.conj()
    main, image = np.split(matrix @ axis, 2)
    image = image.conj()
    filtered = clean[FIR_INDEX] @ kernel
    error = np.linalg.norm(main*filtered + image*filtered.conj() - signal)**2
    return kernel, main, image, error / max(np.linalg.norm(signal)**2, 1e-18), False


def fast_settings(root):
    root = Path(root)
    if read(root / 'state.json')['status'] != 'ANALYZED':
        raise ValueError('The source-U fitted Practical mixture must already be frozen')
    with np.load(root / 'configuration.npz', allow_pickle=False) as stored:
        factors, anchors = stored['factors'], stored['anchors']
        noise = [stored[f'noise_{i}'] for i in range(len(anchors))]
    return factors, anchors, noise, read(root / 'configuration.json')


def variance_replay(rt):
    root = Path(rt['variance_replay'])
    closure = read(root / 'estimates/complete.json')
    contract = read(root / 'inputs/source_contract.json')
    if (closure.get('status'), closure.get('epochs'), closure.get('simulator_supervision_reads')) != (
            'ESTIMATES_FROZEN', 20, 0) or contract.get('historical_received_epochs') != str(Path(rt['action_iq'])/'epochs'):
        raise ValueError('Expected the existing matched IQ-estimated Adaptive variances')
    return root / 'estimates'


def estimate_pair(x, y, mode='waveform', *, settings=None, replayed_variance=None):
    if mode not in SUPERVISION:
        raise ValueError('Unknown original DURA supervision target')
    if np.array_equal(x, y):
        kernel = np.zeros(SHAPES['kernel'], np.float32)
        coefficients = np.zeros(SHAPES['coefficients'], np.float32)
        kernel[0, 0] = 1.
        coefficients[0] = 1.
        variance = np.float32(1e-8)
        if replayed_variance is not None and replayed_variance != variance:
            raise ValueError('Identity IQ must retain the original variance floor')
        return kernel, coefficients, variance, 0., True, False, False
    clean, observed = voltages(x), voltages(y)
    if mode == 'fast_mixture':
        from .wisig_practical_mixture import estimate_operator

        signal, complex_variance, operator = estimate_operator(clean, observed, *settings)
        if replayed_variance is None:
            raise ValueError('Fast action comparison requires the matched Adaptive variance')
    else:
        signal, complex_variance, operator = choose_and_apply(clean, observed, OPTIONS,
                                                              return_operator=mode == 'operator')
    if mode == 'waveform':
        kernel, main, image, mismatch = project(clean, signal)
        fallback = False
    else:
        kernel, main, image, mismatch, fallback = project_operator(clean, signal, operator)
    real_variance = complex_variance / 2. if replayed_variance is None else float(replayed_variance)
    values = (np.stack((kernel.real, kernel.imag)).astype(np.float32),
              np.stack((main.real, main.imag, image.real, image.imag)).astype(np.float32),
              np.float32(np.clip(real_variance, 1e-8, 1.)))
    if not all(np.isfinite(value).all() for value in (*values, mismatch, real_variance)):
        raise FloatingPointError('Nonfinite source IQ estimate')
    return *values, float(mismatch), False, bool(real_variance < 1e-8 or real_variance > 1.), fallback


def received(source, epoch):
    values = load(source / 'epochs' / f'{epoch:02d}', 'received')
    if values.shape != (COUNT, 2, 256) or values.dtype != np.float32:
        raise ValueError('Original R07 source observation layout changed')
    return values


def init_worker(source, mode, mixture_root, variance_root):
    global CLEAN, SOURCE, EPOCH, OBSERVED, MODE, FAST, VARIANCE_ROOT, VARIANCE
    SOURCE, EPOCH, MODE = Path(source), None, mode
    CLEAN = load(SOURCE / 'train', 'clean')
    FAST = fast_settings(mixture_root) if mixture_root else None
    VARIANCE_ROOT, VARIANCE = variance_root, None


def block(job):
    global EPOCH, OBSERVED, VARIANCE
    epoch, start, end = job
    if EPOCH != epoch:
        OBSERVED, EPOCH = received(SOURCE, epoch), epoch
        if VARIANCE_ROOT is not None:
            VARIANCE = load(VARIANCE_ROOT / f'{epoch:02d}', 'variance')
            if VARIANCE.shape != (COUNT,) or VARIANCE.dtype != np.float32:
                raise ValueError('Adaptive variance source layout differs')
    return start, [estimate_pair(CLEAN[i], OBSERVED[i], MODE, settings=FAST,
                                replayed_variance=None if VARIANCE is None else VARIANCE[i]) for i in range(start, end)]


def prepare(rt):
    source, root = map(Path, (rt['action_iq'], rt['run_root']))
    folder = root / 'estimates'
    folder.mkdir()
    configured = read(rt['iq_estimator'])
    if {key: configured.get(key) for key in OPTIONS} != OPTIONS:
        raise ValueError('Frozen Adaptive source configuration changed')
    mode = rt['action_supervision']
    if mode not in SUPERVISION:
        raise ValueError('Unknown original DURA supervision target')
    fast = mode == 'fast_mixture'
    variance_root = variance_replay(rt) if fast else None
    started = perf_counter()
    context = multiprocessing.get_context('spawn')
    with context.Pool(WORKERS, initializer=init_worker,
                      initargs=(source, mode, rt.get('mixture_root'), variance_root)) as pool:
        for epoch in range(1, 21):
            part = folder / f'{epoch:02d}'
            part.mkdir()
            arrays = {key: np.lib.format.open_memmap(part / f'{key}.npy', 'w+', dtype=np.float32,
                       shape=(COUNT, *shape)) for key, shape in SHAPES.items()}
            stats = np.empty((COUNT, 4))
            jobs = ((epoch, i, min(i + 32, COUNT)) for i in range(0, COUNT, 32))
            count = 0
            for start, rows in pool.imap_unordered(block, jobs):
                to = start + len(rows)
                columns = tuple(zip(*rows))
                for key, values in zip(arrays, columns[:3]):
                    arrays[key][start:to] = np.asarray(values)
                stats[start:to] = np.column_stack(columns[3:])
                count += len(rows)
                if count // 4096 != (count - len(rows)) // 4096:
                    print('R07_ESTIMATE_ROWS', epoch, count, round(perf_counter()-started, 2), flush=True)
            for array in arrays.values():
                array.flush()
            distorted = stats[:, 1] == 0
            clipping = (read(variance_root / f'{epoch:02d}' / 'complete.json')['variance_clipped']
                        if fast else int(stats[:, 2].sum()))
            row = dict(status='ESTIMATES_FROZEN', epoch=epoch, count=count, identity=int((~distorted).sum()),
                       variance_clipped=clipping, variance_unit='per_real_component',
                       action_supervision=mode, operator_fallbacks=int(stats[:, 3].sum()),
                       adaptive_variance_replay=fast,
                       projection_relative_error_quantiles=np.quantile(stats[distorted, 0], (.1,.5,.9,.99)).tolist(),
                       simulator_supervision_reads=0, seconds=perf_counter()-started)
            write(part / 'closing.json', row)
            (part / 'closing.json').rename(part / 'complete.json')
            print('R07_ESTIMATE_EPOCH', row, flush=True)
    write(folder / 'complete.json', dict(status='ESTIMATES_FROZEN', epochs=20, action_supervision=mode,
                                        simulator_supervision_reads=0))
