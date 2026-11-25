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


def estimate_pair(x, y):
    if np.array_equal(x, y):
        kernel = np.zeros(SHAPES['kernel'], np.float32)
        coefficients = np.zeros(SHAPES['coefficients'], np.float32)
        kernel[0, 0] = 1.
        coefficients[0] = 1.
        return kernel, coefficients, np.float32(1e-8), 0., True, False
    clean = voltages(x)
    signal, complex_variance, _ = choose_and_apply(clean, voltages(y), OPTIONS)
    kernel, main, image, mismatch = project(clean, signal)
    real_variance = complex_variance / 2.
    values = (np.stack((kernel.real, kernel.imag)).astype(np.float32),
              np.stack((main.real, main.imag, image.real, image.imag)).astype(np.float32),
              np.float32(np.clip(real_variance, 1e-8, 1.)))
    if not all(np.isfinite(value).all() for value in (*values, mismatch, real_variance)):
        raise FloatingPointError('Nonfinite source IQ estimate')
    return *values, float(mismatch), False, bool(real_variance < 1e-8 or real_variance > 1.)


def received(source, epoch):
    values = load(source / 'epochs' / f'{epoch:02d}', 'received')
    if values.shape != (COUNT, 2, 256) or values.dtype != np.float32:
        raise ValueError('Original R07 source observation layout changed')
    return values


def init_worker(source):
    global CLEAN, SOURCE, EPOCH, OBSERVED
    SOURCE, EPOCH = Path(source), None
    CLEAN = load(SOURCE / 'train', 'clean')


def block(job):
    global EPOCH, OBSERVED
    epoch, start, end = job
    if EPOCH != epoch:
        OBSERVED, EPOCH = received(SOURCE, epoch), epoch
    return start, [estimate_pair(CLEAN[i], OBSERVED[i]) for i in range(start, end)]


def prepare(rt):
    source, root = map(Path, (rt['action_iq'], rt['run_root']))
    folder = root / 'estimates'
    folder.mkdir()
    configured = read(rt['iq_estimator'])
    if {key: configured.get(key) for key in OPTIONS} != OPTIONS:
        raise ValueError('Frozen Adaptive source configuration changed')
    started = perf_counter()
    context = multiprocessing.get_context('spawn')
    with context.Pool(WORKERS, initializer=init_worker, initargs=(source,)) as pool:
        for epoch in range(1, 21):
            part = folder / f'{epoch:02d}'
            part.mkdir()
            arrays = {key: np.lib.format.open_memmap(part / f'{key}.npy', 'w+', dtype=np.float32,
                       shape=(COUNT, *shape)) for key, shape in SHAPES.items()}
            stats = np.empty((COUNT, 3))
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
            row = dict(status='ESTIMATES_FROZEN', epoch=epoch, count=count, identity=int((~distorted).sum()),
                       variance_clipped=int(stats[:, 2].sum()), variance_unit='per_real_component',
                       projection_relative_error_quantiles=np.quantile(stats[distorted, 0], (.1,.5,.9,.99)).tolist(),
                       simulator_supervision_reads=0, seconds=perf_counter()-started)
            write(part / 'closing.json', row)
            (part / 'closing.json').rename(part / 'complete.json')
            print('R07_ESTIMATE_EPOCH', row, flush=True)
    write(folder / 'complete.json', dict(status='ESTIMATES_FROZEN', epochs=20, simulator_supervision_reads=0))
