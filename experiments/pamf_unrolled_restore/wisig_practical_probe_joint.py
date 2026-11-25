"""Reconstruct an offline source anchor from several Practical received views."""

import numpy as np

from . import wisig_practical_probe_sparse as sparse


SAMPLES = sparse.relative.blind.receive.SAMPLES
TIME = np.arange(SAMPLES)
TAPER = np.minimum(1., np.minimum(TIME, SAMPLES - 1 - TIME) / 34.)


def reference(anchors, noise):
    ref = anchors.mean(axis=0)
    design = ref[sparse.relative.SOURCE_INDICES]
    kernels = np.zeros((len(anchors), SAMPLES), dtype=np.complex128)
    kernels[:, sparse.relative.OFFSETS % SAMPLES] = [
        sparse.fit_relative(design, row[sparse.relative.FIT_TIME], variance) for row, variance in zip(anchors, noise)]
    response = np.fft.fft(kernels, axis=-1)
    weight = 1. / noise.clip(1e-6)
    weight /= weight.sum()
    rhs = np.sum(weight[:, None] * response.conj() * np.fft.fft(anchors, axis=-1), axis=0)
    precision = np.sum(weight[:, None] * abs(response)**2, axis=0)
    spectrum = (rhs + .35 * np.fft.fft(ref)) / (precision + .35)
    return ref + TAPER * (np.fft.ifft(spectrum) - ref)


def target_columns(received, prior):
    return sparse.target_columns(received, prior, anchor_builder=reference)
