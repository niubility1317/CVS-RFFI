import json

import numpy as np
import pytest
import torch

from experiments.pamf_phase3_restoration.covariance_inverse import dense_action
from experiments.pamf_phase3_restoration.frozen_measurement_consistency import (
    RUN,
    apply_cached_action,
    load_completed_folder,
    measurement_metrics,
)


def test_fft_action_matches_existing_dense_real_operator_with_reflection():
    rng = np.random.default_rng(531)
    x = rng.normal(size=(3, 2, 16))
    h = rng.normal(size=(3, 5))+1j*rng.normal(size=(3, 5))
    m = rng.normal(size=(3, 16))+1j*rng.normal(size=(3, 16))
    image = .2*(rng.normal(size=(3, 16))+1j*rng.normal(size=(3, 16)))
    a = dense_action(*[torch.tensor(v.tolist(), dtype=torch.complex128) for v in [h, m, image]])
    target = (a@torch.tensor(x.reshape(3, 32).tolist(), dtype=torch.float64)[:, :, None]).squeeze(-1)
    np.testing.assert_allclose(apply_cached_action(x, h, m, image),
                               np.asarray(target.tolist()).reshape(x.shape), rtol=1e-12, atol=1e-12)


def test_impulse_main_and_image_paths_have_expected_complex_signs():
    x = np.array([[[1., 2., 3.], [.5, -.2, .1]]])
    m = np.full((1, 3), 1+.3j)
    image = np.full((1, 3), .2-.1j)
    z = x[:, 0]+1j*x[:, 1]
    expected = m*z+image*z.conj()
    np.testing.assert_allclose(apply_cached_action(x, np.ones((1, 1), complex), m, image),
                               np.stack((expected.real, expected.imag), 1), atol=1e-14)


def test_action_is_per_record_and_permutation_equivariant():
    rng = np.random.default_rng(632)
    x = rng.normal(size=(4, 2, 16))
    h = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    m = rng.normal(size=(4, 16))+1j*rng.normal(size=(4, 16))
    image = .1*m.conj()
    perm = [2, 0, 3, 1]
    np.testing.assert_allclose(apply_cached_action(x[perm], h[perm], m[perm], image[perm]),
                               apply_cached_action(x, h, m, image)[perm], atol=1e-14)


def test_fitting_observed_noise_is_separate_from_clean_iq_accuracy():
    clean = np.zeros((1, 2, 8)); clean[:, 0] = 1.
    received = clean.copy(); received[:, 0] += .1
    h = np.ones((1, 1), complex); m = np.ones((1, 8), complex); image = np.zeros_like(m)
    variance = np.array([.005])
    reference = measurement_metrics(clean, clean, received, h, m, image, variance)
    noise_fit = measurement_metrics(received, clean, received, h, m, image, variance)
    assert reference['iq_nmse'][0] == 0.
    assert noise_fit['iq_nmse'][0] > reference['iq_nmse'][0]
    assert noise_fit['measurement_mse'][0] < reference['measurement_mse'][0]
    np.testing.assert_allclose(reference['measurement_over_cached_noise_variance'], [1.])


def test_incomplete_source_row_rejected_before_saved_iq_access(tmp_path):
    folder = tmp_path/'runs'/RUN; folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps({'status': 'PENDING'}))
    with pytest.raises(ValueError, match='Completed frozen source V'):
        load_completed_folder(tmp_path)
