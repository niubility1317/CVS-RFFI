import numpy as np

from experiments.pamf_lora_main.w24_native_aperture_factor import (
    CENTER, MODES, center_rms, equal_direction, hybrids, model_iq, noise_measures, power,
)


def waves():
    rng = np.random.default_rng(54)
    fields = [rng.standard_normal((3, 8, 432), dtype=np.float32).astype(np.complex64)
              for _ in range(4)]
    y, c, b, x = fields
    x[0] = y[0]
    return y, c, b, x


def test_equal_direction_preserves_per_region_energy_and_matches_oracle_direction():
    y, c, _, x = waves()
    for region in (slice(0, CENTER), slice(CENTER, None)):
        obs, base, exact = (v[..., region] for v in (y, c, x))
        matched = equal_direction(obs, base, exact)
        np.testing.assert_allclose(power(matched - obs), power(base - obs), rtol=1e-6)
        np.testing.assert_array_equal(matched[0], base[0])
        measured = matched - obs
        ideal = exact - obs
        dot = (measured * ideal.conj()).real.sum((1, 2), keepdims=True)
        np.testing.assert_allclose(dot[1:], np.sqrt(power(ideal) * power(measured))[1:], rtol=1e-6)


def test_native_hybrids_keep_endpoint_and_normalization_contracts():
    y, c, b, x = waves()
    variants = dict(zip(MODES, hybrids(y, c, b, x), strict=True))
    for name, aperture in (('direct', y), ('control_half', c), ('band_half', b)):
        np.testing.assert_array_equal(variants[name], model_iq(aperture, center_rms(aperture)))
    np.testing.assert_array_equal(variants['band_center'][..., :CENTER], variants['band_half'][..., :CENTER])
    np.testing.assert_array_equal(variants['band_edges'][..., :CENTER], variants['control_half'][..., :CENTER])
    np.testing.assert_array_equal(variants['band_edges'][..., CENTER:], model_iq(b, center_rms(c))[..., CENTER:])
    for name in MODES[5:]:
        np.testing.assert_array_equal(variants[name][0], variants['control_half'][0])
    np.testing.assert_array_equal(variants['exact_both'][1:], model_iq(x[1:], center_rms(x[1:])))
    measured = noise_measures(y, c, b, x)
    assert measured.shape == (3, 12) and np.isfinite(measured).all()
