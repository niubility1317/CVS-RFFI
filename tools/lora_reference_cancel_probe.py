"""Check the LoRa leakage guard on a synthetic complex reference model."""

import numpy as np


def projection(clean_reference, clean_band):
    # Project away the largest source-signal direction shared with the reference.
    cross = clean_reference.conj().T @ clean_band
    direction = np.linalg.svd(cross, full_matrices=False)[0][:, :1]
    return np.eye(clean_reference.shape[-1]) - direction @ direction.conj().T


def fit_reference(observed, reference):
    power = reference.conj().T @ reference / len(reference)
    cross = reference.conj().T @ observed / len(reference)
    ridge = 0.01 * power.trace().real / len(power)
    return np.linalg.solve(power + (ridge + 1e-9) * np.eye(len(power)), cross)


def guarded_dose(observed, correction, source_power_bound):
    magnitude = np.mean(np.sum(np.abs(correction) ** 2, axis=-1))
    correlation = np.vdot(observed, correction).real / len(observed)
    margin = 0.025 * (magnitude + source_power_bound)
    bound = correlation - np.sqrt(source_power_bound * magnitude) - margin
    return float(np.clip(bound / (magnitude + 1e-9), 0.0, 1.0))


def simulate(generator, size, interference_gain):
    def sample(count):
        shape = (size, count)
        return (generator.standard_normal(shape) + 1j * generator.standard_normal(shape)) / np.sqrt(2)

    source = sample(1)
    interference = sample(1)
    band = 0.3 * source
    clean_reference = np.concatenate((source, np.zeros_like(source), np.zeros_like(source)), axis=1)
    band_noise = interference_gain * interference + 0.48 * sample(1)
    reference = clean_reference + np.concatenate(
        (0.12 * sample(1), interference + 0.22 * sample(1), sample(1)), axis=1
    )
    return band, clean_reference, band + band_noise, reference


def evaluate(generator, interference_gain, projector, source_power_bound):
    _, _, fit_y, fit_w = simulate(generator, 2048, interference_gain)
    _, _, gate_y, gate_w = simulate(generator, 2048, interference_gain)
    truth, clean_w, test_y, test_w = simulate(generator, 4096, interference_gain)
    unprotected = fit_reference(fit_y, fit_w)
    protected = fit_reference(fit_y, fit_w @ projector)
    dose = guarded_dose(gate_y, gate_w @ projector @ protected, source_power_bound)
    corrected = test_y - dose * (test_w @ projector @ protected)
    original_risk = np.mean(np.abs(test_y - truth) ** 2)
    improvement = 1 - np.mean(np.abs(corrected - truth) ** 2) / original_risk
    raw_leakage = np.linalg.norm(clean_w @ unprotected) / np.linalg.norm(truth)
    survived_leakage = np.linalg.norm(clean_w @ projector @ protected) / np.linalg.norm(truth)
    return dose, float(improvement), float(raw_leakage), float(survived_leakage)


def main():
    generator = np.random.default_rng(20261124)
    band, reference, _, _ = simulate(generator, 4096, 0.0)
    projector = projection(reference, band)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12, rtol=0)
    np.testing.assert_allclose(reference @ projector, 0, atol=1e-12, rtol=0)
    assert guarded_dose(band, band, np.mean(np.abs(band) ** 2)) == 0
    source_power_bound = 1.2 * np.mean(np.sum(np.abs(band) ** 2, axis=-1))

    correlated = evaluate(generator, 1.25 * np.exp(0.7j), projector, source_power_bound)
    independent = evaluate(generator, 0.0, projector, source_power_bound)
    assert correlated[0] > 0.5 and correlated[1] > 0.55
    assert independent[0] == 0 and independent[1] == 0
    assert min(correlated[2], independent[2]) > 0.8
    assert correlated[3] < 1e-12 and independent[3] < 1e-12

    print("scenario dose noise_mse_reduction unconstrained_clean_leakage protected_clean_leakage")
    for name, values in (("correlated", correlated), ("independent", independent)):
        print(name, *(f"{value:.6f}" for value in values))
    print("PASS: projection, pure-signal rejection, and independent-noise bypass")


if __name__ == "__main__":
    main()
