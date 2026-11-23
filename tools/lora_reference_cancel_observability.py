"""Measure the implemented LoRa cancellation guard on six synthetic chirps."""

import json

import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.reference_cancel import ReferenceCancel, initialize, pair_spectrum


def unpack(pairs):
    common, contrasted = pairs.unbind(2)
    spectrum = torch.stack((common + contrasted, common - contrasted), dim=2).flatten(1, 2)
    return torch.fft.ifft(spectrum, norm='ortho')


def samples(model, tracker, generator, count, scenario):
    def draw(*shape):
        return torch.randn(shape, dtype=torch.complex64, generator=generator)

    primary = torch.zeros(count, 3, 2, 256, dtype=torch.complex64)
    primary[:, :, 0, 0] = 16.
    clean = .065 * draw(count, 3, 2, 256)
    noise = torch.zeros_like(primary)
    for name in ('near', 'middle'):
        signed = getattr(model, f'{name}_bins')
        refs_a = (signed.sign() * (signed.abs() + 32)) % 256
        refs_b = (signed.sign() * (signed.abs() + 64)) % 256
        signal = .09 * draw(count, 3, 2, len(signed))
        clean[..., signed % 256] += signal
        clean[..., refs_a] += (.65 + .3j) * signal
        if scenario == 'clean':
            continue

        disturbance = .65 * draw(count, 3, 2, len(signed))
        noise[..., refs_b] = disturbance
        if scenario == 'independent':
            disturbance = .65 * draw(count, 3, 2, len(signed))
        if scenario == 'inconsistent':
            orientation = torch.tensor([1., -.5 + .8660254j, -.5 - .8660254j])
            disturbance = disturbance * orientation[None, :, None, None]
        noise[..., signed % 256] = (.7 + .5j) * disturbance + .25 * draw(count, 3, 2, len(signed))

    chirp = tracker.reference.chirp
    source = unpack(primary + clean) * chirp
    epsilon = unpack(noise) * chirp
    return real_iq(source + epsilon), epsilon


def risk(correction, trace, model):
    terms = []
    for name in ('near', 'middle'):
        select = getattr(model, f'{name}_bins') % 256
        noise = trace[..., select]
        before = noise.abs().square().sum((1, 2, 3))
        after = (noise + correction[..., select]).abs().square().sum((1, 2, 3))
        terms.append(torch.stack((before.sum(), after.sum())))
    return torch.stack(terms)


def evaluate(model, tracker, wave, epsilon):
    edition, levels, metric = [], [], []
    for iq, trace in zip(wave.split(32), epsilon.split(32), strict=True):
        measured, rotation, _, rms = coordinates(tracker, iq)
        delta, dose = model.cancellation(measured)
        analytic_noise = pair_spectrum(trace / (rotation * rms))
        metric.append(risk(pair_spectrum(delta), analytic_noise, model))
        levels.append(dose.flatten(1))
        edition.append(complex_iq(iq) + delta * rotation * rms)
    dose = torch.cat(levels)
    before, after = torch.stack(metric).sum(0).unbind(-1)
    return dict(active_fraction=float((dose > 0).any(-1).float().mean()),
                mean_dose=float(dose.mean()), noise_mse_reduction=(1 - after / before.clamp_min(1e-12)).tolist() if before.any() else None,
                any_iq_change=bool((torch.cat(edition) != complex_iq(wave)).any()))


def main():
    torch.set_num_threads(3)
    tracker = PacketRestorer('consensus').eval()
    fixed, guarded = ReferenceCancel(False), ReferenceCancel(True)
    generator = torch.Generator().manual_seed(20261124)
    training, _ = samples(guarded, tracker, generator, 500, 'clean')
    calibration, _ = samples(guarded, tracker, generator, 500, 'clean')
    initialize(guarded, tracker, training, calibration)
    fixed.load_state_dict(guarded.state_dict())
    results = {}
    for case in ('clean', 'correlated', 'independent', 'inconsistent'):
        iq, noise = samples(guarded, tracker, generator, 256, case)
        results[case] = {name:evaluate(model,tracker,iq,noise)
                         for name,model in (('fixed',fixed),('guarded',guarded))}
    assert not results['clean']['guarded']['any_iq_change']
    assert results['correlated']['guarded']['noise_mse_reduction'][1] > .20
    assert results['independent']['guarded']['mean_dose'] < .15
    assert results['inconsistent']['guarded']['mean_dose'] < .10
    print(json.dumps(dict(uses_registered_iq=False, cases=results), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
