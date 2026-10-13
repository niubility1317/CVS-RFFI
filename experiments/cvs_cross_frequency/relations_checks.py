"""Synthetic focused checks for matched power/complex observation encoders."""
import json
import math
import torch
from .relations import (BINS, EPSILON, COMPLEX_RELATIONS, POWER_RELATIONS,
                        CrossFrequencyRelations)


def iq_from_spectrum(spectrum):
    value = torch.fft.ifft(spectrum, dim=-1)
    return torch.stack((value.real, value.imag), dim=1)


def manual_relations(iq, mode):
    # Independent expression of the fixed observation definition.
    x = torch.complex(iq[:, 0], iq[:, 1])
    raw = torch.fft.fft(x, dim=-1)
    power = raw.abs().square()
    normalized = raw / (power.mean(-1, keepdim=True) + 1e-4).sqrt()
    power = normalized.abs().square()
    index = torch.arange(256)
    values = []
    for edge in [(index + 1) % 256, -index % 256, (index + 16) % 256]:
        if mode == COMPLEX_RELATIONS:
            cross = normalized * normalized[:, edge].conj()
            cross = cross / ((power + 1e-4) * (power[:, edge] + 1e-4)).sqrt()
            values += [cross.real, cross.imag]
        else:
            first = (power + 1e-4).log()
            second = (power[:, edge] + 1e-4).log()
            values += [(first - second).tanh(), (.5 * (first + second)).tanh()]
    return torch.stack(values, dim=1)


def checks():
    torch.set_num_threads(2)
    models = {}
    for mode in (COMPLEX_RELATIONS, POWER_RELATIONS):
        torch.manual_seed(9127)
        models[mode] = CrossFrequencyRelations(mode).double().eval()
    left, right = (models[mode].state_dict() for mode in (COMPLEX_RELATIONS, POWER_RELATIONS))
    assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
    count = {mode: sum(p.numel() for p in model.parameters()) for mode, model in models.items()}
    assert set(count.values()) == {45344}
    torch.manual_seed(7153)
    iq = torch.randn(4, 2, 256, dtype=torch.float64)
    z = torch.complex(iq[:, 0], iq[:, 1])
    angles = torch.tensor([.73, -1.1, 2.7, .19], dtype=torch.float64)[:, None]
    rotated_z = z * torch.complex(angles.cos(), angles.sin())
    rotated = torch.stack((rotated_z.real, rotated_z.imag), dim=1)
    records = {}
    for mode, model in models.items():
        actual = model.relation_channels(iq)
        expected = manual_relations(iq, mode)
        assert actual.shape == (4, 6, BINS)
        assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
        channels = model.input_features(iq)
        position = torch.arange(256, dtype=torch.float64) * (2. * math.pi / 256)
        assert channels.shape == (4, 8, 256)
        assert torch.equal(channels[:, :6], actual)
        assert torch.equal(channels[:, 6], position.sin()[None].expand(4, -1))
        assert torch.equal(channels[:, 7], position.cos()[None].expand(4, -1))
        output = model(iq)
        phase_relations = model.relation_channels(rotated)
        phase_output = model(rotated)
        assert output.shape == (4, 160) and torch.isfinite(output).all()
        assert torch.allclose(actual, phase_relations, atol=2e-12, rtol=2e-12)
        assert torch.allclose(output, phase_output, atol=2e-11, rtol=2e-11)
        individual = torch.cat([model(row[None]) for row in iq])
        assert torch.allclose(output, individual, atol=2e-11, rtol=2e-11)
        changed = iq.clone();changed[1:] = torch.randn_like(changed[1:])
        assert torch.equal(output[0], model(changed)[0])
        inputs = iq.clone().requires_grad_(True)
        objective_output = model(inputs)
        objective = (objective_output * torch.randn_like(objective_output)).sum() + objective_output.square().mean()
        objective.backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all() and inputs.grad.norm() > 0
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, name
            gradients[name] = float(parameter.grad.norm())
        tiny = iq * 1e-12
        weak_spectrum = torch.zeros(1, 256, dtype=torch.complex128)
        weak_spectrum[:, 19] = 1.;weak_spectrum[:, 20] = 1e-15;weak_spectrum[:, 235] = 1e-12
        weak = iq_from_spectrum(weak_spectrum)
        for name, value in [('zero', torch.zeros_like(iq)), ('tiny', tiny), ('weak_bins', weak)]:
            assert torch.isfinite(model.relation_channels(value)).all(), name
            assert torch.isfinite(model(value)).all(), name
            value = value.clone().requires_grad_(True)
            gradient = torch.autograd.grad(model(value).square().mean(), value)[0]
            assert torch.isfinite(gradient).all(), name
        records[mode] = dict(parameters=count[mode], math_max_abs_error=float((actual - expected).abs().max()),
                phase_relation_max_abs_error=float((actual - phase_relations).abs().max()),
                phase_output_max_abs_error=float((output - phase_output).detach().abs().max()),
                input_gradient_norm=float(inputs.grad.norm()), parameter_gradient_norms=gradients,
                output_shape=list(output.shape), packet_independent=True, zero_tiny_weak_bins_finite=True)
    # Identical powers with deliberately different per-bin phases distinguish the
    # complex observation, while the matched power control retains the same input.
    magnitude = torch.linspace(.1, 1.5, 256, dtype=torch.float64)[None]
    phase = .071 * torch.arange(256, dtype=torch.float64)[None] + .37 * torch.sin(torch.arange(256, dtype=torch.float64)[None] * .13)
    first = iq_from_spectrum(torch.complex(magnitude, torch.zeros_like(magnitude)))
    second = iq_from_spectrum(torch.complex(magnitude * phase.cos(), magnitude * phase.sin()))
    c1 = models[COMPLEX_RELATIONS].relation_channels(first)
    c2 = models[COMPLEX_RELATIONS].relation_channels(second)
    p1 = models[POWER_RELATIONS].relation_channels(first)
    p2 = models[POWER_RELATIONS].relation_channels(second)
    assert torch.allclose(p1, p2, atol=2e-12, rtol=2e-12)
    assert torch.allclose(models[POWER_RELATIONS](first), models[POWER_RELATIONS](second), atol=2e-11, rtol=2e-11)
    assert (c1 - c2).abs().max() > .1
    assert (models[COMPLEX_RELATIONS](first) - models[COMPLEX_RELATIONS](second)).abs().max() > 1e-4
    # The epsilon deliberately prevents an exact amplitude-invariance claim.
    weak_amplitude = iq * 1e-5
    amplitude_difference = {mode: float((model.relation_channels(weak_amplitude) - model.relation_channels(weak_amplitude * 2)).abs().max()) for mode, model in models.items()}
    assert all(value > 1e-6 for value in amplitude_difference.values())
    # Production FP32 and a FP32 encoder receiving FP64 IQ both remain finite.
    fp32_records = {}
    for mode in (COMPLEX_RELATIONS, POWER_RELATIONS):
        fp32 = CrossFrequencyRelations(mode).eval()
        before = {name: value.clone() for name, value in fp32.state_dict().items()}
        for inputs in (iq.float(), iq):
            output = fp32(inputs)
            assert output.dtype == torch.float32 and torch.isfinite(output).all()
        raw = iq.float()
        complex_raw = torch.complex(raw[:, 0], raw[:, 1])
        rotation = angles.float()
        phase_raw = complex_raw * torch.complex(rotation.cos(), rotation.sin())
        phase_iq = torch.stack((phase_raw.real, phase_raw.imag), dim=1)
        phase_error = (fp32(raw) - fp32(phase_iq)).detach().abs().max()
        assert torch.allclose(fp32(raw), fp32(phase_iq), atol=2e-5, rtol=2e-5)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            autocast_output = fp32(raw)
        assert torch.equal(fp32(raw), autocast_output)
        inputs = raw.clone().requires_grad_(True)
        output = fp32(inputs)
        (output * torch.randn_like(output)).sum().backward()
        assert torch.isfinite(inputs.grad).all() and inputs.grad.norm() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in fp32.parameters())
        assert all(torch.equal(value, fp32.state_dict()[name]) for name, value in before.items())
        fp32_records[mode] = dict(phase_output_max_abs_error=float(phase_error),
                                 finite_nonzero_input_and_parameter_gradients=True,
                                 parameters_buffers_unchanged=True)
    negative_cases = []
    def reject(name, call):
        try:call()
        except ValueError:negative_cases.append(name)
        else:raise AssertionError('Invalid relations accepted: ' + name)
    reject('mode', lambda: CrossFrequencyRelations('COMPLEX_RELATIONS'))
    for name, inputs in [('non_tensor', [[0.]]), ('short_IQ', torch.zeros(2, 2, 128)),
                         ('channel_count', torch.zeros(2, 3, 256)), ('rank', torch.zeros(2, 256)),
                         ('empty_batch', torch.zeros(0, 2, 256)), ('integer', iq.long()),
                         ('half', iq.half()), ('complex_input', z)]:
        reject(name, lambda: fp32(inputs))
    bad = iq.clone();bad[0, 0, 0] = float('nan')
    reject('nan', lambda: fp32(bad))
    bad = iq.clone();bad[0, 0, 0] = float('inf')
    reject('infinity', lambda: fp32(bad))
    return dict(status='PASS', models=records, identical_initialization=True, matched_parameter_count=45344,
                same_power_control_max_abs_difference=float((p1-p2).abs().max()),
                complex_same_power_phase_difference=float((c1-c2).abs().max()),
                approximate_not_exact_amplitude_invariance=amplitude_difference,
                fp32_outer_autocast_stable=True, fp32=fp32_records, negative_cases=negative_cases,
                persistent_query_state=False, attention=False, real_IQ_or_checkpoint_read=False,
                experiments_started=False)


if __name__ == '__main__':
    print(json.dumps(checks(), allow_nan=False))
