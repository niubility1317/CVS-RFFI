"""Independent synthetic leakage checks; no datasets, models or training."""

from __future__ import annotations

import json
import random
import builtins
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.cvs_rff_physics.known_excitation import template, TONES as PUBLIC_TONES, SIGNS as PUBLIC_SIGNS, REFERENCE as PUBLIC_REFERENCE
from . import objective as primitive
from .objective import repetition_residuals


def _fixture(dtype=torch.float64):
    generator = torch.Generator(device="cpu").manual_seed(6143)
    packet = torch.randn(3, 2, 256, generator=generator, dtype=torch.float64) * .2
    gains = torch.tensor([.7+.2j, 1.4-.4j, .3+.9j], dtype=torch.complex128)
    slopes = torch.tensor([.005, -.009, .012], dtype=torch.float64)
    for row, delay in enumerate((0.0, 1.875, 4.3125)):
        reference = torch.from_numpy(template(np.arange(20), delay)).to(torch.complex128)
        waves = []
        for cycle in range(4):
            n = torch.arange(20, dtype=torch.float64)
            perturbation = .03 * (cycle+1) * torch.exp(2j * torch.pi * 3*n/20)
            noise = torch.complex(torch.randn(20, generator=generator, dtype=torch.float64),
                                  torch.randn(20, generator=generator, dtype=torch.float64)) * .008
            waves.append((gains[row]*reference + perturbation + noise) * torch.exp(1j*slopes[row]*(n+20*cycle)))
        received = torch.cat(waves)
        packet[row, 0, 80:160] = received.real
        packet[row, 1, 80:160] = received.imag
    return packet.to(dtype)


def _residual(cycle, reference, gain, valid):
    bins = torch.tensor([14,15,16,17,18,19,1,2,3,4,5,6])
    ideal = torch.fft.fft(reference)[:, bins]
    safe_gain = torch.where(valid, gain, torch.ones_like(gain))
    relative = torch.fft.fft(cycle)[:, bins] / ideal / safe_gain[:, None] - 1
    relative = torch.where(valid[:, None], relative, torch.zeros_like(relative))
    return torch.cat((relative.real, relative.imag), dim=1)


def _raises(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected malformed input rejection")


def run_checks():
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    packet = _fixture()
    original = repetition_residuals(packet)
    assert bool(original["valid"].all())
    assert primitive.TONES == tuple(PUBLIC_TONES.tolist())
    assert primitive.SIGNS == tuple(PUBLIC_SIGNS.tolist())
    assert primitive.REFERENCE == PUBLIC_REFERENCE
    # Load and execute the primitive as a standalone module with NumPy and
    # project-runtime imports forbidden. Torch is already the installed runtime.
    native_import = builtins.__import__
    def no_numpy_or_project_import(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy.") or name.startswith("experiments."):
            raise ImportError(f"forbidden primitive dependency: {name}")
        return native_import(name, *args, **kwargs)
    standalone = types.ModuleType("repetition_primitive_standalone")
    source = Path(primitive.__file__).read_text(encoding="utf-8")
    with patch("builtins.__import__", no_numpy_or_project_import):
        exec(compile(source, str(primitive.__file__), "exec"), standalone.__dict__)
        standalone_result = standalone.repetition_residuals(packet)
    assert all(torch.equal(value, standalone_result["predictions"][key]) for key, value in original["predictions"].items())
    assert bool((original["masked_input"][:,:,120:140] == 0).all())
    assert bool((original["masked_input"][:,:,:80] == 0).all())
    assert bool((original["masked_input"][:,:,160:] == 0).all())
    changed = packet.clone()
    changed[:, :, 120:140] = torch.linspace(-8, 9, 20).expand(3, 2, 20)
    alternative = repetition_residuals(changed)
    assert torch.equal(original["valid"], alternative["valid"])
    for key, value in original["calibration"].items():
        assert torch.equal(value, alternative["calibration"][key]), key
    for key, value in original["predictions"].items():
        assert torch.equal(value, alternative["predictions"][key]), key
    assert torch.equal(original["masked_input"], alternative["masked_input"])
    assert not torch.equal(original["target"], alternative["target"])
    assert not original["target"].requires_grad
    # Neither outside-window distractors nor the hidden cycle enter calibration.
    outside = packet.clone(); outside[:, :, :80] = 100; outside[:, :, 160:] = -70
    outside_result = repetition_residuals(outside)
    assert torch.equal(outside_result["masked_input"], original["masked_input"])
    assert all(torch.equal(value, outside_result["predictions"][key]) for key, value in original["predictions"].items())
    # The native cache's full-packet scalar RMS may change after replacing the
    # hidden period, but visible homogeneous normalization cancels that scalar.
    def native_rms(value):
        power = (value[:,0].square()+value[:,1].square()).mean(1)
        return value/(power+1e-12).sqrt()[:,None,None]
    cache_original = repetition_residuals(native_rms(packet))
    cache_changed = repetition_residuals(native_rms(changed))
    assert torch.allclose(cache_original["calibration"]["visible_normalized"],cache_changed["calibration"]["visible_normalized"],atol=1e-12,rtol=0)
    assert torch.allclose(cache_original["masked_input"],cache_changed["masked_input"],atol=1e-12,rtol=0)
    assert all(torch.allclose(value,cache_changed["predictions"][key],atol=1e-12,rtol=0) for key,value in cache_original["predictions"].items())
    assert not torch.allclose(cache_original["target"],cache_changed["target"])
    live = packet.clone().requires_grad_()
    live_result = repetition_residuals(live)
    derivative_probes = []
    for key, value in live_result["calibration"].items():
        if value.requires_grad:
            derivative_probes.append((f"calibration:{key}", value.real.sum() + value.imag.sum() if value.is_complex() else value.sum()))
    for key, value in live_result["predictions"].items():
        derivative_probes.append((f"prediction:{key}", value.square().sum()))
    derivative_probes.append(("masked_input", live_result["masked_input"].square().sum()))
    for key, value in derivative_probes:
        derivative = torch.autograd.grad(value, live, retain_graph=True)[0]
        assert bool(torch.isfinite(derivative).all()), key
        assert bool((derivative[:, :, 120:140] == 0).all()), key
        assert bool((derivative[:, :, :80] == 0).all()) and bool((derivative[:, :, 160:] == 0).all()), key
    # Arbitrary positive common scales, including numerically tiny amplitudes.
    maximum_scale_error = 0.0
    for scale in (1e-150, 1e-20, .3, 7.0, 1e150):
        scaled = repetition_residuals(packet * scale)
        assert torch.equal(scaled["valid"], original["valid"])
        assert torch.allclose(scaled["calibration"]["visible_normalized"], original["calibration"]["visible_normalized"], atol=1e-12, rtol=0)
        for key, value in original["predictions"].items():
            error = float((value - scaled["predictions"][key]).abs().max().item())
            maximum_scale_error = max(maximum_scale_error, error)
            assert error < 1e-11, (scale, key, error)
    for scale in (1e-20, 3.0, 100.0):
        base32 = repetition_residuals(packet.to(torch.float32))
        scaled32 = repetition_residuals(packet.to(torch.float32) * scale)
        assert torch.equal(base32["valid"], scaled32["valid"])
        assert all(torch.allclose(value, scaled32["predictions"][key], atol=2e-6, rtol=0) for key, value in base32["predictions"].items())
    # Independent closed forms for visible normalization, slope, LS gain, and
    # each baseline under the returned public template and frozen calibration.
    cal = original["calibration"]
    raw_cycles = torch.complex(packet[:,0,80:160], packet[:,1,80:160]).reshape(3,4,20)
    visible = raw_cycles[:, [0,1,3]]
    expected_normalized = visible / visible.abs().square().mean((1,2)).sqrt()[:,None,None]
    assert torch.allclose(cal["visible_normalized"], expected_normalized, atol=1e-14, rtol=0)
    expected_slope = (expected_normalized[:,0].conj()*expected_normalized[:,1]).sum(1).angle()/20
    assert torch.allclose(cal["phase_slope"], expected_slope, atol=1e-14, rtol=0)
    positions = torch.tensor([[0,20,60]], dtype=torch.float64)[:,:,None]+torch.arange(20,dtype=torch.float64)[None,None,:]
    corrected = expected_normalized*torch.exp(-1j*expected_slope[:,None,None]*positions)
    mean = corrected.mean(1)
    public_bank = torch.from_numpy(np.stack([template(np.arange(20),index/16) for index in range(320)])).to(torch.complex128)
    assert torch.allclose(cal["reference"],public_bank[cal["reference_index"]],atol=1e-13,rtol=0)
    all_correlations = (mean@public_bank.conj().T).abs().square()
    chosen_correlation = all_correlations.gather(1,cal["reference_index"][:,None]).squeeze(1)
    assert bool((chosen_correlation >= all_correlations.max(1).values-1e-12).all())
    expected_gain = (cal["reference"].conj()*mean).sum(1)/cal["reference"].abs().square().sum(1)
    assert torch.allclose(expected_gain, cal["gain"], atol=1e-14, rtol=0)
    expected_copy = _residual(corrected[:,1], cal["reference"], expected_gain, original["valid"])
    expected_mean = _residual(mean, cal["reference"], expected_gain, original["valid"])
    expected_reference = _residual(expected_gain[:,None]*cal["reference"], cal["reference"], expected_gain, original["valid"])
    for key, expected in (("copy_cycle1",expected_copy),("visible_mean",expected_mean),("known_reference",expected_reference)):
        assert torch.allclose(original["predictions"][key], expected, atol=1e-13, rtol=0), key
    assert bool((original["predictions"]["known_reference"].abs() < 1e-13).all())
    raw_hold = raw_cycles[:,2] / visible.abs().square().mean((1,2)).sqrt()[:,None]
    corrected_hold = raw_hold*torch.exp(-1j*expected_slope[:,None]*(40+torch.arange(20,dtype=torch.float64)))
    expected_target = _residual(corrected_hold,cal["reference"],expected_gain,original["valid"])
    assert torch.allclose(original["target"],expected_target,atol=1e-13,rtol=0)
    # Zero visible input stays invalid even if its hidden period has energy.
    zero = torch.zeros(2,2,256,dtype=torch.float64,requires_grad=True)
    zero_result = repetition_residuals(zero)
    hidden_energy = zero.detach().clone(); hidden_energy[:,:,120:140] = 5
    hidden_result = repetition_residuals(hidden_energy)
    assert not bool(zero_result["valid"].any()) and not bool(hidden_result["valid"].any())
    assert all(bool((value == 0).all()) for value in zero_result["predictions"].values())
    assert bool((zero_result["target"] == 0).all())
    zero_gradient = torch.autograd.grad(zero_result["masked_input"].sum()+sum(value.sum() for value in zero_result["predictions"].values()),zero)[0]
    assert bool(torch.isfinite(zero_gradient).all())
    flat = torch.ones(1,2,256,dtype=torch.float64)
    flat_result = repetition_residuals(flat)
    assert not bool(flat_result["valid"].any())
    empty = repetition_residuals(torch.empty(0,2,256,dtype=torch.float64))
    assert empty["target"].shape == (0,24) and empty["valid"].shape == (0,)
    _raises(lambda: repetition_residuals(torch.zeros(2,256)))
    _raises(lambda: repetition_residuals(torch.zeros(1,2,255)))
    _raises(lambda: repetition_residuals(torch.zeros(1,2,256,dtype=torch.int64)))
    invalid=packet.clone();invalid[0,0,120]=float('nan')
    _raises(lambda: repetition_residuals(invalid))
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(),torch_rng)
    final_numpy_rng = np.random.get_state()
    assert numpy_rng[0] == final_numpy_rng[0] and np.array_equal(numpy_rng[1],final_numpy_rng[1]) and numpy_rng[2:] == final_numpy_rng[2:]
    json.dumps(original["summary"],allow_nan=False)
    return {"status":"VERIFIED","environment":{"torch":torch.__version__,"device":"cpu","dtypes":["float64","float32"]},
            "checks":{"heldout_replacement_only_target_changes":True,"heldout_and_outside_derivatives_zero":True,
                      "masked_holdout_and_outside_exact_zero":True,"standalone_no_numpy_or_project_runtime_dependency":True,
                      "positive_global_scale_invariant":True,"zero_and_low_amplitude_explicit_validity":True,
                      "copy_mean_reference_closed_forms":True,"fixed_visible_only_calibration":True,
                      "public_bank_visible_only_routing":True,"native_fullpacket_RMS_scalar_cancels":True,
                      "no_RNG_or_new_parameters":True,"malformed_input_rejection":True},
            "derivative_probe_count":len(derivative_probes),"maximum_FP64_scale_prediction_error":maximum_scale_error,
            "boundary":"cached equalized IQ only; upstream equalization unknown; no raw-noise-independence or TX-identifiability claim"}


if __name__ == "__main__":
    print(json.dumps(run_checks(),ensure_ascii=False,allow_nan=False,indent=2))
