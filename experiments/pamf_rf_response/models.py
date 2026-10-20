"""All perturbations are source diagnostic targets, never deployment inputs."""
import math
import numpy as np
import torch
from . import design as d
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_uncertainty_restore.models import weighted_map
from experiments.pamf_restoration.models import complex_iq, pair


def directions(clean, indices, amplitude):
    if (clean.ndim != 3 or clean.shape[1:] != (2, 256) or not torch.isfinite(clean).all()
            or amplitude not in d.AMPLITUDES or len(indices) != len(clean)
            or any(int(i) != i or i < 0 for i in indices)):
        raise ValueError('Finite matched source IQ, original panel indices and registered amplitude required')
    # Construct and project before rounding to the deployed FP32 IQ interface.
    # Otherwise a near-zero residual can become a normalized scalar direction.
    x = complex_iq(clean.to(torch.float64))
    power = x.abs().square()
    delayed = torch.cat((x[:, :1], x[:, :-1]), -1)
    # A fresh generator per panel index makes batching and traversal order irrelevant.
    random = np.stack([np.random.default_rng(np.random.SeedSequence([2026102031, int(i)]))
                       .normal(size=(2, 256)).astype(np.float32) for i in indices])
    random = torch.tensor(random.tolist(), device=x.device, dtype=torch.float64)
    candidates = (x*power, 1j*x*power, x.conj(), delayed, complex_iq(random))
    denominator = power.sum(-1)
    values = {}
    for name, raw in zip(d.DIRECTIONS, candidates):
        projection = (x.conj()*raw).sum(-1)/denominator.clamp_min(1e-20)
        tangent = raw-projection[:, None]*x
        tangent = tangent-((x.conj()*tangent).sum(-1)/denominator.clamp_min(1e-20))[:, None]*x
        energy = tangent.abs().square().sum(-1)
        raw_energy = raw.abs().square().sum(-1)
        valid = (denominator > 1e-12) & (raw_energy > 1e-20) & (energy > 1e-8*raw_energy)
        scale = amplitude*(denominator/energy.clamp_min(1e-20)).sqrt()
        value = torch.where(valid[:, None], tangent*scale[:, None], torch.zeros_like(tangent))
        values[name] = (value.to(complex_iq(clean).dtype), valid)
    return values


def oracle_response(action, observation_delta, variance):
    response = weighted_map(action, observation_delta, torch.zeros_like(observation_delta),
                            variance, steps=d.ORACLE_STEPS)
    weight = variance[:, None].reciprocal()
    rhs = action.adjoint(weight*observation_delta)
    residual = action.adjoint(weight*action(response))+response/SCALE**2-rhs
    relative = (real_dot(residual, residual)/real_dot(rhs, rhs).clamp_min(1e-20)).sqrt()
    return response, relative


def response_metrics(plus, minus, base, delta, known_response, observation_delta, variance, valid, cg_residual):
    if (plus.shape != minus.shape or plus.shape != base.shape or plus.shape != pair(delta).shape
            or variance.shape != (len(plus),) or valid.shape != variance.shape
            or cg_residual.shape != variance.shape or not math.isfinite(SCALE)):
        raise ValueError('Matched finite central-response arrays required')
    tensors = (plus, minus, base, delta, known_response, observation_delta, variance, cg_residual)
    if any(not torch.isfinite(t).all() for t in tensors) or (variance <= 0).any():
        raise ValueError('Nonfinite response or invalid simulator variance')
    response = complex_iq((plus-minus)/2)
    curvature = complex_iq((plus+minus)/2-base)
    energy = real_dot(delta, delta).clamp_min(1e-20)
    gain = real_dot(response, delta)/energy
    model_nmse = real_dot(response-delta, response-delta)/energy
    oracle_nmse = real_dot(known_response-delta, known_response-delta)/energy
    response_energy = real_dot(response, response)
    cosine = real_dot(response, delta)/(response_energy*energy).clamp_min(1e-30).sqrt()
    fisher = real_dot(observation_delta, observation_delta)/variance
    observable = valid & (fisher >= 9.) & (oracle_nmse <= .1) & (cg_residual <= .001)
    return dict(valid=valid, observable=observable, gain=gain, response_nmse=model_nmse,
        response_cosine=cosine, response_energy_ratio=response_energy/energy,
        curvature_nmse=real_dot(curvature, curvature)/energy, oracle_nmse=oracle_nmse,
        oracle_gain=real_dot(known_response, delta)/energy, fisher_energy=fisher,
        oracle_CG_relative_residual=cg_residual)
