import math

import torch

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map

ARMS = ('continuation', 'prior_anchor')


def stage_details(model, received):
    """Expose both existing priors; leave the deployed graph unchanged."""
    base, action, native = model.estimate_context(received)
    observed, current = complex_iq(received), base
    variance = native[:, None].expand(-1, 256)
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
    outputs, priors = [], []
    for core in model.prior:
        innovation = pair(action.adjoint(observed-action(complex_iq(current))))
        innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
        prior = base+SCALE*core(torch.cat((received, base, current, innovation, logvar), 1))
        current = pair(weighted_map(action, observed, complex_iq(prior), variance,
                                    steps=2, initial=complex_iq(current)))
        priors.append(prior)
        outputs.append(current)
    if len(outputs) != 2:
        raise ValueError('Only the original two-stage DURA graph is allowed')
    return outputs, priors


def spectral_error(estimate, target):
    predicted = complex_iq(estimate)
    desired = complex_iq(target.detach())
    errors = []
    for size in (64, 128, 256):
        window = torch.hann_window(size, periodic=True, device=estimate.device, dtype=estimate.dtype)
        def power(value):
            return torch.fft.fft(value.unfold(-1, size, size//2)*window, norm='ortho').abs().square()
        reference = power(desired)
        scale = reference.mean((1, 2), keepdim=True).clamp_min(1e-4)
        errors.append((torch.log1p(power(predicted)/scale)-torch.log1p(reference/scale)).square().mean((1, 2)))
    return torch.stack(errors).mean(0)


def objective(outputs, priors, teacher, clean, epoch, arm):
    if arm not in ARMS or not 1 <= epoch <= 20:
        raise ValueError('Fixed arm and twenty-epoch schedule required')
    if any(len(values) != 2 for values in (outputs, priors, teacher)):
        raise ValueError('Two full aligned stages required')
    if any(value.shape != clean.shape for value in (*outputs, *priors, *teacher)):
        raise ValueError('Matched paired source IQ required')
    target = clean.detach()
    energy = target.square().mean((1, 2)).clamp_min(1e-4)
    paired = (outputs[-1]-target).square().mean((1, 2))/energy
    fidelity = torch.stack([(a-b.detach()).square().mean((1, 2))/energy for a, b in zip(outputs, teacher)]).mean(0)
    anchor = torch.stack([(prior-target).square().mean((1, 2))/energy for prior in priors]).mean(0)
    spectral = spectral_error(outputs[-1], target)
    teacher_weight = 1-.9*(epoch-1)/19
    total = paired+.05*spectral+teacher_weight*fidelity
    if arm == 'prior_anchor':
        total = total+.1*anchor
    return total, dict(iq_nmse=paired, spectral_error=spectral, teacher_stage_nmse=fidelity, prior_nmse=anchor)


def learning_rate(update):
    if not 0 <= update < 4940:
        raise ValueError('Fixed source update schedule required')
    if update < 247:
        return 2e-4*(update+1)/247
    fraction = (update-247)/(4940-247-1)
    return 2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*fraction))
