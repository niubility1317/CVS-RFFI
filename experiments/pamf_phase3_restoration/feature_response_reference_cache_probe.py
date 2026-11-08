"""Source-only production-G check: cache reference, keep both restored F0 forwards."""
import torch

from . import feature_response_reuse_drift_probe as drift
from .feature_response_loss import feature_response_loss
from .response_joint_prior import response_loss, selected_perturbation

RUN = '20261108-diagnostic-Restore-FeatureReferenceCache-sourceLU-s2026092701-r01'


def cached_readout(model, capture, received, clean, targets, scale, objective, indices, epoch, trace, cache):
    if epoch % 2 != 1 or not torch.equal(trace['received'], received):
        raise ValueError('Only fixed global-prior odd-epoch source cases allowed')
    base, aux, loss, _ = drift.feature.original.control.joint_readout(model, capture, received, clean, targets,
        scale, objective, source_indices=indices, epoch=epoch, priors=None, assignment=None)
    delta, choice = selected_perturbation(clean, indices, epoch)
    kernel, main, image, _ = drift.feature.original.known_parameters(trace)
    action = drift.feature.original.dense_action(kernel, main, image)
    changed = drift.feature.original.inject(received, delta, action)
    altered, altered_aux = model.source_readout(changed, prior=None)
    iq_response = response_loss(base, altered, delta)
    changed_source = cache[torch.arange(len(clean), device=clean.device), choice]
    # Preserve the original order and its independent restored-input F0 graph.
    fbase = capture.features(base)
    fchanged = capture.features(altered)
    response = feature_response_loss(fbase, fchanged, targets, changed_source, scale)
    aux['relative_solve_residual'] = torch.maximum(aux['relative_solve_residual'], altered_aux['relative_solve_residual'])
    return base, aux, loss + drift.feature.original.RESPONSE_WEIGHT*iq_response + drift.feature.FEATURE_RESPONSE_WEIGHT*response


def main():
    drift.RUN = RUN
    drift.shared_readout = cached_readout
    drift.main()


if __name__ == '__main__':
    main()
