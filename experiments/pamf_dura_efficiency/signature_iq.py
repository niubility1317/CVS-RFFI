"""Exact frozen time-signature prefix and bounded SOURCE-only IQ probes."""
import torch


class _TimeFeatureReady(Exception):
    def __init__(self, value):
        self.value = value


def signature_features(reader, iq):
    """Stop the actual original F at t2; preserve its input-gradient graph.

    SIGTimeCVS captures this same module output before the native mixstyle
    call. Its finally block also removes its own hook when the prefix exits.
    No native backbone operations are copied or approximated here.
    """
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Frozen eval original F required')
    def stop(module, args, value):
        raise _TimeFeatureReady(value)
    handle = reader.core.id_backbone.t2.register_forward_hook(stop)
    try:
        try:
            reader.features(iq)
        except _TimeFeatureReady as captured:
            local = captured.value
        else:
            raise ValueError('Original F t2 prefix was bypassed')
    finally:
        handle.remove()
    result = reader.sequence_pool(local)
    if result.shape != (len(iq), 160):
        raise ValueError('Original weighted 160D TripleSIG output required')
    return result


def project(anchor, displacement, radius=.01):
    anchor_rms = anchor.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    rms = displacement.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    return anchor + displacement * (radius * anchor_rms / rms).clamp_max(1.)


def signature_steps(reader, anchor, reference):
    """Four fixed 0.25%-RMS steps, final only, projected to a 1% ball.

    Paired clean source signatures are oracle training-feasibility references,
    unavailable at deployment. There is no class/pseudo-class loss or G update.
    """
    value = anchor.detach().clone()
    anchor_rms = anchor.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    history = []
    for _ in range(4):
        value.requires_grad_(True)
        signature = signature_features(reader, value)
        loss = (signature-reference).square().sum(1) / reference.square().sum(1).clamp_min(1e-4)
        gradient = torch.autograd.grad(loss.sum(), value)[0]
        if not torch.isfinite(loss).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite source signature IQ derivative')
        history.append(loss.detach())
        rms = gradient.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
        with torch.no_grad():
            proposal = value - .0025 * anchor_rms * gradient / rms
            value = project(anchor, proposal-anchor)
    return value.detach(), torch.stack(history)
