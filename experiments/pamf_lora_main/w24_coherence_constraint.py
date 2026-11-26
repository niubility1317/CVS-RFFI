"""Preserve weak complex spectral relationships in the actual W24 readout."""

import torch


BANDS = ((1, 9), (9, 33), (33, 81), (81, 193))


def pair_phase_risk(output_a, output_b, source_a, source_b):
    def descriptor(a, b):
        product = a * b.conj()
        return product / ((a.abs().square() + .002) * (b.abs().square() + .002)).sqrt()

    observed = descriptor(output_a, output_b)
    reference = descriptor(source_a, source_b)
    support = reference.abs()
    loss = (observed - reference).abs().square()
    # Keep each observation equally weighted rather than normalizing once for the batch.
    dims = (-1, -2)
    return ((support * loss).sum(dims) / support.sum(dims).clamp_min(.01)).mean()


def coherence_risks(output, source):
    output, source = (torch.fft.fft(x, norm='ortho') for x in (output, source))
    bins = (torch.fft.fftfreq(output.shape[-1], device=output.device) * output.shape[-1]).round()
    frequency, chirp = [], []
    for low, high in BANDS:
        band = (bins.abs() >= low) & (bins.abs() < high)
        x, ref = output[..., band], source[..., band]
        chirp.append(pair_phase_risk(x[..., 1:, :], x[..., :-1, :], ref[..., 1:, :], ref[..., :-1, :]))
        # Neighboring signed bins stay in the same band and on the same side of DC.
        pair = band[1:] & band[:-1] & (bins[1:] * bins[:-1] > 0)
        frequency.append(pair_phase_risk(output[..., 1:][..., pair], output[..., :-1][..., pair],
                                         source[..., 1:][..., pair], source[..., :-1][..., pair]))
    return torch.stack(frequency).mean(), torch.stack(chirp).mean()


def compatible_gradient(primary, extra):
    """Remove each observation's conflicting component and cap the remaining gradient."""
    dims = tuple(range(1, primary.ndim))
    power = primary.abs().square().sum(dims, keepdim=True)
    interaction = (primary.conj() * extra).real.sum(dims, keepdim=True)
    projected = extra - (interaction.clamp_max(0.) / power.clamp_min(1e-20)) * primary
    relative = power.sqrt() / projected.abs().square().sum(dims, keepdim=True).clamp_min(1e-20).sqrt()
    return projected * relative.clamp_max(1.), (interaction < 0).float().mean()


def constrained_loss(loss, auxiliary, readout, *, observe_only):
    primary = torch.autograd.grad(loss, readout, retain_graph=True)[0]
    extra = torch.autograd.grad(auxiliary, readout, retain_graph=True)[0]
    update, conflict = compatible_gradient(primary, extra)
    if observe_only:
        return loss, conflict
    surrogate = (update.detach().conj() * readout).real.sum()
    return loss + .35 * (auxiliary.detach() + surrogate - surrogate.detach()), conflict
