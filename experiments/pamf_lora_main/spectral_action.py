"""Use held-chirp evidence to bound the existing restoration edit by band."""

import torch

from .action_support import independent_proposals


def band_confidence(a, b, pa, pb, bands):
    """Held-group improvement of the half edit and a conservative best scalar."""
    fft = lambda x: torch.fft.fft(x, norm='ortho')
    gap = fft(b-a)
    energy = (gap.abs().square() @ bands.T).clamp_min(1e-8)
    gains, doses = [], []
    for proposed, before, sign in ((pa,a,1.),(pb,b,-1.)):
        change = fft(proposed-before)
        cross = (sign * (gap.conj()*change).real) @ bands.T
        square = change.abs().square() @ bands.T
        gains.append((cross-.25*square)/energy)
        doses.append((cross / square.clamp_min(1e-10)).clamp(0.,1.))
    return torch.minimum(*gains), torch.minimum(*doses)


def band_support(prior, captured, quality, bands, selectors):
    folds = [(y.mean(1), proposal.mean(1)) for y, proposal in
             independent_proposals(prior, captured, quality, bands, selectors)]
    return band_confidence(folds[0][0], folds[1][0], folds[0][1], folds[1][1], bands)


def band_doses(old, gain, optimum, threshold):
    """Return hard-band, tapered weak-band, and support-limited boost doses."""
    base = .5 * old[:,None].to(gain.dtype)
    trust = (2*(gain-threshold)/(1-threshold).clamp_min(1e-5)).clamp(0.,1.)
    hard = torch.maximum(base,.5*(gain>threshold).to(gain.dtype))
    taper = torch.maximum(base,.5*trust)
    boost = taper + trust * (optimum-.5).clamp_min(0.) * old[:,None]
    return hard,taper,boost

def apply_bands(edit, dose, bands):
    return edit * (dose @ bands)[:, None]
