"""F-free source waveform relation regularizer from fixed complex moments."""
import torch

LAGS = (1, 2, 4, 8, 16)
MOMENT_WEIGHT = .1


def moment_features(iq):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Source IQ windows [batch, 2, 256] required')
    z = torch.complex(iq[:, 0], iq[:, 1])
    z = z/z.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    power = z.abs().square()
    nonlinear = z * (power/(1+power))
    values = []
    for lag in LAGS:
        ref = z[:, :-lag].conj()
        values.extend(((z[:, lag:]*ref).mean(-1), (nonlinear[:, lag:]*ref).mean(-1)))
    moments = torch.stack(values, dim=1)
    return torch.cat((moments.real, moments.imag), dim=1)


def moment_relational_loss(restored, clean):
    if restored.shape != clean.shape or len(restored) == 0:
        raise ValueError('Matching nonempty source IQ pairs required')
    error = moment_features(restored)-moment_features(clean).detach()
    relative = error[:, None, :]-error[None, :, :]
    return relative.square().mean() + .25*error.square().mean()
