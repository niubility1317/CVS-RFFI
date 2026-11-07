"""Source-only IQ loss with bounded normalization in four frequency bands."""
import torch

WEIGHT = .25
ENERGY_FLOOR_FRACTION = .01
BANDS = ((0., .125), (.125, .25), (.25, .375), (.375, .501))


def balanced_spectral_loss(output, clean):
    if output.shape != clean.shape or output.ndim != 3 or output.shape[1:] != (2, 256):
        raise ValueError('Matching source IQ windows required')
    target = torch.fft.fft(torch.complex(clean[:, 0],clean[:, 1]),dim=-1,norm='ortho')
    delta = output-clean
    error = torch.fft.fft(torch.complex(delta[:, 0],delta[:, 1]),dim=-1,norm='ortho')
    frequency = torch.fft.fftfreq(256,device=output.device).abs()
    total = target.abs().square().sum(-1).clamp_min(1e-8)
    terms = []
    for low, high in BANDS:
        mask = (frequency >= low) & (frequency < high)
        energy = target[:,mask].abs().square().sum(-1)
        denominator = torch.maximum(energy,ENERGY_FLOOR_FRACTION*total)
        terms.append(error[:,mask].abs().square().sum(-1)/denominator)
    return torch.stack(terms,dim=1).mean()
