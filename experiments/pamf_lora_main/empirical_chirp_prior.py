"""Complex posterior from clean source templates and disjoint repeated chirps."""

import torch
from torch import nn

from .weak_signal_probe import observed_noise_rms


NEIGHBORS = 16


class EmpiricalChirpPrior(nn.Module):
    def __init__(self, library):
        super().__init__()
        self.register_buffer('library', library)
        k = torch.fft.fftfreq(library.shape[-1], device=library.device).abs()
        self.register_buffer('matching', (k>0)&(k<=1/8), persistent=False)
        self.register_buffer('noncarrier', k>0, persistent=False)
        for name, choice in (('all',slice(None)),('even',slice(0,None,2)),('odd',slice(1,None,2))):
            key = self.key(library[:,choice])
            self.register_buffer(f'key_{name}', key, persistent=False)
            self.register_buffer(f'energy_{name}', key.square().mean(-1), persistent=False)

    def key(self, spectrum):
        return torch.view_as_real(spectrum[...,self.matching].mean(1)).flatten(1)

    def posterior(self, spectrum, noise, name):
        query = self.key(spectrum)
        key = getattr(self,f'key_{name}')
        energy = getattr(self,f'energy_{name}')
        distance = query.square().mean(-1,keepdim=True)+energy-2*(query@key.T)/query.shape[-1]
        errors, ids = distance.topk(NEIGHBORS,largest=False,sorted=True)
        temperature = (noise.mean(1)/spectrum.shape[1]).clamp_min(.002)
        weights = (-(errors-errors[:, :1])/temperature).softmax(dim=-1)[:, :, None, None]
        donors = self.library[ids]
        center = (weights*donors).sum(1)
        spread = (weights*(donors-center[:,None]).abs().square()).sum(1)
        return center, spread

    def forward(self, observed):
        spectrum = torch.fft.fft(observed,norm='ortho')
        noise = observed_noise_rms(observed).square()
        common, common_spread = self.posterior(spectrum,noise,'all')
        even, even_spread = self.posterior(spectrum[:,::2], noise[:,::2],'even')
        odd, odd_spread = self.posterior(spectrum[:,1::2], noise[:,1::2],'odd')
        use_odd = (torch.arange(spectrum.shape[1], device=spectrum.device)%2 == 0)[None,:,None]
        cross = torch.where(use_odd,odd,even)
        cross_spread = torch.where(use_odd,odd_spread,even_spread)
        uncertainty = noise+even_spread+odd_spread+.002
        confidence = torch.exp(-(even-odd).abs().square()/uncertainty)

        def correction(prior, variance):
            strength = noise/(noise+variance+.002)
            return self.noncarrier*strength*(prior-spectrum)

        complete = correction(common,common_spread)
        held_out = correction(cross,cross_spread)
        deltas = (.5*complete,.5*held_out,.5*confidence*held_out,confidence*held_out)
        return tuple(observed+torch.fft.ifft(delta,norm='ortho') for delta in deltas)
