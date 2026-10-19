"""Shared IQ interface; no TX predictions, channel truth, or scene routing."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_restoration.models import (Block, ConditionalDiffusion,
                                                ConditionalFlow, complex_iq, pair, zero_output)

NAMES = ('unet_residual', 'unet_full', 'unet_mixed', 'tf_unet_mixed',
         'diffusion_mixed', 'flow_mixed')


def mirror_spectrum(spectrum):
    index = (-torch.arange(spectrum.shape[-1], device=spectrum.device)) % spectrum.shape[-1]
    return spectrum.index_select(-1, index).conj()


class Encoder(nn.Module):
    def __init__(self, incoming=2, widths=(32, 64, 96, 128)):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(Block(a, b), Block(b, b))
                                   for a, b in zip((incoming, *widths[:-1]), widths))

    def forward(self, value):
        features = []
        for index, block in enumerate(self.blocks):
            value = block(F.avg_pool1d(value, 2) if index else value)
            features.append(value)
        return features


class Core(nn.Module):
    def __init__(self, incoming=2, outgoing=2, widths=(32, 64, 96, 128), conditional=False):
        super().__init__()
        self.encoder = Encoder(incoming, widths)
        self.conditional = conditional
        if conditional:
            self.condition_fusion = nn.ModuleList(nn.Conv1d(2*w, w, 1) for w in widths)
            self.time_fusion = nn.ModuleList(nn.Linear(64, w) for w in widths)
        self.decode = nn.ModuleList(nn.Sequential(Block(a+b, b), Block(b, b))
                                   for a, b in zip(widths[:0:-1], widths[-2::-1]))
        self.out = zero_output(nn.Conv1d(widths[0], outgoing, 1))

    def forward(self, value, condition=None, time=None):
        features = self.encoder(value)
        if self.conditional:
            if condition is None or time is None or len(condition) != len(features):
                raise ValueError('Current received IQ multiscale condition and time required')
            features = [fusion(torch.cat((feature, context), 1))+projection(time)[:, :, None]
                        for feature, context, fusion, projection in zip(
                            features, condition, self.condition_fusion, self.time_fusion)]
        value = features[-1]
        for block, skip in zip(self.decode, features[-2::-1]):
            value = block(torch.cat((F.interpolate(value, size=skip.shape[-1], mode='linear',
                                                    align_corners=False), skip), 1))
        return self.out(value)


class ReceptionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core()

    def forward(self, received):
        return received+self.core(received)


class ComplexTFUNet(nn.Module):
    """Bounded widely-linear spectral correction, then multiscale IQ decoding.

    The spectral path predicts no free waveform template. Main gain magnitude
    is in [1/4,4]; mirror coefficient magnitude <=sqrt(2)/4. Final IQ remains
    linear at its output (no amplitude clipping), and all PAMF views derive it.
    """
    def __init__(self):
        super().__init__()
        self.spectral = Core(incoming=4, outgoing=4, widths=(24, 48, 72, 96))
        self.temporal = Core(incoming=4)

    def spectral_observation(self, received):
        spectrum = torch.fft.fft(complex_iq(received), norm='ortho')
        mirror = mirror_spectrum(spectrum)
        parameters = self.spectral(torch.cat((pair(spectrum), pair(mirror)), 1))
        main = torch.exp(math.log(4)*parameters[:, 0].tanh())*torch.exp(1j*math.pi*parameters[:, 1].tanh())
        image = .25*torch.complex(parameters[:, 2].tanh(), parameters[:, 3].tanh())
        corrected = pair(torch.fft.ifft(main*spectrum+image*mirror, norm='ortho'))
        return corrected

    def forward(self, received):
        frequency_view = self.spectral_observation(received)
        return received+self.temporal(torch.cat((received, frequency_view), 1))


class MultiscaleDiffusion(ConditionalDiffusion):
    """v-prediction with received features injected at every resolution/step."""
    def __init__(self):
        super().__init__(steps=64, sampling_steps=16)
        self.denoiser = Core(conditional=True)
        self.conditioner = Encoder()
        self.time_encoder = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))

    def field(self, point, condition, time):
        frequency = torch.logspace(0, 3, 16, device=point.device, dtype=point.dtype)
        angle = time.to(point.dtype)[:, None]/self.steps*frequency[None]*math.pi
        embedded = self.time_encoder(torch.cat((angle.sin(), angle.cos()), -1))
        return self.denoiser(point, condition, embedded)

    def velocity(self, noisy, received, time):
        return self.field(noisy, self.conditioner(received), time)

    def training_prediction(self,clean,received,generator=None,return_diagnostics=False):
        time=torch.randint(self.steps,(len(clean),),device=clean.device,generator=generator)
        noise=torch.randn(clean.shape,dtype=clean.dtype,device=clean.device,generator=generator)
        a=self.alpha_bar[time,None,None].sqrt();b=(1-self.alpha_bar[time,None,None]).sqrt()
        noisy=a*clean+b*noise;target=a*noise-b*clean
        velocity=self.field(noisy,self.conditioner(received),time)
        error=(velocity-target).square().mean((1,2))
        values=(a*noisy-b*velocity,error.mean())
        if return_diagnostics:
            return (*values,dict(time_fraction=time.to(clean.dtype)/self.steps,velocity_error=error))
        return values

    def forward(self, received, noise):
        if noise is None or noise.shape != received.shape:
            raise ValueError('One declared noise tensor per current observation required')
        condition = self.conditioner(received)
        point = noise
        indices = torch.linspace(self.steps-1, 0, self.sampling_steps).round().long().tolist()
        for offset, index in enumerate(indices):
            time = torch.full((len(point),), index, dtype=torch.long, device=point.device)
            a, b = self.alpha_bar[index].sqrt(), (1-self.alpha_bar[index]).sqrt()
            velocity = self.field(point, condition, time)
            estimate, epsilon = a*point-b*velocity, b*point+a*velocity
            if offset == len(indices)-1:
                point = estimate
            else:
                following = self.alpha_bar[indices[offset+1]]
                point = following.sqrt()*estimate+(1-following).sqrt()*epsilon
        return point


class MultiscaleFlow(ConditionalFlow):
    def __init__(self):
        super().__init__(sampling_steps=16)
        self.network = MultiscaleDiffusion()

    def training_prediction(self, clean, received, generator=None,return_diagnostics=False):
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        start = received+self.sigma*noise
        time = torch.rand((len(clean),), device=clean.device, generator=generator)
        point = (1-time[:, None, None])*start+time[:, None, None]*clean
        velocity = self.network.field(point, self.network.conditioner(received), time*self.network.steps)
        error=(velocity-(clean-start)).square().mean((1,2))
        values=(point+(1-time[:, None, None])*velocity,error.mean())
        if return_diagnostics:return (*values,dict(time_fraction=time,velocity_error=error))
        return values

    def forward(self, received, noise):
        if noise is None or noise.shape != received.shape:
            raise ValueError('One declared noise tensor per current observation required')
        condition = self.network.conditioner(received)
        point = received+self.sigma*noise
        for index in range(self.sampling_steps):
            time = received.new_full((len(received),), index/self.sampling_steps*self.network.steps)
            point = point+self.network.field(point, condition, time)/self.sampling_steps
        return point


def build(name):
    factories = dict(unet_residual=ReceptionUNet, unet_full=ReceptionUNet,
                     unet_mixed=ReceptionUNet, tf_unet_mixed=ComplexTFUNet,
                     diffusion_mixed=MultiscaleDiffusion, flow_mixed=MultiscaleFlow)
    if name not in factories:
        raise ValueError('Unregistered reception-restoration candidate')
    return factories[name]()
