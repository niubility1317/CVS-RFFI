"""Joint time and frequency packet context for each DURA prior."""

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_restoration.models import complex_iq, pair


TOKENS, WIDTH = 32, 48


def encoder():
    return nn.Sequential(
        nn.Conv1d(9, WIDTH, 5, padding=2),
        nn.SiLU(),
        nn.Conv1d(WIDTH, WIDTH, 5, padding=2, groups=WIDTH),
        nn.SiLU(),
    )


def decoder():
    result = nn.Sequential(nn.Conv1d(2*WIDTH, WIDTH, 1), nn.SiLU(), nn.Conv1d(WIDTH, 2, 3, padding=1))
    nn.init.zeros_(result[-1].weight)
    nn.init.zeros_(result[-1].bias)
    return result


class PacketContextPrior(nn.Module):
    """Keep high-resolution residuals and exchange information through packet tokens."""

    def __init__(self, *, global_context):
        super().__init__()
        self.time_encoder, self.frequency_encoder = encoder(), encoder()
        self.time_pool = nn.Conv1d(2*WIDTH, WIDTH, 1)
        self.frequency_pool = nn.Conv1d(2*WIDTH, WIDTH, 1)
        self.positions = nn.Parameter(torch.empty(1,2*TOKENS,WIDTH))
        nn.init.normal_(self.positions,std=.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(WIDTH,4,dim_feedforward=2*WIDTH,dropout=0.,
                                       activation='gelu',batch_first=True,norm_first=True)
            for _ in range(2)
        )
        group = (torch.arange(TOKENS)//8).repeat(2)
        mask = None if global_context else group[:,None]!=group[None,:]
        self.register_buffer('attention_mask',mask,persistent=False)
        self.time_decoder, self.frequency_decoder = decoder(), decoder()

    def mix(self, tokens):
        value = tokens + self.positions
        for layer in self.layers:
            value = layer(value,src_mask=self.attention_mask)
        return value

    def readout_weights(self):
        return self.time_decoder[-1].weight, self.frequency_decoder[-1].weight

    @staticmethod
    def pool(activations, mapping):
        mean = F.adaptive_avg_pool1d(activations,TOKENS)
        energy = F.adaptive_avg_pool1d(activations.square(),TOKENS).clamp_min(1e-8).sqrt()
        return mapping(torch.cat((mean,energy),dim=1)).transpose(1,2)

    @staticmethod
    def expand(tokens, local):
        context = F.interpolate(tokens.transpose(1,2),size=local.shape[-1],mode='linear',align_corners=False)
        return torch.cat((local,context),dim=1)

    def forward(self, inputs):
        quadratures = inputs[:,:8].unflatten(1,(4,2))
        wave = torch.complex(quadratures[:,:,0],quadratures[:,:,1])
        spectrum = torch.fft.fftshift(torch.fft.fft(wave,norm='ortho'),dim=-1)
        spectral_inputs = torch.stack((spectrum.real,spectrum.imag),dim=2).flatten(1,2)
        time = self.time_encoder(inputs.asinh())
        frequency = self.frequency_encoder(torch.cat((spectral_inputs,inputs[:,8:]),dim=1).asinh())
        context = self.mix(torch.cat((self.pool(time,self.time_pool),self.pool(frequency,self.frequency_pool)),dim=1))
        time_context, frequency_context = context.split(TOKENS,dim=1)
        temporal_delta = self.time_decoder(self.expand(time_context,time))
        spectral_delta = complex_iq(self.frequency_decoder(self.expand(frequency_context,frequency)))
        inverse = torch.fft.ifft(torch.fft.ifftshift(spectral_delta,dim=-1),norm='ortho')
        return temporal_delta + pair(inverse)
