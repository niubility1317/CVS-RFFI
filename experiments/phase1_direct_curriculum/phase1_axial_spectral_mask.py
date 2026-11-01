"""Direct IQ restoration with temporal convolutions and axial spectral attention."""
from __future__ import annotations
import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(channels,channels,5,padding=2*dilation,dilation=dilation),nn.GELU(),nn.Conv1d(channels,channels,1))
    def forward(self,x):
        return x+self.net(x)*.25


class AxialSpectralMask(nn.Module):
    """Identity-initialized direct raw-IQ network with long temporal context."""
    def __init__(self,channels=48,heads=4,temporal_channels=96):
        super().__init__()
        self.temporal_input=nn.Conv1d(2,temporal_channels,9,padding=4)
        self.temporal_blocks=nn.Sequential(*(TemporalBlock(temporal_channels,d) for d in (1,2,4,8,16,32,64)))
        self.temporal_output=nn.Conv1d(temporal_channels,2,1)
        self.input=nn.Conv2d(5,channels,3,padding=1)
        self.local=nn.Sequential(nn.GELU(),nn.Conv2d(channels,channels,3,padding=1),nn.GELU())
        self.time_norm=nn.LayerNorm(channels);self.freq_norm=nn.LayerNorm(channels)
        self.time_attention=nn.MultiheadAttention(channels,heads,dropout=0.,batch_first=True)
        self.freq_attention=nn.MultiheadAttention(channels,heads,dropout=0.,batch_first=True)
        self.output=nn.Conv2d(channels,2,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.temporal_output.weight);nn.init.zeros_(self.temporal_output.bias)
        self.register_buffer('window32',torch.hann_window(32),persistent=False)
        self.register_buffer('window64',torch.hann_window(64),persistent=False)

    def _filter(self,raw,proposal,n_fft):
        window=getattr(self,f'window{n_fft}').to(dtype=raw.real.dtype);hop=n_fft//4
        z=torch.stft(raw,n_fft=n_fft,hop_length=hop,window=window,center=True,onesided=False,return_complex=True)
        p=torch.stft(proposal,n_fft=n_fft,hop_length=hop,window=window,center=True,onesided=False,return_complex=True)
        norm=float(n_fft)**.5
        features=torch.stack((z.real/norm,z.imag/norm,p.real/norm,p.imag/norm,(p-z).abs()/norm),dim=1)
        h=self.local(self.input(features));batch,channels,freq,time=h.shape
        temporal=h.permute(0,2,3,1).reshape(batch*freq,time,channels)
        q=self.time_norm(temporal);temporal=temporal+self.time_attention(q,q,q,need_weights=False)[0]
        spectral=temporal.reshape(batch,freq,time,channels).permute(0,2,1,3).reshape(batch*time,freq,channels)
        q=self.freq_norm(spectral);spectral=spectral+self.freq_attention(q,q,q,need_weights=False)[0]
        h=spectral.reshape(batch,time,freq,channels).permute(0,3,2,1)
        p=self.output(h)
        amplitude=torch.exp(0.6931471805599453*torch.tanh(p[:,0]));phase=(torch.pi/2)*torch.tanh(p[:,1])
        mask=torch.complex(amplitude*torch.cos(phase),amplitude*torch.sin(phase))
        return torch.istft(z*mask,n_fft=n_fft,hop_length=hop,window=window,center=True,onesided=False,length=raw.shape[-1],return_complex=True)

    def forward(self,signal):
        if signal.ndim!=3 or signal.shape[1:]!=(2,256):raise ValueError('signal must have shape [batch,2,256]')
        rms=signal.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt();x=signal/rms
        hidden=self.temporal_blocks(self.temporal_input(x));delta=.5*torch.tanh(self.temporal_output(hidden))
        raw=torch.complex(x[:,0],x[:,1]);proposal=torch.complex(x[:,0]+delta[:,0],x[:,1]+delta[:,1])
        filtered=.5*(self._filter(raw,proposal,32)+self._filter(raw,proposal,64))
        return (torch.stack((filtered.real,filtered.imag),1)+delta)*rms
