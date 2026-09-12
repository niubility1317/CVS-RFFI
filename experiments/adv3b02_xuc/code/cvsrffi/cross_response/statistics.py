"""Detached physical waveform statistics and immutable source-fitted scaling."""
import torch
from torch import nn


class FixedSourceNormalizer(nn.Module):
    def __init__(self, dim, min_scale=1e-6):
        super().__init__()
        if dim < 1 or min_scale <= 0:
            raise ValueError('invalid normalization dimension or floor')
        self.min_scale = float(min_scale)
        self.register_buffer('location', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('near_zero', torch.zeros(dim, dtype=torch.bool))
        self.register_buffer('fitted', torch.tensor(False))
        self.register_buffer('sample_count', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def fit(self, values, source_role='source_train'):
        if source_role != 'source_train':
            raise ValueError('normalization may only fit legal source_train records')
        if bool(self.fitted):
            raise RuntimeError('source normalization is frozen; refitting prohibited')
        if values.ndim < 2 or values.shape[-1] != self.location.numel():
            raise ValueError('source statistics must have the configured coordinate dimension')
        x = values.detach().double().reshape(-1, self.location.numel())
        if x.shape[0] < 2 or not torch.isfinite(x).all():
            raise ValueError('at least two finite source records required')
        std = x.std(0, unbiased=False)
        self.location.copy_(x.mean(0))
        self.near_zero.copy_(std < self.min_scale)
        self.scale.copy_(torch.where(std < self.min_scale, torch.ones_like(std), std))
        self.sample_count.fill_(x.shape[0])
        self.fitted.fill_(True)
        return self

    def forward(self, values):
        if not bool(self.fitted):
            raise RuntimeError('source normalization must be fitted before use')
        if values.shape[-1] != self.location.numel() or not torch.isfinite(values).all():
            raise ValueError('invalid normalization input')
        return ((values.detach().float()-self.location)/self.scale).detach()


class WaveformStatistics(nn.Module):
    """IQ is [...,2,T] or complex [...,T]. Event family uses real protocol windows.

    Event metadata is a dictionary with event_ids (nonempty real identifiers) and
    windows ([start,stop] protocol positions). Each window reports mean received
    power, mean envelope, and envelope slope, without amplitude normalization.
    """
    def __init__(self, family='fft', bands=8, lags=(1,2,4), eps=1e-8, input_length=None):
        super().__init__()
        if family not in ('fft','autocorr','iq','event') or bands < 1 or eps <= 0:
            raise ValueError('invalid statistics configuration')
        if not lags or any(int(k) != k or k < 1 for k in lags):
            raise ValueError('autocorrelation lags must be positive integers')
        self.family, self.bands, self.lags, self.eps = family, int(bands), tuple(lags), float(eps)
        self.register_buffer('input_length', torch.tensor(-1 if input_length is None else input_length))

    @property
    def output_dim(self):
        return {'fft': self.bands, 'autocorr': 2*len(self.lags), 'iq': 5}.get(self.family)

    @torch.no_grad()
    def forward(self, iq, event_metadata=None):
        if iq.is_complex():
            z = iq.detach().to(torch.complex64)
        else:
            if iq.ndim < 2 or iq.shape[-2] != 2:
                raise ValueError('IQ must be [...,2,T] or complex [...,T]')
            x = iq.detach().float()
            z = torch.complex(x[...,0,:], x[...,1,:])
        n = z.shape[-1]
        if n < 1 or not torch.isfinite(z).all():
            raise ValueError('empty or nonfinite IQ is unavailable')
        if int(self.input_length) < 0:
            self.input_length.fill_(n)
        if n != int(self.input_length):
            raise ValueError('statistic input length/band boundaries are frozen')
        power = z.abs().square()
        if not torch.isfinite(power).all():
            raise ValueError('IQ power exceeds FP32 statistic range')
        if self.family == 'fft':
            if n < self.bands:
                raise ValueError('sequence shorter than fixed frequency band count')
            spectrum = torch.fft.fftshift(torch.fft.fft(z), dim=-1).abs().square()
            energy = torch.stack([v.sum(-1) for v in torch.tensor_split(spectrum,self.bands,dim=-1)],-1)
            if not torch.isfinite(energy.sum(-1)).all():
                raise ValueError('FFT energy exceeds FP32 statistic range')
            return torch.log(energy/energy.sum(-1,keepdim=True).clamp_min(self.eps)+self.eps)
        if self.family == 'autocorr':
            if n <= max(self.lags):
                raise ValueError('sequence does not support requested lags')
            vals = [(z[...,k:]*z[...,:-k].conj()).mean(-1)/power.mean(-1).clamp_min(self.eps) for k in self.lags]
            return torch.stack([part for v in vals for part in (v.real,v.imag)],-1)
        if self.family == 'iq':
            centered = z-z.mean(-1,keepdim=True)
            i,q = centered.real,centered.imag
            pseudo = centered.square().mean(-1)
            return torch.stack((i.square().mean(-1),q.square().mean(-1),(i*q).mean(-1),pseudo.real,pseudo.imag),-1)
        if not isinstance(event_metadata,dict) or not event_metadata.get('event_ids') or not event_metadata.get('windows'):
            raise ValueError('event statistics unavailable without real event IDs and protocol windows')
        if len(event_metadata['event_ids']) != z.numel()//n or any(v is None or str(v)=='' for v in event_metadata['event_ids']):
            raise ValueError('each original record requires a real event identifier')
        out = []
        for start, stop in event_metadata['windows']:
            if not (0 <= start < stop <= n) or stop-start < 2:
                raise ValueError('event window must contain at least two real samples')
            a = z[...,start:stop].abs()
            t = torch.arange(stop-start,device=z.device,dtype=torch.float32)
            t = t-t.mean()
            out.extend((a.square().mean(-1),a.mean(-1),(a*t).sum(-1)/t.square().sum()))
        return torch.stack(out,-1)


def query_cell_targets(statistics, clean_iq, normalizer=None, event_metadata=None):
    """Compute each original record first, then mean over K (axis -2)."""
    values = statistics(clean_iq,event_metadata=event_metadata)
    if normalizer is not None:
        values = normalizer(values)
    if values.ndim != 4 or values.shape[2] < 1:
        raise ValueError('expected clean IQ statistics [P,Q,K,D]')
    return values.mean(2).detach()
