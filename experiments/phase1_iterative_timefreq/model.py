"""Single-window, phase-equivariant iterative time/frequency IQ recovery."""
import torch
from torch import nn
from torch.nn import functional as F


class IterativeTimeFreq(nn.Module):
    def __init__(self,width=64,steps=3):
        super().__init__()
        if width<4 or width%4 or steps<1:raise ValueError('Width divisible by four and positive steps required')
        self.steps=steps
        self.entry=nn.Conv1d(5,width,7,padding=3,padding_mode='reflect')
        self.frequency_entry=nn.Conv1d(5,width,7,padding=3,padding_mode='circular')
        self.time_layers=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d,padding_mode='reflect') for d in (1,2,4,8))
        self.frequency_layers=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d,padding_mode='circular') for d in (1,2,4,8))
        self.context_norm=nn.LayerNorm(width)
        self.context=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.gate_output=nn.Conv1d(width,2,1)
        self.frequency_gate=nn.Conv1d(width,2,1)
        for layer in (self.gate_output,self.frequency_gate):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,received,initial,log_variance):
        n=len(received)
        if received.shape!=(n,2,256) or initial.shape!=received.shape or log_variance.shape!=(n,):
            raise ValueError('Matching single-record IQ windows and blind variance required')
        if not all(torch.isfinite(v).all() for v in (received,initial,log_variance)):
            raise FloatingPointError('Finite IQ and noise evidence required')
        power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8);scale=power.sqrt()
        raw=torch.complex(received[:,0],received[:,1])/scale[:,0]
        initial_complex=torch.complex(initial[:,0],initial[:,1])/scale[:,0]
        pivot=raw.gather(1,raw.abs().argmax(1,keepdim=True))
        unit=torch.where(pivot.abs()>1e-8,pivot/pivot.abs().clamp_min(1e-8),torch.ones_like(pivot))
        x=raw*unit.conj();anchor=initial_complex*unit.conj();state=anchor
        noise=((log_variance-power[:,0,0].log()).clamp(-16,8)/8)[:,None,None].expand(-1,1,256)
        xf=torch.fft.fft(x,norm='ortho')
        for _ in range(self.steps):
            time=torch.stack((x.real,x.imag,state.real,state.imag),1)
            h=F.silu(self.entry(torch.cat((time,noise),1)))
            for layer in self.time_layers:h=h+.2*F.silu(layer(h))
            token=F.avg_pool1d(h,4).transpose(1,2);q=self.context_norm(token)
            attention,_=self.context(q,q,q,need_weights=False)
            h=h+.2*F.interpolate(attention.transpose(1,2),size=256,mode='linear',align_corners=False)
            t=self.gate_output(h);time_delta=torch.complex(t[:,0],t[:,1])
            sf=torch.fft.fft(state,norm='ortho')
            spectrum=torch.stack((xf.real,xf.imag,sf.real,sf.imag),1)
            k=F.silu(self.frequency_entry(torch.cat((spectrum,noise),1)))
            for layer in self.frequency_layers:k=k+.2*F.silu(layer(k))
            f=self.frequency_gate(k);freq_delta=torch.fft.ifft(torch.complex(f[:,0],f[:,1]),norm='ortho')
            state=state+.5*(time_delta+freq_delta)
        correction=(state-anchor)*unit
        output=initial+torch.stack((correction.real,correction.imag),1)*scale
        if not torch.isfinite(output).all():raise FloatingPointError('Finite restored IQ required')
        return output
