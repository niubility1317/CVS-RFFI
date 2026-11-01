"""Input-conditioned mixture of invariant and rotation-canceling phase rules."""
import torch
from torch import nn

class AdaptivePhaseFactor(nn.Module):
    def __init__(self,channels=64):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(5,channels,7,padding=3),nn.GELU(),nn.Conv1d(channels,channels,7,padding=3),nn.GELU(),nn.Conv1d(channels,channels,5,padding=2),nn.GELU())
        self.weights=nn.Conv1d(channels,2,1)
        self.mix=nn.Sequential(nn.Linear(channels*2,64),nn.GELU(),nn.Linear(64,3))
        with torch.no_grad():self.mix[-1].bias[1]=1.;self.mix[-1].bias[2]=0.

    def components(self,iq):
        if iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('Expected [batch,2,256] IQ')
        x=iq/iq.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt();z=torch.complex(x[:,0],x[:,1]);a=z.abs();c1=z*torch.roll(z,1,-1).conj();c20=z*torch.roll(z,20,-1).conj()
        feat=torch.stack((a,c1.real,c1.imag,c20.real,c20.imag),1);h=self.features(feat);w=self.weights(h);q=(torch.complex(w[:,0],w[:,1])*z).mean(-1)
        canonical=q.conj()/q.abs().clamp_min(1e-8);summary=torch.cat((h.mean(-1),h.var(-1,unbiased=False).clamp_min(1e-12).sqrt()),1);params=self.mix(summary);mix=params[:,0].sigmoid();orientation=torch.complex(params[:,1],params[:,2]);orientation=orientation/orientation.abs().clamp_min(1e-8)
        return canonical,orientation,mix

    def forward(self,iq):
        canonical,orientation,mix=self.components(iq);z=(1.-mix)+mix*canonical*orientation;return z/z.abs().clamp_min(1e-8)
