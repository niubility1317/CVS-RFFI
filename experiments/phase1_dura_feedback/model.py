"""Continuous DURA correction conditioned on same-window frozen-reader features."""
import torch
from torch import nn
from torch.nn import functional as F

def pack_context(anchor,features,logits):
    n=len(anchor)
    if anchor.shape!=(n,2,256) or features.shape!=(n,160) or logits.shape!=(n,6):
        raise ValueError('One matching anchor, feature and logit vector per window is required')
    if not all(torch.isfinite(v).all() for v in (anchor,features,logits)):
        raise FloatingPointError('Finite frozen-reader evidence is required')
    condition=torch.cat((features.detach(),logits.detach(),anchor.new_zeros(n,90)),dim=1).unsqueeze(1)
    return torch.cat((anchor,condition),dim=1)

class FeedbackRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry=nn.Conv1d(5,48,7,padding=3)
        self.down1=nn.Conv1d(48,64,5,stride=2,padding=2)
        self.down2=nn.Conv1d(64,96,5,stride=2,padding=2)
        self.condition=nn.Sequential(nn.Linear(166,96),nn.SiLU(),nn.Linear(96,96))
        self.context_norm=nn.LayerNorm(96)
        self.context=nn.MultiheadAttention(96,4,dropout=0.,batch_first=True)
        self.up1=nn.Conv1d(160,64,5,padding=2)
        self.up0=nn.Conv1d(112,48,5,padding=2)
        self.gate_output=nn.Conv1d(48,2,1)
        nn.init.zeros_(self.gate_output.weight);nn.init.zeros_(self.gate_output.bias)

    def forward(self,received,packed,log_variance):
        n=len(received)
        if received.shape!=(n,2,256) or packed.shape!=(n,3,256) or log_variance.shape!=(n,):
            raise ValueError('Matching IQ windows and one same-record condition are required')
        if not all(torch.isfinite(v).all() for v in (received,packed,log_variance)):
            raise FloatingPointError('Finite same-record evidence is required')
        initial=packed[:,:2];features=packed[:,2,:160].detach();logits=packed[:,2,160:166].detach()
        condition=self.condition(torch.cat((F.normalize(features,dim=1),F.softmax(logits,dim=1)),dim=1))
        power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8);scale=power.sqrt()
        z=torch.complex(received[:,0],received[:,1])/scale[:,0]
        a=torch.complex(initial[:,0],initial[:,1])/scale[:,0]
        pivot=z.gather(1,z.abs().argmax(dim=1,keepdim=True))
        unit=pivot/pivot.abs().clamp_min(1e-8)
        zr,ar=z*unit.conj(),a*unit.conj()
        noise=((log_variance-power[:,0,0].log()).clamp(-16,8)/8)[:,None,None]
        local=torch.stack((zr.real,zr.imag,ar.real,ar.imag),dim=1)
        x0=F.silu(self.entry(torch.cat((local,noise.expand(-1,1,256)),dim=1)))
        x1=F.silu(self.down1(x0));x2=F.silu(self.down2(x1))
        q=self.context_norm(x2.transpose(1,2))+condition[:,None,:]
        context,_=self.context(q,q,q,need_weights=False)
        x2=x2+0.2*context.transpose(1,2)
        h1=F.silu(self.up1(torch.cat((F.interpolate(x2,size=128,mode='linear',align_corners=False),x1),dim=1)))
        h0=F.silu(self.up0(torch.cat((F.interpolate(h1,size=256,mode='linear',align_corners=False),x0),dim=1)))
        delta=self.gate_output(h0).tanh();cd=torch.complex(delta[:,0],delta[:,1])*unit
        return initial+0.5*torch.stack((cd.real,cd.imag),dim=1)*scale
