"""Larger per-packet multiscale prior with global temporal attention."""
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.core=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,width,3,padding=1),
            nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,width,3,padding=1))

    def forward(self,x):return x+self.core(x)


class Attention(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.norm=nn.LayerNorm(width);self.attention=nn.MultiheadAttention(width,8,dropout=0.,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))

    def forward(self,x):
        value=self.norm(x);x=x+self.attention(value,value,value,need_weights=False)[0]
        return x+self.ff(x)


class MultiscaleAttentionRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('coordinates',torch.linspace(-1,1,256)[None,None])
        self.stem=nn.Conv1d(3,64,5,padding=2)
        widths=(64,128,192,256)
        self.encoders=nn.ModuleList(nn.Sequential(Residual(w),Residual(w)) for w in widths[:3])
        self.down=nn.ModuleList(nn.Conv1d(a,b,4,stride=2,padding=1) for a,b in zip(widths[:-1],widths[1:]))
        self.position=nn.Parameter(torch.zeros(1,32,256))
        self.bottleneck=nn.Sequential(*[Attention() for _ in range(4)])
        self.up=nn.ModuleList(nn.Conv1d(a+b,b,3,padding=1) for a,b in [(256,192),(192,128),(128,64)])
        self.decoders=nn.ModuleList(nn.Sequential(Residual(w),Residual(w)) for w in (192,128,64))
        self.output=nn.Conv1d(64,2,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Current per-packet FP32 [N,2,256] received IQ required')
        x=self.stem(torch.cat((received,self.coordinates.expand(len(received),-1,-1)),dim=1));skips=[]
        for encoder,down in zip(self.encoders,self.down):
            x=encoder(x);skips.append(x);x=down(x)
        x=self.bottleneck(x.transpose(1,2)+self.position).transpose(1,2)
        for up,decoder,skip in zip(self.up,self.decoders,reversed(skips)):
            x=F.interpolate(x,size=skip.shape[-1],mode='linear',align_corners=False)
            x=decoder(up(torch.cat((x,skip),dim=1)))
        return received+self.output(x)


def scratch_multiscale(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=MultiscaleAttentionRestorer()
    model.initialization_report=dict(mode='random_single_stage',model_seed=seed,checkpoint_sources=[],ancestors=[],
        training_updates=0,TX_labels_read=0,F_reads=0,V_reads=0,OE_reads=0,target_reads=0)
    return model
