"""Multiscale waveform restoration with a global temporal attention bottleneck."""
import torch
from torch import nn
from torch.nn import functional as F

class AttentionRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv=nn.Sequential(nn.Conv1d(2,48,7,padding=3),nn.GELU())
        self.down1=nn.Sequential(nn.Conv1d(48,80,5,stride=2,padding=2),nn.GELU())
        self.down2=nn.Sequential(nn.Conv1d(80,128,5,stride=2,padding=2),nn.GELU())
        layer=nn.TransformerEncoderLayer(128,4,dim_feedforward=256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.global_context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.position=nn.Parameter(torch.zeros(1,64,128))
        self.up1=nn.Sequential(nn.Conv1d(208,80,5,padding=2),nn.GELU())
        self.up2=nn.Sequential(nn.Conv1d(128,48,5,padding=2),nn.GELU())
        self.output=nn.Conv1d(48,2,7,padding=3)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected IQ shape [N,2,256]')
        scale=x.square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt();z=x/scale
        a=self.in_conv(z);b=self.down1(a);c=self.down2(b)
        c=self.global_context(c.transpose(1,2)+self.position).transpose(1,2)
        d=self.up1(torch.cat((F.interpolate(c,size=b.shape[-1],mode='linear',align_corners=False),b),1))
        d=self.up2(torch.cat((F.interpolate(d,size=a.shape[-1],mode='linear',align_corners=False),a),1))
        return x+scale*self.output(d)
