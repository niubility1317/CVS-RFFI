import copy
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from experiments.pamf_unrolled_restore.models import UnrolledRestorer


class TemporalBlock(nn.Module):
    def __init__(self,width=128,heads=4):
        super().__init__()
        self.heads=heads;self.depth=width//heads
        self.attention_norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.project=nn.Linear(width,width)
        self.local_norm=nn.LayerNorm(width)
        self.local=nn.Conv1d(width,width,5,padding=2,groups=width)
        self.mix=nn.Conv1d(width,width,1)
        self.mlp_norm=nn.LayerNorm(width)
        self.mlp=nn.Sequential(nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))

    def forward(self,value):
        batch,length,width=value.shape
        q,k,v=self.qkv(self.attention_norm(value)).reshape(batch,length,3,self.heads,self.depth).permute(2,0,3,1,4).unbind(0)
        weights=((q@k.transpose(-1,-2))/math.sqrt(self.depth)).softmax(-1)
        attention=(weights@v).transpose(1,2).reshape(batch,length,width)
        value=value+.5*self.project(attention)
        local=self.local_norm(value).transpose(1,2)
        value=value+.5*self.mix(torch.nn.functional.silu(self.local(local))).transpose(1,2)
        return value+.5*self.mlp(self.mlp_norm(value))


class TemporalAttentionPrior(nn.Module):
    width=128
    heads=4

    def __init__(self,checkpoint_blocks=True):
        super().__init__();self.checkpoint_blocks=checkpoint_blocks
        self.stem=nn.Conv1d(9,self.width,1)
        self.blocks=nn.ModuleList(TemporalBlock(self.width,self.heads) for _ in range(6))
        self.final_norm=nn.LayerNorm(self.width);self.out=nn.Conv1d(self.width,2,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        locations=torch.arange(256,dtype=torch.float32)[:,None]
        frequencies=torch.exp(-math.log(10000)*torch.arange(0,self.width,2,dtype=torch.float32)/self.width)
        position=torch.empty(256,self.width)
        position[:,0::2]=torch.sin(locations*frequencies)
        position[:,1::2]=torch.cos(locations*frequencies)
        self.register_buffer('position',.1*position[None])

    def forward(self,value):
        if (value.ndim!=3 or value.shape[1:]!=(9,256) or value.dtype!=torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError('Finite FP32 current-packet 9x256 prior inputs required')
        value=self.stem(value).transpose(1,2)+self.position
        for block in self.blocks:
            value=(checkpoint(block,value,use_reentrant=False)
                   if self.checkpoint_blocks and self.training and torch.is_grad_enabled() else block(value))
        return self.out(self.final_norm(value).transpose(1,2))


class AttentionRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in ('unet_control','attention_prior'):
            raise ValueError('Registered U-Net or attention prototype required')
        if candidate=='unet_control':
            super().__init__('unroll_unshared',frozen)
        else:
            nn.Module.__init__(self)
            self.frozen=frozen.requires_grad_(False).eval()
            core=TemporalAttentionPrior()
            self.prior=nn.ModuleList(copy.deepcopy(core) for _ in range(3))
        self.candidate=candidate
