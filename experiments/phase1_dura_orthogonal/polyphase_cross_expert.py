"""Experimental cross-expert variant using polyphase convolutions."""
import torch
from torch import nn
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.polyphase_conv import polyphase_conv1d


class PolyphaseConv1d(nn.Conv1d):
    def forward(self, x):
        if self.stride!=(1,) or self.groups!=1 or self.padding!=(self.dilation[0]*(self.kernel_size[0]//2),):
            raise ValueError('Fixed stride-one same-padded ungroupped convolution required')
        return polyphase_conv1d(x,self.weight,self.bias,self.dilation[0])


class PolyphaseCrossExpertRefiner(CrossExpertRefiner):
    """Same parameter names and initialization as the fixed full model."""
    def __init__(self,width=192):
        super().__init__(width=width)
        for block in (*self.time_blocks,*self.freq_blocks):
            original=block.body[0]
            if original.dilation[0] <= 1:continue
            with torch.random.fork_rng(devices=[]):
                converted=PolyphaseConv1d(original.in_channels,original.out_channels,original.kernel_size,
                    stride=original.stride,padding=original.padding,dilation=original.dilation,
                    groups=original.groups,bias=original.bias is not None)
            converted.load_state_dict(original.state_dict(),strict=True)
            block.body[0]=converted
