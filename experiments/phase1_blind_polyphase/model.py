"""Equivalent-structure blind recovery with polyphase dilated convolutions.

Floating-point reductions can differ; exact training trajectories or final
classification decisions are not assumed to match the original execution.
"""
from torch import nn
from experiments.phase1_blind_unrolled.model import BlindUnrolledRecovery, ResidualBlock
from experiments.phase1_dura_orthogonal.polyphase_conv import polyphase_conv1d


class PolyphaseConv1d(nn.Conv1d):
    def forward(self, x):
        return polyphase_conv1d(x, self.weight, self.bias, self.dilation[0])


class BlindPolyphaseRecovery(BlindUnrolledRecovery):
    def __init__(self, width=192, steps=4):
        super().__init__(width=width, steps=steps)
        for block in self.modules():
            if isinstance(block, ResidualBlock):
                conv = block.block[0]
                if conv.dilation[0] > 1:
                    fast = PolyphaseConv1d(conv.in_channels, conv.out_channels, conv.kernel_size,
                        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                        groups=conv.groups, bias=conv.bias is not None)
                    fast.weight = conv.weight
                    fast.bias = conv.bias
                    block.block[0] = fast
