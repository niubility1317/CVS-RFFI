"""MoPC-HR author attention ResNet: explicit WiSig256/input adaptation.

Derived from ``ads_resnet.py`` and ``ConvBlockAttention.py`` in
https://github.com/xmuLdz/MoPC-HR at
ae6554316ad1a2175920e330133a2f103408bf78. No author checkpoint is loaded.

The author Conv2d(3, 1) stem receives one observation as [N, 3, 1, T]:
I, Q, and sqrt(I**2 + Q**2). Only the hardcoded 1024-point reshapes become
length-preserving flatten operations. All ten residual/attention blocks and
the author's unusual AdaptiveAvgPool2d applied to a 3-D tensor are retained.
The latter pools the channel and time axes to [N, 1, 256]; replacing it with
global temporal pooling would change the author's embedding computation.

WiSig uses T=256 and six source classes instead of the author's 100 classes.
This is an input/class-count adaptation, not a bitwise reproduction of the
paper's dataset, input preparation, training procedure, or reported results.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


AUTHOR_REPOSITORY = 'https://github.com/xmuLdz/MoPC-HR'
AUTHOR_COMMIT = 'ae6554316ad1a2175920e330133a2f103408bf78'
ARCHITECTURE_VERSION = 'MoPC-HR author attention ResNet WiSig256/input adaptation'
MODEL_CONFIG = ((8, 2, 8), (32, 2, 8), (64, 2, 4), (128, 2, 4), (256, 2, 2))


def architecture_metadata() -> dict:
    return dict(architecture=ARCHITECTURE_VERSION, author_repository=AUTHOR_REPOSITORY,
                author_commit=AUTHOR_COMMIT, author_files=['ads_resnet.py', 'ConvBlockAttention.py'],
                input='FP32[N,2,256]', author_layout='[N,3,1,T]: I,Q,amplitude',
                embedding_dim=256, source_classes=6, initialization='scratch',
                input_resampling=False, observations_per_input=1,
                author_residual_attention_preserved=True,
                author_3d_adaptive_avg_pool2d_preserved=True,
                paper_original_dataset_reproduction=False)


def iq_to_author_layout(x: torch.Tensor) -> torch.Tensor:
    """Derive the three stem channels from each individual canonical IQ record."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError('Expected IQ [batch,2,length] from one observation per row')
    # vector_norm has the exact amplitude value and a finite zero subgradient,
    # unlike a bare sqrt(sum(square(x))) at all-zero IQ samples.
    amplitude = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return torch.cat((x, amplitude), dim=1).unsqueeze(2)


class BasicConv(nn.Module):
    """Author attention convolution, including its BN momentum of 0.01."""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x) if self.relu is not None else x


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def log_sum_exp_1d(x):
    flattened = x.reshape(x.size(0), x.size(1), -1)
    largest = flattened.max(dim=2, keepdim=True).values
    return largest + (flattened - largest).exp().sum(dim=2, keepdim=True).log()


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                 nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = ['avg', 'max'] if pool_types is None else pool_types

    def forward(self, x):
        total = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pooled = F.avg_pool1d(x, x.size(2), stride=x.size(2))
            elif pool_type == 'max':
                pooled = F.max_pool1d(x, x.size(2), stride=x.size(2))
            elif pool_type == 'lp':
                pooled = F.lp_pool1d(x, 2., x.size(2), stride=x.size(2))
            elif pool_type == 'lse':
                pooled = log_sum_exp_1d(x)
            else:
                raise ValueError('Unknown author attention pool type: ' + str(pool_type))
            value = self.mlp(pooled)
            total = value if total is None else total + value
        return x * torch.sigmoid(total).unsqueeze(2).expand_as(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, 7, stride=1, padding=3, relu=False)

    def forward(self, x):
        return x * torch.sigmoid(self.spatial(self.compress(x)))


class ConvBlockAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        return x if self.no_spatial else self.SpatialGate(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=2, downsample=None,
                 attention=ConvBlockAttention):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        # The author explicitly overrides the attention reduction ratio to 2.
        self.attention = attention(planes, 2) if attention is not None else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attention is not None:
            out = self.attention(out)
        difference = residual.size(2) - out.size(2)
        # Preserve the author's right-zero-padding residual alignment. F.pad
        # also keeps the active tensor's dtype/device rather than default CUDA.
        if difference > 0:
            out = F.pad(out, (0, difference))
        elif difference < 0:
            residual = F.pad(residual, (0, -difference))
        out += residual
        return self.relu(out)


class MoPCAuthorWiSig256(nn.Module):
    """Author ResNet with an explicit single-record, variable-length IQ input.

    The supported experiment input is FP32 [N,2,256]. The same computation also
    accepts longer records (including 1024 for author-forward equivalence);
    no observations are joined, repeated, resized, or resampled at the input.
    ``forward`` exposes the existing comparison-suite mapping around the
    author's unnormalized embedding and ordinary linear classifier.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv1_ = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.in_planes = 2
        self.conv1 = nn.Conv1d(1, 2, kernel_size=7, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)
        self.attention = ConvBlockAttention
        self.layers = nn.Sequential(*(self.make_layer(channel, count, kernel)
                                      for channel, count, kernel in MODEL_CONFIG))
        self.out_dim = 256
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, self.out_dim))
        self.fc = nn.Linear(self.out_dim, num_classes)

    def make_layer(self, planes, blocks, kernel_size):
        downsample = None
        if self.in_planes != planes:
            downsample = nn.Sequential(nn.Conv1d(self.in_planes, planes, 1, stride=1, bias=False),
                                       nn.BatchNorm1d(planes))
        layers = [BasicBlock(self.in_planes, planes, kernel_size=kernel_size,
                             downsample=downsample, attention=self.attention)]
        self.in_planes = planes
        layers.extend(BasicBlock(planes, planes, kernel_size=kernel_size, attention=self.attention)
                      for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def feature_extract_stage1(self, author_layout):
        return self.conv1_(author_layout).flatten(start_dim=2)

    def feature_extract_stage2(self, author_layout):
        x = self.feature_extract_stage1(author_layout)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.avg_pooling(self.layers(x)).flatten(start_dim=1)

    def features(self, x):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] < 19:
            raise ValueError('Expected IQ [batch,2,length>=19]; WiSig uses length=256')
        return self.feature_extract_stage2(iq_to_author_layout(x))

    def classify_features(self, z):
        return self.fc(z)

    def forward(self, x):
        z = self.features(x)
        return {'features': z, 'tx_logits': self.classify_features(z)}
