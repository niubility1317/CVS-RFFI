"""Independent Torch implementations of the specified ISSL and LoRa topologies.

No author source file is redistributed. ISSL's 1D ResNet18 has the documented
2/2/2/2 residual blocks, 64/128/256/512 channels and pre-head average embedding.
LoRa follows the channel-independent-spectrogram 32/64-channel residual encoder
and 512-dimensional unit embedding. This is a Torch port, not TF training parity.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ISSLBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual)


class ISSLResNet18(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(4, 2, 1)
        self.layer1 = nn.Sequential(ISSLBlock(64, 64), ISSLBlock(64, 64))
        self.layer2 = nn.Sequential(ISSLBlock(64, 128, 2), ISSLBlock(128, 128))
        self.layer3 = nn.Sequential(ISSLBlock(128, 256, 2), ISSLBlock(256, 256))
        self.layer4 = nn.Sequential(ISSLBlock(256, 512, 2), ISSLBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(.1)  # Author forward leaves this unused.
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        z = self.avgpool(x).flatten(1)
        return self.fc(z), z


class SameConv(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv2d(cin, cout, kernel, stride)

    def forward(self, x):
        pad = [max((math.ceil(s / self.stride) - 1) * self.stride + self.kernel - s, 0)
               for s in x.shape[-2:]]
        x = F.pad(x, (pad[1] // 2, pad[1] - pad[1] // 2, pad[0] // 2, pad[0] - pad[0] // 2))
        return self.conv(x)


class LoRaBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1, self.conv2 = SameConv(cin, cout, 3), SameConv(cout, cout, 3)
        self.shortcut = SameConv(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        return F.relu(self.conv2(F.relu(self.conv1(x))) + residual)


class LoRaExtractor(nn.Module):
    def __init__(self, height=26, width=6):
        super().__init__()
        self.conv1 = SameConv(1, 32, 7, 2)
        self.resblock1, self.resblock2 = LoRaBlock(32, 32), LoRaBlock(32, 32)
        self.resblock3, self.resblock4 = LoRaBlock(32, 64), LoRaBlock(64, 64)
        self.avgpool = nn.AvgPool2d(2)
        with torch.no_grad():
            n = self.convolution(torch.zeros(1, 1, height, width)).numel()
        self.fc = nn.Linear(n, 512)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def convolution(self, x):
        x = F.relu(self.conv1(x))
        return self.avgpool(self.resblock4(self.resblock3(self.resblock2(self.resblock1(x)))))

    def forward(self, x):
        x = self.convolution(x).permute(0, 2, 3, 1).contiguous().flatten(1)
        return F.normalize(self.fc(x), dim=1, eps=1e-12)
