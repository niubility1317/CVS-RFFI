from __future__ import annotations

import torch
from torch import nn

from baselines.common.grl import gradient_reverse


def _to_paper_conv_input(x: torch.Tensor) -> torch.Tensor:
    """Convert WiSig IQ tensors to the paper's Conv2D input layout [B,1,256,2]."""
    if x.ndim == 3 and x.shape[1] == 2:
        return x.transpose(1, 2).unsqueeze(1)
    if x.ndim == 3 and x.shape[-1] == 2:
        return x.unsqueeze(1)
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] == 2:
        return x
    raise ValueError("expected IQ input shaped [batch,2,256], [batch,256,2], or [batch,1,256,2]")


class ConvBNReLUPoolBlock(nn.Module):
    """One paper block: convolution, batch normalization, ReLU, and max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReceiverAgnosticFeatureExtractor(nn.Module):
    """Four-block CNN feature extractor with a 128-D embedding, matching Sec.IV-B."""

    def __init__(self, feature_dim: int = 128, channels: tuple[int, int, int, int] = (16, 32, 64, 128)) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(ConvBNReLUPoolBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(in_channels, int(feature_dim))

    def forward(self, x: torch.Tensor, *, return_activations: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = _to_paper_conv_input(x)
        activations: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            if return_activations:
                activations.append(x)
        features = self.projection(self.pool(x).flatten(1))
        return features, activations


class DenseReLUHead(nn.Module):
    """Dense-128 + ReLU classifier head used for TX and source/target domain logits."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReceiverAgnosticUDANet(nn.Module):
    """DANN + LMMD-ready network for closed-set cross-receiver RFFI reproduction."""

    def __init__(
        self,
        *,
        num_tx: int,
        feature_dim: int = 128,
        classifier_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_extractor = ReceiverAgnosticFeatureExtractor(feature_dim=feature_dim)
        self.tx_classifier = DenseReLUHead(feature_dim, int(num_tx), classifier_hidden_dim)
        self.domain_classifier = DenseReLUHead(feature_dim, 1, classifier_hidden_dim)

    def forward(self, x: torch.Tensor, *, grl_lambda: float = 1.0, return_activations: bool = False) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features, activations = self.feature_extractor(x, return_activations=return_activations)
        tx_logits = self.tx_classifier(features)
        domain_logits = self.domain_classifier(gradient_reverse(features, float(grl_lambda)))
        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "features": features,
            "tx_logits": tx_logits,
            "domain_logits": domain_logits,
        }
        if return_activations:
            outputs["activations"] = activations
        return outputs


class ResidualBlock1D(nn.Module):
    """ResNet18-style residual block using 1-D convolutions for IQ time series."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


def _to_resnet1d_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 2:
        return x
    if x.ndim == 3 and x.shape[-1] == 2:
        return x.transpose(1, 2)
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] == 2:
        return x.squeeze(1).transpose(1, 2)
    raise ValueError("expected IQ input shaped [batch,2,256], [batch,256,2], or [batch,1,256,2]")


class ResNet18FeatureExtractor1D(nn.Module):
    """Feature extractor from the IoTJ 2024 paper: ResNet18 with 1-D convolutions."""

    def __init__(self, feature_dim: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(base_channels * 8, int(feature_dim))

    def _make_layer(self, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, *, return_activations: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = _to_resnet1d_input(x)
        activations: list[torch.Tensor] = []
        x = self.stem(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            if return_activations:
                activations.append(x)
        features = self.projection(self.pool(x).flatten(1))
        return features, activations


class ThreeLayerFCNet(nn.Module):
    """Three-layer fully connected network used for classifier and estimate heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReceiverImpactGADNet(nn.Module):
    """IoTJ 2024 GAD model with feature extractor E, classifier C, and estimate network T."""

    def __init__(self, *, num_tx: int, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_extractor = ResNet18FeatureExtractor1D(feature_dim=feature_dim)
        self.classifier = ThreeLayerFCNet(feature_dim, int(num_tx), hidden_dim)
        self.estimate_network = ThreeLayerFCNet(feature_dim, 1, hidden_dim)

    def forward(self, x: torch.Tensor, *, return_activations: bool = False) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features, activations = self.feature_extractor(x, return_activations=return_activations)
        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "features": features,
            "tx_logits": self.classifier(features),
            "estimate_logits": self.estimate_network(features),
        }
        if return_activations:
            outputs["activations"] = activations
        return outputs
