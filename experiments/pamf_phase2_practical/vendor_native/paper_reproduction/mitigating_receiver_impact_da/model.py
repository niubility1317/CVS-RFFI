from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _to_resnet1d_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 2:
        return x
    if x.ndim == 3 and x.shape[-1] == 2:
        return x.transpose(1, 2)
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] == 2:
        return x.squeeze(1).transpose(1, 2)
    raise ValueError("expected IQ input shaped [batch,2,256], [batch,256,2], or [batch,1,256,2]")


class ResidualBlock1D(nn.Module):
    """ResNet18-style residual block using 1-D convolutions for IQ time series."""

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


class ResNet18FeatureExtractor1D(nn.Module):
    """Feature extractor E from the IoTJ 2024 paper: ResNet18 with 1-D convolutions."""

    def __init__(self, feature_dim: int = 512, base_channels: int = 64) -> None:
        super().__init__()
        self.output_channels = int(base_channels) * 8
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
        if int(feature_dim) == self.output_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(self.output_channels, int(feature_dim))

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


class SamePadConv1d(nn.Module):
    """The SAME-padding convolution used by the authors' linked template."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_len = (x.shape[-1] + self.stride - 1) // self.stride
        padding = max(0, (out_len - 1) * self.stride + self.kernel_size - x.shape[-1])
        return self.conv(F.pad(x, (padding // 2, padding - padding // 2)))


class SamePadMaxPool1d(nn.Module):
    def __init__(self, *, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_len = (x.shape[-1] + self.kernel_size - 1) // self.kernel_size
        padding = max(0, (out_len - 1) * self.kernel_size + self.kernel_size - x.shape[-1])
        return self.pool(F.pad(x, (padding // 2, padding - padding // 2)))


class TemplateResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        downsample: bool,
        first_block: bool,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride) if downsample else 1
        self.downsample = bool(downsample)
        self.first_block = bool(first_block)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = SamePadConv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
        )
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = SamePadConv1d(
            self.out_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=1,
        )
        self.identity_pool = SamePadMaxPool1d(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        if not self.first_block:
            out = self.relu1(self.bn1(out))
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.downsample:
            identity = self.identity_pool(identity)
        if self.out_channels != self.in_channels:
            left = (self.out_channels - self.in_channels) // 2
            right = self.out_channels - self.in_channels - left
            identity = F.pad(identity.transpose(1, 2), (left, right)).transpose(1, 2)
        return out + identity


class PytorchTemplateResNet18FeatureExtractor1D(nn.Module):
    """The linked ResNet1D template instantiated with eight basic blocks."""

    def __init__(self, *, base_filters: int = 64, kernel_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.first_conv = SamePadConv1d(2, base_filters, kernel_size=kernel_size, stride=1)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_relu = nn.ReLU()
        blocks: list[nn.Module] = []
        in_channels = int(base_filters)
        for block_index in range(8):
            if block_index == 0:
                out_channels = int(base_filters)
            else:
                in_channels = int(base_filters) * (2 ** ((block_index - 1) // 2))
                out_channels = in_channels * 2 if block_index % 2 == 0 else in_channels
            blocks.append(
                TemplateResidualBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    downsample=(block_index % 2 == 1),
                    first_block=(block_index == 0),
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.final_bn = nn.BatchNorm1d(in_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_channels = int(in_channels)

    def forward(self, x: torch.Tensor, *, return_activations: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out = _to_resnet1d_input(x)
        out = self.first_relu(self.first_bn(self.first_conv(out)))
        activations: list[torch.Tensor] = []
        for block in self.blocks:
            out = block(out)
            if return_activations:
                activations.append(out)
        features = self.pool(self.final_relu(self.final_bn(out))).flatten(1)
        return features, activations


class ThreeLayerFCNet(nn.Module):
    """Three-layer fully connected network used for classifier C and estimate network T."""

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


class TemplateThreeLayerFCNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemplateProjectorFCNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LeakyReLU(0.03),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LeakyReLU(0.03),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReceiverImpactGADNet(nn.Module):
    """GAD model with feature extractor E, classifier C, and estimate network T."""

    def __init__(
        self,
        *,
        num_tx: int,
        feature_dim: int = 512,
        hidden_dim: int | None = None,
        model_profile: str = "standard_resnet18",
    ) -> None:
        super().__init__()
        self.num_tx = int(num_tx)
        self.model_profile = str(model_profile).strip().lower()
        if self.model_profile == "standard_resnet18":
            resolved_hidden_dim = int(feature_dim if hidden_dim is None else hidden_dim)
            self.feature_extractor = ResNet18FeatureExtractor1D(feature_dim=feature_dim)
            self.classifier = ThreeLayerFCNet(feature_dim, self.num_tx, resolved_hidden_dim)
            self.estimate_network = ThreeLayerFCNet(feature_dim, 1, resolved_hidden_dim)
        elif self.model_profile == "pytorch_template_resnet18_hypothesis_v1":
            resolved_hidden_dim = int(128 if hidden_dim is None else hidden_dim)
            self.feature_extractor = PytorchTemplateResNet18FeatureExtractor1D()
            template_feature_dim = int(self.feature_extractor.output_channels)
            self.classifier = TemplateThreeLayerFCNet(template_feature_dim, self.num_tx, resolved_hidden_dim)
            self.estimate_network = TemplateProjectorFCNet(template_feature_dim, 1, resolved_hidden_dim)
        else:
            raise ValueError(
                "model_profile must be one of: standard_resnet18, pytorch_template_resnet18_hypothesis_v1"
            )

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

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path after training: use E and C, drop estimate network T."""
        features, _ = self.feature_extractor(x, return_activations=False)
        return self.classifier(features)

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        """Export the post-training E/C state; the estimate network T is dropped."""
        state = self.state_dict()
        return {
            key: value.detach().clone()
            for key, value in state.items()
            if key.startswith("feature_extractor.") or key.startswith("classifier.")
        }
