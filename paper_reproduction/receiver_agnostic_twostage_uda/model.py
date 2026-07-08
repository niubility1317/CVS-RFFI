from __future__ import annotations

import torch
from torch import nn

from baselines.common.grl import gradient_reverse


def _to_paper_conv_input(x: torch.Tensor) -> torch.Tensor:
    """Convert WiSig IQ tensors to the paper's Conv2D input layout [B,1,256,2]."""
    if x.ndim == 3 and x.shape[1] == 2:
        if x.shape[2] != 256:
            raise ValueError("paper-faithful IQ inputs must contain exactly 256 samples")
        return x.transpose(1, 2).unsqueeze(1)
    if x.ndim == 3 and x.shape[-1] == 2:
        if x.shape[1] != 256:
            raise ValueError("paper-faithful IQ inputs must contain exactly 256 samples")
        return x.unsqueeze(1)
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] == 2:
        if x.shape[2] != 256:
            raise ValueError("paper-faithful IQ inputs must contain exactly 256 samples")
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
        self.domain_classifier = DenseReLUHead(feature_dim, 2, classifier_hidden_dim)

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
