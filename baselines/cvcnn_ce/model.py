from __future__ import annotations

import torch
from torch import nn

from model import SincConv1d


class ComplexConv1d(nn.Module):
    """Strict complex convolution for `[real, imag]` channel pairs."""

    def __init__(self, in_complex: int, out_complex: int, kernel_size: int = 5, stride: int = 1, bias: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.in_complex = int(in_complex)
        self.real = nn.Conv1d(in_complex, out_complex, kernel_size, stride=stride, padding=padding, bias=bias)
        self.imag = nn.Conv1d(in_complex, out_complex, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x[:, : self.in_complex]
        xi = x[:, self.in_complex :]
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexBlock(nn.Module):
    def __init__(self, in_complex: int, out_complex: int, kernel_size: int = 5, pool: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            ComplexConv1d(in_complex, out_complex, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(2 * out_complex),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(pool) if pool > 1 else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SincComplexStem(nn.Module):
    """SincNet first layer applied symmetrically to I/Q branches."""

    def __init__(
        self,
        out_complex: int,
        kernel_size: int = 79,
        pool: int = 2,
        sample_rate_hz: float = 25e6,
        input_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_complex = int(out_complex)
        self.sinc = SincConv1d(
            out_channels=int(out_complex),
            kernel_size=int(kernel_size),
            sample_rate=float(sample_rate_hz),
            dataset="wisig",
            input_len=int(input_len),
            pad_crop_mode="center",
        )
        self.net = nn.Sequential(
            nn.BatchNorm1d(2 * int(out_complex)),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(pool) if pool > 1 else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"SincComplexStem expects [B,2,L], got {tuple(x.shape)}")
        i = self.sinc(x[:, 0:1, :])
        q = self.sinc(x[:, 1:2, :])
        return self.net(torch.cat([i, q], dim=1))


class _CVCNNTail(nn.Module):
    """Shared CVCNN layers after the first front-end layer."""

    def __init__(
        self,
        num_classes: int,
        input_len: int = 256,
        base_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.0,
        stem: nn.Module | None = None,
    ):
        super().__init__()
        ch = int(base_channels)
        self.stem = stem if stem is not None else ComplexBlock(1, ch, kernel_size=7, pool=2, dropout=dropout)
        self.tail = nn.Sequential(
            ComplexBlock(ch, ch * 2, kernel_size=5, pool=2, dropout=dropout),
            ComplexBlock(ch * 2, ch * 4, kernel_size=3, pool=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * ch * 4, int(embedding_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))
        self.input_len = int(input_len)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"CVCNN expects [B,2,L], got {tuple(x.shape)}")
        z = self.tail(self.stem(x))
        return self.embedding(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class BasicCVCNN(_CVCNNTail):
    """Plain CVCNN baseline trained only with cross entropy."""

    def __init__(
        self,
        num_classes: int,
        input_len: int = 256,
        base_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            input_len=input_len,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            stem=ComplexBlock(1, int(base_channels), kernel_size=7, pool=2, dropout=dropout),
        )


class SincCVCNN(_CVCNNTail):
    """CVCNN whose only architectural change is a SincNet first layer."""

    def __init__(
        self,
        num_classes: int,
        input_len: int = 256,
        base_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.0,
        sinc_kernel_size: int = 79,
        sample_rate_hz: float = 25e6,
    ):
        super().__init__(
            num_classes=num_classes,
            input_len=input_len,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            stem=SincComplexStem(
                out_complex=int(base_channels),
                kernel_size=int(sinc_kernel_size),
                pool=2,
                sample_rate_hz=float(sample_rate_hz),
                input_len=int(input_len),
                dropout=dropout,
            ),
        )
