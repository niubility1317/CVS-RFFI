from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import TimeEmbedding


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(int(p.numel()) for p in model.parameters() if (p.requires_grad or not trainable_only))


def _groups(channels: int, requested: int = 8) -> int:
    for g in (requested, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResDSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        groups: int = 8,
        time_dim: int = 64,
        cond_dim: int = 24,
    ) -> None:
        super().__init__()
        pad = (int(kernel_size) // 2) * int(dilation)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=int(kernel_size),
            padding=pad,
            dilation=int(dilation),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(int(time_dim), out_channels)
        self.cond_proj = nn.Linear(int(cond_dim), out_channels)
        self.out = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_norm = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        h = h + self.time_proj(time_emb).unsqueeze(-1) + self.cond_proj(cond).unsqueeze(-1)
        h = self.act(h)
        h = self.out(h)
        h = self.out_norm(h)
        return self.act(h + self.skip(x))


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        return self.conv(x)


class LightweightTemporalAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2))
        return x + self.proj(h)


class CxResUNet1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        condition_dim: int = 24,
        time_embed_dim: int = 64,
        channels: Sequence[int] = (32, 48, 64, 72),
        kernel_size: int = 5,
        stem_kernel_size: int = 7,
        group_norm_groups: int = 8,
        bottleneck_attention: bool = True,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if len(tuple(channels)) != 4:
            raise ValueError("channels must contain four entries.")
        c1, c2, c3, c4 = [int(v) for v in channels]
        self.in_channels = int(in_channels)
        self.condition_dim = int(condition_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.output_activation = str(output_activation)
        self.time_embedding = TimeEmbedding(self.time_embed_dim)
        stem_in = self.in_channels * 2 + self.condition_dim
        self.stem = nn.Sequential(
            nn.Conv1d(stem_in, c1, kernel_size=int(stem_kernel_size), padding=int(stem_kernel_size) // 2),
            nn.GroupNorm(_groups(c1, group_norm_groups), c1),
            nn.SiLU(),
        )
        self.enc1 = ResDSBlock(c1, c1, kernel_size=kernel_size, groups=group_norm_groups, time_dim=time_embed_dim, cond_dim=condition_dim)
        self.down1 = Downsample(c1, c2)
        self.enc2 = ResDSBlock(c2, c2, kernel_size=kernel_size, groups=group_norm_groups, time_dim=time_embed_dim, cond_dim=condition_dim)
        self.down2 = Downsample(c2, c3)
        self.enc3 = ResDSBlock(c3, c3, kernel_size=kernel_size, groups=group_norm_groups, time_dim=time_embed_dim, cond_dim=condition_dim)
        self.down3 = Downsample(c3, c4)
        self.bottleneck = ResDSBlock(
            c4,
            c4,
            kernel_size=kernel_size,
            dilation=2,
            groups=group_norm_groups,
            time_dim=time_embed_dim,
            cond_dim=condition_dim,
        )
        self.attn = LightweightTemporalAttention(c4) if bottleneck_attention else nn.Identity()
        self.up3 = Upsample(c4, c3)
        self.dec3 = ResDSBlock(c3 + c3, c3, kernel_size=kernel_size, groups=group_norm_groups, time_dim=time_embed_dim, cond_dim=condition_dim)
        self.up2 = Upsample(c3, c2)
        self.dec2 = ResDSBlock(c2 + c2, c2, kernel_size=kernel_size, groups=group_norm_groups, time_dim=time_embed_dim, cond_dim=condition_dim)
        self.up1 = Upsample(c2, c1)
        self.dec1 = ResDSBlock(c1 + c1, c1, kernel_size=kernel_size, groups=group_norm_groups, time_dim=time_embed_dim, cond_dim=condition_dim)
        self.head = nn.Sequential(nn.Conv1d(c1, 16, kernel_size=3, padding=1), nn.SiLU(), nn.Conv1d(16, self.in_channels, kernel_size=3, padding=1))

    def forward(self, *, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or y.ndim != 3 or x_t.shape != y.shape:
            raise ValueError("x_t and y must share shape [B, 2, T].")
        if c.ndim != 2 or c.size(0) != y.size(0):
            raise ValueError("c must be shaped [B, condition_dim].")
        time_emb = self.time_embedding(t.to(device=y.device))
        c = c.to(device=y.device, dtype=y.dtype)
        cond_map = c.unsqueeze(-1).expand(-1, -1, y.size(-1))
        h = self.stem(torch.cat([x_t, y, cond_map], dim=1))
        e1 = self.enc1(h, time_emb, c)
        e2 = self.enc2(self.down1(e1), time_emb, c)
        e3 = self.enc3(self.down2(e2), time_emb, c)
        b = self.attn(self.bottleneck(self.down3(e3), time_emb, c))
        d3 = self.up3(b, e3.size(-1))
        d3 = self.dec3(torch.cat([d3, e3], dim=1), time_emb, c)
        d2 = self.up2(d3, e2.size(-1))
        d2 = self.dec2(torch.cat([d2, e2], dim=1), time_emb, c)
        d1 = self.up1(d2, e1.size(-1))
        d1 = self.dec1(torch.cat([d1, e1], dim=1), time_emb, c)
        out = self.head(d1)
        if self.output_activation == "tanh":
            out = torch.tanh(out)
        return out
