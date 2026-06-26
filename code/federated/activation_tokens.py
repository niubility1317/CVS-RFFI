from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class ActivationTokenPacket:
    route: str
    token: torch.Tensor
    original_shape: Sequence[int]
    payload_bytes: int
    raw_bytes: int
    compression_ratio: float
    quant_bits: int = 0
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None
    quantization_error: float = float("nan")
    rank: int = 0
    sketch_dim: int = 0


class ActivationTokenCodec:
    """Compact activation-token codec for Split-BEX02 approximation diagnostics."""

    def __init__(
        self,
        *,
        route: str = "none",
        quant_bits: int = 8,
        sketch_dim: int = 64,
        rank: int = 8,
        seed: int = 0,
    ):
        self.route = str(route or "none").lower()
        self.quant_bits = max(2, min(16, int(quant_bits)))
        self.sketch_dim = max(1, int(sketch_dim))
        self.rank = max(1, int(rank))
        self.seed = int(seed)

    def encode(self, features: torch.Tensor) -> ActivationTokenPacket:
        if not torch.is_tensor(features):
            raise TypeError("features must be a tensor")
        x = features.detach().cpu().float()
        raw_bytes = int(x.numel()) * int(x.element_size())
        route = self.route
        if route in {"", "none", "raw"}:
            token = x.clone()
            payload = raw_bytes
            return ActivationTokenPacket(
                route="none",
                token=token,
                original_shape=tuple(x.shape),
                payload_bytes=payload,
                raw_bytes=raw_bytes,
                compression_ratio=1.0,
                quant_bits=0,
                quantization_error=0.0,
            )
        if route in {"quantized", "quant", "q"}:
            return self._encode_quantized(x, raw_bytes)
        if route in {"sketch", "sketched"}:
            return self._encode_sketch(x, raw_bytes)
        if route in {"lowrank", "low_rank"}:
            return self._encode_lowrank(x, raw_bytes)
        raise ValueError("--activation_token_route must be one of: none, quantized, sketch, lowrank")

    def decode(self, packet: ActivationTokenPacket) -> torch.Tensor:
        if packet.route == "quantized":
            if packet.scale is None or packet.zero_point is None:
                raise ValueError("quantized token packet requires scale and zero_point")
            decoded = (packet.token.float() - packet.zero_point.float()) * packet.scale.float()
            return decoded.view(tuple(packet.original_shape))
        if packet.route == "none":
            return packet.token.float().view(tuple(packet.original_shape))
        raise ValueError(f"decode is only exact for none/quantized tokens, got route={packet.route}")

    def _encode_quantized(self, x: torch.Tensor, raw_bytes: int) -> ActivationTokenPacket:
        qmax = float((1 << self.quant_bits) - 1)
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min).clamp_min(1e-12) / qmax
        zero_point = torch.round(-x_min / scale).clamp(0.0, qmax)
        token = torch.round(x / scale + zero_point).clamp(0.0, qmax).to(torch.uint8 if self.quant_bits <= 8 else torch.int16)
        decoded = (token.float() - zero_point.float()) * scale.float()
        qerr = float(torch.mean((decoded - x) ** 2).sqrt().item())
        # The token is kept in a tensor for transport/diagnostics here; count actual tensor bytes,
        # not an ideal bit-packed lower bound.
        payload = int(token.numel()) * int(token.element_size()) + 8
        return ActivationTokenPacket(
            route="quantized",
            token=token,
            original_shape=tuple(x.shape),
            payload_bytes=payload,
            raw_bytes=raw_bytes,
            compression_ratio=float(payload / max(1, raw_bytes)),
            quant_bits=int(self.quant_bits),
            scale=scale.detach().cpu(),
            zero_point=zero_point.detach().cpu(),
            quantization_error=qerr,
        )

    def _encode_sketch(self, x: torch.Tensor, raw_bytes: int) -> ActivationTokenPacket:
        flat = x.view(int(x.size(0)), -1) if x.dim() > 1 else x.view(1, -1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(flat.size(1)) * 13 + self.sketch_dim)
        proj = torch.randn(flat.size(1), self.sketch_dim, generator=generator) / math.sqrt(float(self.sketch_dim))
        token = flat @ proj
        payload = int(token.numel()) * int(token.element_size())
        return ActivationTokenPacket(
            route="sketch",
            token=token,
            original_shape=tuple(x.shape),
            payload_bytes=payload,
            raw_bytes=raw_bytes,
            compression_ratio=float(payload / max(1, raw_bytes)),
            sketch_dim=int(self.sketch_dim),
            quantization_error=float("nan"),
        )

    def _encode_lowrank(self, x: torch.Tensor, raw_bytes: int) -> ActivationTokenPacket:
        flat = x.view(int(x.size(0)), -1) if x.dim() > 1 else x.view(1, -1)
        k = max(1, min(int(self.rank), min(flat.shape)))
        try:
            u, s, vh = torch.linalg.svd(flat, full_matrices=False)
            token = torch.cat([u[:, :k].reshape(-1), s[:k].reshape(-1), vh[:k, :].reshape(-1)], dim=0)
        except Exception:
            token = flat[:, :k].contiguous().reshape(-1)
        payload = int(token.numel()) * int(token.element_size())
        return ActivationTokenPacket(
            route="lowrank",
            token=token,
            original_shape=tuple(x.shape),
            payload_bytes=payload,
            raw_bytes=raw_bytes,
            compression_ratio=float(payload / max(1, raw_bytes)),
            rank=int(k),
            quantization_error=float("nan"),
        )
