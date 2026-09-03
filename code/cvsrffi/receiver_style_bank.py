from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def extract_receiver_statistics(iq: torch.Tensor) -> torch.Tensor:
    """Extract nine source-receiver summaries without retaining sample IQ."""

    if iq.ndim != 3 or int(iq.shape[1]) != 2:
        raise ValueError("iq must have shape [batch,2,time]")
    x = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    amplitude = x.abs().clamp_min(1e-8)
    log_amplitude = amplitude.log()
    phase_increment = torch.angle(x[:, 1:] * x[:, :-1].conj())
    i, q = x.real, x.imag
    centered_i = i - i.mean(dim=1, keepdim=True)
    centered_q = q - q.mean(dim=1, keepdim=True)
    spectrum = torch.fft.fft(x, dim=1).abs().clamp_min(1e-8).log()
    return torch.stack(
        (
            log_amplitude.mean(dim=1),
            log_amplitude.std(dim=1, unbiased=False),
            spectrum.std(dim=1, unbiased=False),
            centered_i.square().mean(dim=1),
            centered_q.square().mean(dim=1),
            (centered_i * centered_q).mean(dim=1),
            phase_increment.mean(dim=1),
            phase_increment.std(dim=1, unbiased=False),
            x.mean(dim=1).abs(),
        ),
        dim=1,
    )


@dataclass
class ReceiverStyleBank:
    mean: torch.Tensor
    basis: torch.Tensor
    coefficients: torch.Tensor
    receiver_ids: torch.Tensor
    explained_variance: torch.Tensor

    @classmethod
    def fit(
        cls,
        statistics: torch.Tensor,
        *,
        receiver_ids: torch.Tensor,
        role: str,
        rank: int,
    ) -> "ReceiverStyleBank":
        if str(role) != "source":
            raise ValueError("ReceiverStyleBank is source-only")
        statistics = torch.as_tensor(statistics, dtype=torch.float32)
        receiver_ids = torch.as_tensor(receiver_ids, dtype=torch.long).reshape(-1)
        if statistics.ndim != 2 or statistics.shape[0] != receiver_ids.numel():
            raise ValueError("statistics and receiver_ids must align")
        if not bool(torch.isfinite(statistics).all()):
            raise ValueError("receiver statistics must be finite")
        rank = min(int(rank), int(statistics.shape[0]) - 1, int(statistics.shape[1]))
        if rank < 1:
            raise ValueError("receiver style rank must be positive and identifiable")
        mean = statistics.mean(dim=0)
        centered = statistics - mean
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:rank].t().contiguous()
        coefficients = centered @ basis
        variance = singular.square()
        explained = variance[:rank] / variance.sum().clamp_min(1e-12)
        return cls(mean, basis, coefficients, receiver_ids, explained)

    def sample(
        self,
        *,
        batch_size: int,
        seed: int,
        extension: Sequence[float] = (0.8, 1.2),
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        lo, hi = float(extension[0]), float(extension[1])
        if batch_size < 1 or not (0.0 < lo <= hi <= 1.2):
            raise ValueError("invalid receiver style sampling request")
        generator = torch.Generator(device=self.coefficients.device)
        generator.manual_seed(int(seed))
        raw = torch.rand(
            (batch_size, self.coefficients.shape[0]),
            generator=generator,
            device=self.coefficients.device,
        ).clamp_min(1e-8)
        convex = raw / raw.sum(dim=1, keepdim=True)
        coefficient = convex @ self.coefficients
        center = self.coefficients.mean(dim=0, keepdim=True)
        scale = lo + (hi - lo) * torch.rand(
            (batch_size, 1), generator=generator, device=self.coefficients.device
        )
        coefficient = center + scale * (coefficient - center)
        return self.mean.unsqueeze(0) + coefficient @ self.basis.t()

    def apply(self, iq: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or int(iq.shape[1]) != 2 or style.shape != (iq.shape[0], self.mean.numel()):
            raise ValueError("iq and receiver style must align")
        x = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        relative = torch.tanh((style.float() - self.mean.to(style.device)) * 0.25)
        gain = torch.exp(0.08 * relative[:, 0]).unsqueeze(1)
        phase = (0.08 * relative[:, 6]).unsqueeze(1)
        image = (0.03 * relative[:, 5]).unsqueeze(1) * x.conj()
        changed = gain * (x + image) * torch.exp(1j * phase)
        alpha = (0.04 * relative[:, 2].abs()).view(-1, 1)
        previous = torch.roll(changed, shifts=1, dims=1)
        previous[:, 0] = changed[:, 0]
        changed = (1.0 - alpha) * changed + alpha * previous
        return torch.stack((changed.real, changed.imag), dim=1).to(dtype=iq.dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "mean": self.mean.detach().cpu(),
            "basis": self.basis.detach().cpu(),
            "coefficients": self.coefficients.detach().cpu(),
            "receiver_ids": self.receiver_ids.detach().cpu(),
            "explained_variance": self.explained_variance.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReceiverStyleBank":
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported receiver style bank version")
        return cls(
            torch.as_tensor(state["mean"], dtype=torch.float32),
            torch.as_tensor(state["basis"], dtype=torch.float32),
            torch.as_tensor(state["coefficients"], dtype=torch.float32),
            torch.as_tensor(state["receiver_ids"], dtype=torch.long),
            torch.as_tensor(state["explained_variance"], dtype=torch.float32),
        )


class OnlineReceiverStyleBank:
    """Streaming source-only wrapper used by the V2 training path."""

    def __init__(self, *, rank: int = 3, min_receivers: int = 3, momentum: float = 0.95) -> None:
        if int(rank) < 1 or int(min_receivers) < 2 or not 0.0 <= float(momentum) < 1.0:
            raise ValueError("invalid online receiver-style configuration")
        self.rank = int(rank)
        self.min_receivers = int(min_receivers)
        self.momentum = float(momentum)
        self._statistics: dict[int, torch.Tensor] = {}
        self._counts: dict[int, int] = {}
        self.bank: ReceiverStyleBank | None = None

    @property
    def ready(self) -> bool:
        return self.bank is not None

    def update(self, iq: torch.Tensor, *, receiver_ids: torch.Tensor, role: str) -> None:
        if str(role) != "source":
            raise ValueError("OnlineReceiverStyleBank is source-only")
        receiver_ids = receiver_ids.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        statistics = extract_receiver_statistics(iq).detach().to(device="cpu", dtype=torch.float32)
        if receiver_ids.numel() != statistics.shape[0]:
            raise ValueError("receiver_ids must align with IQ samples")
        for receiver_id in receiver_ids.unique(sorted=True).tolist():
            selected = receiver_ids.eq(int(receiver_id))
            current = statistics[selected].mean(dim=0)
            old = self._statistics.get(int(receiver_id))
            self._statistics[int(receiver_id)] = (
                current if old is None else self.momentum * old + (1.0 - self.momentum) * current
            )
            self._counts[int(receiver_id)] = self._counts.get(int(receiver_id), 0) + int(selected.sum())
        if len(self._statistics) >= self.min_receivers:
            keys = sorted(self._statistics)
            values = torch.stack([self._statistics[key] for key in keys])
            self.bank = ReceiverStyleBank.fit(
                values,
                receiver_ids=torch.tensor(keys, dtype=torch.long),
                role="source",
                rank=min(self.rank, len(keys) - 1),
            )

    def apply_sampled(self, iq: torch.Tensor, *, seed: int) -> torch.Tensor:
        if self.bank is None:
            return iq
        style = self.bank.sample(batch_size=int(iq.shape[0]), seed=int(seed)).to(iq.device)
        return self.bank.apply(iq, style)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "min_receivers": self.min_receivers,
            "momentum": self.momentum,
            "statistics": {key: value.clone() for key, value in self._statistics.items()},
            "counts": dict(self._counts),
            "bank": None if self.bank is None else self.bank.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rank = int(state["rank"])
        self.min_receivers = int(state["min_receivers"])
        self.momentum = float(state["momentum"])
        self._statistics = {
            int(key): torch.as_tensor(value, dtype=torch.float32).clone()
            for key, value in dict(state.get("statistics", {})).items()
        }
        self._counts = {int(key): int(value) for key, value in dict(state.get("counts", {})).items()}
        bank_state = state.get("bank")
        self.bank = None if bank_state is None else ReceiverStyleBank.from_state_dict(bank_state)
