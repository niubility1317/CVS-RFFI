from __future__ import annotations

import math
from typing import Any, Optional

import torch

from .style_packet import StylePacket


def _safe_flat(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)


def _stat_float(x: torch.Tensor, default: float = 0.0) -> float:
    if not torch.is_tensor(x) or x.numel() == 0:
        return float(default)
    return float(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).mean().item())


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _db20(value: torch.Tensor, eps: float) -> torch.Tensor:
    return 20.0 * torch.log10(value.clamp_min(eps))


def _wrapped_delta(phase: torch.Tensor) -> torch.Tensor:
    delta = phase[..., 1:] - phase[..., :-1] if phase.size(-1) > 1 else phase.new_zeros(phase.shape[:-1] + (1,))
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _mean_numeric_stats(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({str(key) for stats in items for key in stats.keys()})
    out: dict[str, float] = {}
    for key in keys:
        vals: list[float] = []
        for stats in items:
            try:
                val = float(stats[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(val):
                vals.append(val)
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


class RFStyleExtractor:
    """Extracts privacy-conscious, class-marginalized RF style statistics."""

    def __init__(self, fft_bins: int = 16, eps: float = 1e-6, sample_rate_hz: float = 0.0):
        self.fft_bins = max(4, int(fft_bins))
        self.eps = float(eps)
        self.sample_rate_hz = float(sample_rate_hz or 0.0)

    def extract(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        client_id: str,
        round_idx: int,
        d_raw: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StylePacket:
        x = _safe_flat(x)
        if x.dim() < 2:
            raise ValueError("RFStyleExtractor expects at least [batch, channel, time] shaped IQ tensors.")
        count = int(x.size(0))
        if y is not None and count > 0:
            labels = torch.as_tensor(y, device=x.device).detach().view(-1)
            if int(labels.numel()) == count:
                per_class: list[dict[str, float]] = []
                for cls in torch.unique(labels):
                    mask = labels == cls
                    if not bool(mask.any()):
                        continue
                    feature_cls = None
                    if feature is not None and torch.is_tensor(feature) and int(feature.size(0)) == count:
                        feature_cls = feature[mask]
                    packet = self.extract(
                        x[mask],
                        None,
                        client_id=client_id,
                        round_idx=round_idx,
                        d_raw=None,
                        feature=feature_cls,
                        metadata=None,
                    )
                    per_class.append(dict(packet.stats))
                if per_class:
                    meta = dict(metadata or {})
                    meta["style_class_balance"] = "equal_present_classes"
                    meta["style_num_classes"] = len(per_class)
                    if d_raw is not None:
                        d = torch.as_tensor(d_raw).view(-1)
                        meta["num_raw_domains"] = int(torch.unique(d).numel()) if d.numel() else 0
                    return StylePacket(
                        client_id=str(client_id),
                        round_idx=int(round_idx),
                        count=count,
                        stats=_mean_numeric_stats(per_class),
                        metadata=meta,
                    )
        channels = x
        if x.dim() == 2:
            i = x
            q = torch.zeros_like(i)
        else:
            i = channels[:, 0, :]
            q = channels[:, 1, :] if channels.size(1) > 1 else torch.zeros_like(i)
        amp = torch.sqrt(i * i + q * q + self.eps)
        phase = torch.atan2(q, i + self.eps)
        phase_diff = _wrapped_delta(phase)
        phase_diff_mean_per_sample = torch.nan_to_num(phase_diff.float(), nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)
        phase_residual = phase_diff - phase_diff_mean_per_sample.unsqueeze(-1)
        spec = torch.fft.fft(torch.complex(i, q), dim=-1).abs()[..., : max(1, i.size(-1) // 2 + 1)]
        if spec.size(-1) > self.fft_bins:
            spec = torch.nn.functional.adaptive_avg_pool1d(spec.unsqueeze(1), self.fft_bins).squeeze(1)
        freq = torch.linspace(0.0, 1.0, steps=spec.size(-1), device=spec.device, dtype=spec.dtype)
        spec_mass = spec.sum(dim=-1).clamp_min(self.eps)
        centroid = (spec * freq).sum(dim=-1) / spec_mass
        bandwidth = torch.sqrt(((freq - centroid.unsqueeze(-1)) ** 2 * spec).sum(dim=-1) / spec_mass.clamp_min(self.eps))
        spec_prob = (spec / spec_mass.unsqueeze(-1)).clamp_min(self.eps)
        spec_entropy = -(spec_prob * torch.log(spec_prob)).sum(dim=-1) / max(1.0, float(torch.log(torch.tensor(float(spec_prob.size(-1)), device=spec_prob.device)).item()))
        spec_flatness = torch.exp(torch.log(spec.clamp_min(self.eps)).mean(dim=-1)) / spec.mean(dim=-1).clamp_min(self.eps)
        cdf = torch.cumsum(spec, dim=-1) / spec_mass.unsqueeze(-1)
        cutoff_idx = (cdf >= 0.95).float().argmax(dim=-1).to(spec.dtype)
        cutoff_frac = cutoff_idx / max(1.0, float(spec.size(-1) - 1))
        i_std_t = i.std(dim=-1, unbiased=False).clamp_min(self.eps)
        q_std_t = q.std(dim=-1, unbiased=False).clamp_min(self.eps)
        i_center = i - i.mean(dim=-1, keepdim=True)
        q_center = q - q.mean(dim=-1, keepdim=True)
        iq_corr = (i_center * q_center).mean(dim=-1) / (i_std_t * q_std_t).clamp_min(self.eps)
        iq_corr = iq_corr.clamp(-0.95, 0.95)
        amp_p95 = torch.quantile(amp.float().flatten(start_dim=1), 0.95, dim=-1)
        amp_mean = amp.mean(dim=-1)
        amp_std = amp.std(dim=-1, unbiased=False)
        snr_proxy = _db20(amp_mean / amp_std.clamp_min(self.eps), self.eps).clamp(5.0, 80.0)
        iq_rms = torch.sqrt(torch.mean(x * x).clamp_min(self.eps)) if x.numel() else torch.tensor(1.0, device=x.device)
        cfo_cycles = (phase_diff_mean_per_sample / (2.0 * torch.pi)).clamp(-0.49, 0.49)
        cfo_hz = cfo_cycles * float(self.sample_rate_hz) if self.sample_rate_hz > 0.0 else cfo_cycles.new_zeros(cfo_cycles.shape)

        stats: dict[str, float] = {
            "iq_mean": _stat_float(x),
            "iq_std": float(x.std(unbiased=False).item()) if x.numel() else 0.0,
            "iq_rms": float(iq_rms.item()) if x.numel() else 0.0,
            "i_mean": _stat_float(i),
            "q_mean": _stat_float(q),
            "i_std": float(i.std(unbiased=False).item()) if i.numel() else 0.0,
            "q_std": float(q.std(unbiased=False).item()) if q.numel() else 0.0,
            "amp_mean": _stat_float(amp),
            "amp_std": float(amp.std(unbiased=False).item()) if amp.numel() else 0.0,
            "phase_diff_mean": _stat_float(phase_diff),
            "phase_diff_std": float(phase_diff.std(unbiased=False).item()) if phase_diff.numel() else 0.0,
            "spectrum_centroid": _stat_float(centroid),
            "spectrum_bandwidth": _stat_float(bandwidth),
            "spectrum_log_energy": _stat_float(torch.log1p(spec)),
            "phys_cfo_cycles_per_sample": _stat_float(cfo_cycles),
            "phys_cfo_hz": _stat_float(cfo_hz),
            "phys_sro_ppm": 0.0,
            "phys_agc_gain_db": float(_db20(iq_rms.view(1), self.eps).item()) if x.numel() else 0.0,
            "phys_softclip_level": max(self.eps, _stat_float(amp_p95, 1.0)),
            "phys_iq_gain_imbalance_db": _clamp_float(_stat_float(_db20(i_std_t / q_std_t, self.eps), 0.0), -12.0, 12.0),
            "phys_iq_phase_imbalance_deg": _clamp_float(_stat_float(torch.rad2deg(torch.asin(iq_corr)), 0.0), -20.0, 20.0),
            "phys_phase_noise_std": _clamp_float(float(phase_residual.std(unbiased=False).item()) if phase_residual.numel() else 0.0, 0.0, 0.05),
            "phys_awgn_snr_db": _stat_float(snr_proxy, 80.0),
            "phys_multipath_strength": _clamp_float(_stat_float(1.0 - spec_entropy, 0.0), 0.0, 1.0),
            "phys_lowpass_cutoff_frac": _clamp_float(_stat_float(cutoff_frac, 1.0), 0.05, 1.0),
            "phys_lowpass_transition_frac": _clamp_float(_stat_float(spec_flatness, 0.05) * 0.10, 0.01, 0.15),
        }
        if feature is not None and torch.is_tensor(feature) and feature.numel() > 0:
            feat = _safe_flat(feature).view(feature.size(0), -1)
            stats["feature_mean"] = _stat_float(feat)
            stats["feature_std"] = float(feat.std(unbiased=False).item())
        meta = dict(metadata or {})
        if d_raw is not None:
            d = torch.as_tensor(d_raw).view(-1)
            meta["num_raw_domains"] = int(torch.unique(d).numel()) if d.numel() else 0
        return StylePacket(client_id=str(client_id), round_idx=int(round_idx), count=count, stats=stats, metadata=meta)
