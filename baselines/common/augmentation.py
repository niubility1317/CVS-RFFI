from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from .spectrogram import ensure_iq_2xl

try:
    from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario
    from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
    from baseline_origin_sat_view import parse_sat_view_schedule
except Exception:
    parse_sat_scenarios = None
    sat_channel_config_for_scenario = None
    SatSimConfig = None
    apply_sat_gnd_channel_batch = None
    parse_sat_view_schedule = None


class OnlineRFChannelAugment:
    """Approximate TDL/Doppler/AWGN augmentation for IQ tensors.

    This is a practical reproduction of the channel/noise augmentation used by
    the compared papers. It keeps the sample length fixed and avoids operations
    that intentionally erase transmitter hardware fingerprints.
    """

    def __init__(
        self,
        sample_rate: float,
        rms_delay_ns_range: Tuple[float, float] = (5.0, 300.0),
        doppler_hz_range: Tuple[float, float] = (0.0, 5.0),
        snr_db_range: Tuple[float, float] = (10.0, 40.0),
        max_taps: int = 8,
        p: float = 1.0,
        rms_normalize: bool = True,
    ):
        self.sample_rate = float(sample_rate)
        self.rms_delay_ns_range = tuple(float(x) for x in rms_delay_ns_range)
        self.doppler_hz_range = tuple(float(x) for x in doppler_hz_range)
        self.snr_db_range = tuple(float(x) for x in snr_db_range)
        self.max_taps = max(1, int(max_taps))
        self.p = float(p)
        self.rms_normalize = bool(rms_normalize)

    @staticmethod
    def _rand_uniform(lo: float, hi: float, device) -> torch.Tensor:
        return torch.empty((), device=device).uniform_(float(lo), float(hi))

    def _random_taps(self, device, dtype) -> torch.Tensor:
        n_taps = int(torch.randint(1, self.max_taps + 1, (1,), device=device).item())
        delay_ns = self._rand_uniform(*self.rms_delay_ns_range, device=device).clamp_min(1e-3)
        tau = torch.arange(n_taps, device=device, dtype=dtype)
        decay = torch.exp(-tau / delay_ns.to(dtype=dtype).clamp_min(1.0))
        phase = 2.0 * math.pi * torch.rand(n_taps, device=device, dtype=dtype)
        real = torch.randn(n_taps, device=device, dtype=dtype) * torch.cos(phase)
        imag = torch.randn(n_taps, device=device, dtype=dtype) * torch.sin(phase)
        taps = torch.complex(real, imag) * decay
        return taps / torch.sqrt((taps.real.square() + taps.imag.square()).sum().clamp_min(1e-8))

    @staticmethod
    def _same_length_conv(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        # x: [1, 2, L], taps: complex [K]
        k = int(taps.numel())
        pad = k // 2
        xr = x[:, 0:1]
        xi = x[:, 1:2]
        hr = taps.real.flip(0).view(1, 1, k)
        hi = taps.imag.flip(0).view(1, 1, k)
        yr = F.conv1d(xr, hr, padding=pad) - F.conv1d(xi, hi, padding=pad)
        yi = F.conv1d(xr, hi, padding=pad) + F.conv1d(xi, hr, padding=pad)
        y = torch.cat([yr, yi], dim=1)
        length = x.size(-1)
        if y.size(-1) > length:
            start = (y.size(-1) - length) // 2
            y = y[..., start : start + length]
        elif y.size(-1) < length:
            y = F.pad(y, (0, length - y.size(-1)))
        return y

    @staticmethod
    def _rms_normalize(iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms = torch.sqrt(iq[:, 0].square().add(iq[:, 1].square()).mean(dim=-1, keepdim=True).clamp_min(eps))
        return iq / rms.unsqueeze(1)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        x = ensure_iq_2xl(iq)
        was_single = iq.dim() == 2
        if self.p <= 0 or float(torch.rand(())) > self.p:
            return iq.clone()

        out = []
        for sample in x:
            y = sample.unsqueeze(0)
            taps = self._random_taps(y.device, y.dtype)
            y = self._same_length_conv(y, taps)

            fd = self._rand_uniform(*self.doppler_hz_range, device=y.device).to(dtype=y.dtype)
            if float(fd.abs()) > 0:
                t = torch.arange(y.size(-1), device=y.device, dtype=y.dtype) / self.sample_rate
                phase = 2.0 * math.pi * fd * t
                cr = torch.cos(phase)
                ci = torch.sin(phase)
                yr = y[:, 0] * cr - y[:, 1] * ci
                yi = y[:, 0] * ci + y[:, 1] * cr
                y = torch.stack([yr, yi], dim=1)

            snr_db = self._rand_uniform(*self.snr_db_range, device=y.device).to(dtype=y.dtype)
            sig_power = y[:, 0].square().add(y[:, 1].square()).mean().clamp_min(1e-8)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(y) * torch.sqrt(noise_power / 2.0)
            y = y + noise
            if self.rms_normalize:
                y = self._rms_normalize(y)
            out.append(torch.nan_to_num(y.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0))
        yb = torch.stack(out, dim=0)
        return yb[0] if was_single else yb


def add_sat_channel_view_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--use_sat_channel_view_aug", action="store_true")
    parser.add_argument("--use_concat_sat_channel_aug", action="store_true")
    parser.add_argument("--concat_sat_ce_only", action="store_true")
    parser.add_argument("--concat_sat_start_epoch", type=int, default=80)
    parser.add_argument("--lambda_sat_cls", type=float, default=0.68)
    parser.add_argument("--lambda_sat_cons", type=float, default=0.0)
    parser.add_argument("--sat_train_scenario", type=str, default="clear_leo")
    parser.add_argument("--sat_train_scenarios", type=str, default="")
    parser.add_argument("--sat_view_schedule", type=str, default="")
    parser.add_argument("--sat_view_prob", type=float, default=1.0)
    parser.add_argument("--sat_view_seed", type=int, default=2027)
    return parser


class SatGroundChannelViewAugment:
    """LEO satellite-ground channel view augmentation for CVS baselines."""

    def __init__(
        self,
        *,
        scenarios: Iterable[str],
        fs_hz: float = 25e6,
        fc_hz: float = 2.462e9,
        p: float = 1.0,
        seed: int = 2027,
        schedule: str = "",
    ):
        if SatSimConfig is None or apply_sat_gnd_channel_batch is None or sat_channel_config_for_scenario is None:
            raise ImportError("sat_channel.py and training_controls.py are required for satellite channel view augmentation.")
        names = [str(s).strip().lower().replace("-", "_") for s in scenarios if str(s).strip()]
        if not names:
            names = ["clear_leo"]
        self.schedule = str(schedule or "").strip()
        self.stages = (
            parse_sat_view_schedule(self.schedule, default_prob=float(p))
            if self.schedule and parse_sat_view_schedule is not None
            else tuple()
        )
        scheduled_names = [scenario for stage in self.stages for scenario in stage.scenarios]
        self.scenarios = list(dict.fromkeys(scheduled_names or names))
        self.fs_hz = float(fs_hz)
        self.fc_hz = float(fc_hz)
        self.p = max(0.0, min(1.0, float(p)))
        self.seed = int(seed)
        self.epoch = 1
        self._generators: Dict[str, torch.Generator] = {}
        self._configs = {name: self._make_config(name) for name in self.scenarios}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def _epoch_policy(self) -> tuple[tuple[str, ...], float]:
        if not self.stages:
            return tuple(self.scenarios), self.p
        current = self.stages[0]
        for stage in self.stages:
            if self.epoch >= int(stage.start_epoch):
                current = stage
            else:
                break
        return tuple(current.scenarios), float(current.view_prob)

    def _make_config(self, scenario: str):
        kwargs = sat_channel_config_for_scenario(scenario)
        kwargs["fs_hz"] = self.fs_hz
        kwargs["fc_hz"] = self.fc_hz
        return SatSimConfig(**kwargs)

    def _generator_for(self, device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            try:
                gen = torch.Generator(device=device)
            except Exception:
                gen = torch.Generator()
            gen.manual_seed(self.seed)
            self._generators[key] = gen
        return self._generators[key]

    @staticmethod
    def _safe_iq(x: torch.Tensor, clamp: float = 8.0) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(-float(clamp), float(clamp))

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        x = ensure_iq_2xl(iq)
        was_single = iq.dim() == 2
        gen = self._generator_for(x.device)
        scenarios, view_prob = self._epoch_policy()
        if view_prob <= 0 or float(torch.rand((), device=x.device, generator=gen)) > view_prob:
            out = x.clone()
        else:
            scenario_idx = int(torch.randint(0, len(scenarios), (1,), device=x.device, generator=gen).item())
            cfg = self._configs[scenarios[scenario_idx]]
            y, _, _ = apply_sat_gnd_channel_batch(self._safe_iq(x), cfg, gen=gen, return_meta=False)
            out = y.to(device=x.device, dtype=x.dtype)
        return out[0] if was_single else out


def build_sat_channel_view_augment(args: Any) -> Optional[SatGroundChannelViewAugment]:
    if not (
        bool(getattr(args, "use_sat_channel_view_aug", False))
        or bool(getattr(args, "use_concat_sat_channel_aug", False))
    ):
        return None
    if parse_sat_scenarios is None:
        raise ImportError("training_controls.py is required for --use_sat_channel_view_aug.")
    raw = str(getattr(args, "sat_train_scenarios", "") or "")
    scenarios = parse_sat_scenarios(raw) if raw.strip() else parse_sat_scenarios(getattr(args, "sat_train_scenario", "clear_leo"))
    return SatGroundChannelViewAugment(
        scenarios=scenarios,
        fs_hz=float(getattr(args, "sat_fs_hz", 25e6)),
        fc_hz=float(getattr(args, "sat_fc_hz", 2.462e9)),
        p=float(getattr(args, "sat_view_prob", 1.0)),
        seed=int(getattr(args, "sat_view_seed", 2027)),
        schedule=str(getattr(args, "sat_view_schedule", "") or ""),
    )


def supervised_sat_view_batch(batch: Dict[str, Any], device, sat_augment: Optional[SatGroundChannelViewAugment]) -> Dict[str, Any]:
    x_clean = batch["iq"].to(device)
    if sat_augment is None:
        out = dict(batch)
        out["iq"] = x_clean
        return out

    x_sat = sat_augment(x_clean)
    out: Dict[str, Any] = {}
    batch_size = int(x_clean.size(0))
    for key, value in batch.items():
        if key == "iq":
            out[key] = torch.cat([x_clean, x_sat], dim=0)
        elif torch.is_tensor(value) and value.dim() > 0 and int(value.size(0)) == batch_size:
            v = value.to(device)
            out[key] = torch.cat([v, v], dim=0)
        else:
            out[key] = value
    return out


def concat_sat_ce_only_loss(
    clean_loss: torch.Tensor,
    sat_tx_logits: torch.Tensor,
    tx_labels: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add only transmitter CE from the satellite view to a clean objective."""
    sat_ce = F.cross_entropy(sat_tx_logits, tx_labels)
    return clean_loss + float(weight) * sat_ce, sat_ce


def forward_concat_sat_ce_only(
    model,
    clean_iq: torch.Tensor,
    sat_iq: torch.Tensor,
    *,
    logits_key: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Run one clean+satellite concatenated forward and split the two halves."""
    if clean_iq.shape != sat_iq.shape:
        raise ValueError("clean and satellite IQ batches must have identical shapes")
    clean_count = int(clean_iq.size(0))
    joint = model(torch.cat([clean_iq, sat_iq], dim=0), **(model_kwargs or {}))
    joint_count = 2 * clean_count
    clean_outputs: Dict[str, torch.Tensor] = {}
    for key, value in joint.items():
        if not torch.is_tensor(value) or value.dim() == 0 or int(value.size(0)) != joint_count:
            raise ValueError(f"model output {key!r} is not aligned with the concatenated batch")
        clean_outputs[key] = value[:clean_count]
    return clean_outputs, joint[logits_key][clean_count:]


def resolve_sat_view_stage(schedule: str, epoch: int) -> tuple[tuple[str, ...], float]:
    if parse_sat_view_schedule is None:
        raise ImportError("baseline_origin_sat_view.py is required for satellite curricula.")
    stages = parse_sat_view_schedule(str(schedule), default_prob=1.0)
    if not stages:
        raise ValueError("satellite view schedule must not be empty")
    current = stages[0]
    for stage in stages:
        if int(epoch) >= int(stage.start_epoch):
            current = stage
        else:
            break
    return tuple(current.scenarios), float(current.view_prob)
