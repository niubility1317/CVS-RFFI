from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Optional

import torch


ApplySatFn = Callable[..., tuple]

CRRA_NUISANCE_FIELDS = (
    "snr_db",
    "cfo_hz",
    "residual_cfo_hz",
    "fD_hz",
    "pl_db",
    "K_db",
    "theta_deg",
    "h_km",
    "state",
)

CRRA_NUISANCE_SCALES = {
    "snr_db": 20.0,
    "cfo_hz": 100_000.0,
    "residual_cfo_hz": 100_000.0,
    "fD_hz": 100_000.0,
    "pl_db": 200.0,
    "K_db": 20.0,
    "theta_deg": 90.0,
    "h_km": 2_000.0,
    "state": 2.0,
}


@dataclass(frozen=True)
class SatViewStage:
    start_epoch: int
    scenarios: tuple[str, ...]
    view_prob: float


@dataclass
class SatViewTransform:
    x: torch.Tensor
    scenario: str
    stage_start_epoch: int
    stage_index: int
    view_prob: float
    applied: bool
    clean_batch_size: int
    meta: Optional[dict[str, Any]] = None
    nuisance: Optional[torch.Tensor] = None
    nuisance_valid: Optional[torch.Tensor] = None
    nuisance_fields: tuple[str, ...] = ()
    pair_id: Optional[tuple[str, ...]] = None
    physical_sample_id: Optional[tuple[str, ...]] = None
    crop_offset: Optional[torch.Tensor] = None


@dataclass
class BaselineOriginSatViewBatch:
    x: torch.Tensor
    y: torch.Tensor
    d_raw: Optional[torch.Tensor]
    scenario: str
    clean_batch_size: int
    total_batch_size: int
    stage_start_epoch: int
    stage_index: int
    view_prob: float
    applied: bool
    meta: Optional[dict[str, Any]] = None
    nuisance: Optional[torch.Tensor] = None
    nuisance_valid: Optional[torch.Tensor] = None
    nuisance_fields: tuple[str, ...] = ()


def normalize_scenario_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _clamp_prob(value: float) -> float:
    prob = float(value)
    if (not math.isfinite(prob)) or prob < 0.0 or prob > 1.0:
        raise ValueError("satellite view probability must be in [0, 1]")
    return prob


def _parse_explicit_prob(value: str) -> float:
    prob = float(str(value).strip())
    if (not math.isfinite(prob)) or prob < 0.0 or prob > 1.0:
        raise ValueError("satellite view schedule probabilities must be in [0, 1]")
    return prob


def _expand_scenario_token(token: str) -> list[str]:
    raw = str(token or "").strip()
    if not raw:
        return []
    if "*" in raw:
        name, repeat_text = raw.rsplit("*", 1)
        repeat = int(repeat_text.strip())
        if repeat < 1:
            raise ValueError("satellite view schedule repeat counts must be >= 1")
    else:
        name = raw
        repeat = 1
    scenario = normalize_scenario_name(name)
    return [scenario] * repeat if scenario else []


def parse_sat_view_schedule(raw: str, *, default_prob: float = 1.0) -> tuple[SatViewStage, ...]:
    stages: list[SatViewStage] = []
    text = str(raw or "").strip()
    if not text:
        return tuple()
    for part in text.split(";"):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"satellite view schedule stage must use '<epoch>[:/@]<scenarios>': {item!r}")
        head, scenario_text = item.split(":", 1)
        if "@" in head:
            epoch_text, prob_text = head.split("@", 1)
            prob = _parse_explicit_prob(prob_text)
        else:
            epoch_text = head
            prob = _clamp_prob(default_prob)
        start_epoch = int(epoch_text.strip())
        if start_epoch < 1:
            raise ValueError("satellite view schedule epochs are 1-based and must be >= 1")
        scenarios: list[str] = []
        for token in scenario_text.split(","):
            scenarios.extend(_expand_scenario_token(token))
        if not scenarios:
            raise ValueError(f"satellite view schedule stage has no scenarios: {item!r}")
        stages.append(SatViewStage(start_epoch=start_epoch, scenarios=tuple(scenarios), view_prob=prob))
    stages.sort(key=lambda stage: stage.start_epoch)
    if stages and stages[0].start_epoch != 1:
        raise ValueError("satellite view schedules must start at epoch 1")
    starts = [stage.start_epoch for stage in stages]
    if len(starts) != len(set(starts)):
        raise ValueError("satellite view schedule start epochs must be unique")
    return tuple(stages)


def build_default_sat_view_stages(
    *,
    scenarios: Optional[Iterable[str]],
    schedule: str = "",
    default_prob: float = 1.0,
) -> tuple[SatViewStage, ...]:
    parsed = parse_sat_view_schedule(schedule, default_prob=default_prob)
    if parsed:
        return parsed
    names = tuple(normalize_scenario_name(name) for name in (scenarios or []) if normalize_scenario_name(name))
    return (SatViewStage(start_epoch=1, scenarios=names or ("mixed_orbit",), view_prob=_clamp_prob(default_prob)),)


def _concat_optional_domain(d_raw: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if not torch.is_tensor(d_raw):
        return None
    d_view = d_raw.to(device=device)
    return torch.cat([d_view, d_view], dim=0)


def _batch_meta_column(value: Any, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    try:
        if torch.is_tensor(value):
            column = value.detach().to(device=device, dtype=torch.float32).reshape(-1)
        else:
            column = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None
    if column.numel() == 1 and int(batch_size) > 1:
        column = column.expand(int(batch_size))
    if column.numel() != int(batch_size):
        return None
    return column


def _normalize_nuisance_meta(
    meta: Any,
    *,
    scenario: str,
    batch_size: int,
    device: torch.device,
) -> tuple[Optional[dict[str, Any]], Optional[torch.Tensor], Optional[torch.Tensor], tuple[str, ...]]:
    if not isinstance(meta, Mapping):
        return {"scenario": str(scenario), "valid": False}, None, None, ()
    raw = dict(meta)
    raw.setdefault("scenario", str(scenario))
    if "residual_cfo_hz" not in raw and "cfo_hz" in raw:
        raw["residual_cfo_hz"] = raw["cfo_hz"]
    columns = []
    missing_fields = []
    for field in CRRA_NUISANCE_FIELDS:
        column = _batch_meta_column(raw.get(field), int(batch_size), device)
        if column is None:
            missing_fields.append(field)
            column = torch.zeros(int(batch_size), device=device, dtype=torch.float32)
        columns.append(column / float(CRRA_NUISANCE_SCALES[field]))
    nuisance = torch.stack(columns, dim=1)
    finite_valid = torch.isfinite(nuisance).all(dim=1)
    valid = (
        torch.zeros(int(batch_size), dtype=torch.bool, device=device)
        if missing_fields
        else finite_valid
    )
    nuisance = torch.nan_to_num(nuisance, nan=0.0, posinf=0.0, neginf=0.0)
    raw["valid"] = bool(valid.any().item())
    raw["missing_fields"] = tuple(missing_fields)
    return raw, nuisance, valid, CRRA_NUISANCE_FIELDS


normalize_crra_nuisance_meta = _normalize_nuisance_meta


def _expand_nuisance(
    nuisance: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
    clean_count: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if nuisance is None or valid is None:
        return None, None
    clean_nuisance = nuisance.new_zeros((int(clean_count), int(nuisance.size(1))))
    clean_valid = torch.zeros(int(clean_count), dtype=torch.bool, device=valid.device)
    return torch.cat([clean_nuisance, nuisance], dim=0), torch.cat([clean_valid, valid], dim=0)


def _pair_meta_from_batch(
    batch_meta: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Optional[tuple[str, ...]], Optional[torch.Tensor]]:
    if not isinstance(batch_meta, Mapping):
        return None, None
    raw_ids = batch_meta.get("physical_sample_id")
    raw_offsets = batch_meta.get("crop_offset")
    if torch.is_tensor(raw_ids):
        raw_ids = raw_ids.detach().cpu().reshape(-1).tolist()
    if torch.is_tensor(raw_offsets):
        raw_offsets = raw_offsets.detach().to(device=device, dtype=torch.long).reshape(-1)
    if not isinstance(raw_ids, (tuple, list)) or len(raw_ids) != int(batch_size):
        return None, None
    if raw_offsets is None:
        return None, None
    if not torch.is_tensor(raw_offsets):
        try:
            raw_offsets = torch.as_tensor(raw_offsets, dtype=torch.long, device=device).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return None, None
    if int(raw_offsets.numel()) != int(batch_size):
        return None, None
    return tuple(str(value) for value in raw_ids), raw_offsets


class BaselineOriginSatViewAugment:
    """Baseline-origin supervised satellite view generator.

    The module exposes both a transform-only API for auxiliary/federated losses
    and a clean+satellite expansion API for baseline-style supervised batches.
    """

    def __init__(
        self,
        *,
        scenarios: Optional[Iterable[str]] = None,
        schedule: str = "",
        p: float = 1.0,
        seed: int = 2027,
        apply_fn: ApplySatFn,
    ) -> None:
        self.stages = build_default_sat_view_stages(scenarios=scenarios, schedule=schedule, default_prob=p)
        self.scenarios = list(self.stages[0].scenarios)
        self.schedule = str(schedule or "")
        self.seed = int(seed)
        self.apply_fn = apply_fn

    def _generator(self, device: torch.device, epoch: int, batch_idx: int) -> torch.Generator:
        try:
            gen = torch.Generator(device=device)
        except Exception:
            gen = torch.Generator()
        gen.manual_seed(self.seed + int(epoch) * 1009 + int(batch_idx))
        return gen

    def stage_for_epoch(self, epoch: int) -> tuple[int, SatViewStage]:
        cur_index = 0
        for index, stage in enumerate(self.stages):
            if int(epoch) >= int(stage.start_epoch):
                cur_index = index
            else:
                break
        return cur_index, self.stages[cur_index]

    def _select_scenario(self, stage: SatViewStage, gen: torch.Generator, device: torch.device) -> str:
        if len(stage.scenarios) == 1:
            return stage.scenarios[0]
        idx = int(torch.randint(0, len(stage.scenarios), (1,), device=device, generator=gen).item())
        return stage.scenarios[idx]

    def transform(
        self,
        x: torch.Tensor,
        *,
        args: Any,
        epoch: int,
        batch_idx: int,
        batch_meta: Optional[Mapping[str, Any]] = None,
    ) -> SatViewTransform:
        clean_bsz = int(x.size(0))
        physical_sample_id, crop_offset = _pair_meta_from_batch(
            batch_meta, batch_size=clean_bsz, device=x.device
        )
        stage_index, stage = self.stage_for_epoch(epoch)
        gen = self._generator(x.device, epoch, batch_idx)
        p = _clamp_prob(stage.view_prob)
        if p <= 0.0 or float(torch.rand((), device=x.device, generator=gen).item()) > p:
            return SatViewTransform(
                x=x.clone(),
                scenario="clean_duplicate",
                stage_start_epoch=int(stage.start_epoch),
                stage_index=int(stage_index),
                view_prob=p,
                applied=False,
                clean_batch_size=clean_bsz,
                meta={"scenario": "clean_duplicate", "valid": False},
                pair_id=physical_sample_id,
                physical_sample_id=physical_sample_id,
                crop_offset=crop_offset,
            )
        scenario = self._select_scenario(stage, gen, x.device)
        x_sat, raw_meta = self.apply_fn(x, scenario, args, gen=gen, return_meta=True)
        meta, nuisance, nuisance_valid, nuisance_fields = _normalize_nuisance_meta(
            raw_meta,
            scenario=scenario,
            batch_size=clean_bsz,
            device=x.device,
        )
        return SatViewTransform(
            x=x_sat.to(device=x.device, dtype=x.dtype),
            scenario=scenario,
            stage_start_epoch=int(stage.start_epoch),
            stage_index=int(stage_index),
            view_prob=p,
            applied=True,
            clean_batch_size=clean_bsz,
            meta=meta,
            nuisance=nuisance,
            nuisance_valid=nuisance_valid,
            nuisance_fields=nuisance_fields,
            pair_id=physical_sample_id,
            physical_sample_id=physical_sample_id,
            crop_offset=crop_offset,
        )

    def expand(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
        *,
        args: Any,
        epoch: int,
        batch_idx: int,
    ) -> BaselineOriginSatViewBatch:
        view = self.transform(x, args=args, epoch=epoch, batch_idx=batch_idx)
        y_view = y.to(device=x.device)
        x_cat = torch.cat([x, view.x], dim=0)
        y_cat = torch.cat([y_view, y_view], dim=0)
        d_cat = _concat_optional_domain(d_raw, x.device)
        nuisance, nuisance_valid = _expand_nuisance(
            view.nuisance,
            view.nuisance_valid,
            int(x.size(0)),
        )
        return BaselineOriginSatViewBatch(
            x=x_cat,
            y=y_cat,
            d_raw=d_cat,
            scenario=view.scenario,
            clean_batch_size=int(x.size(0)),
            total_batch_size=int(x_cat.size(0)),
            stage_start_epoch=int(view.stage_start_epoch),
            stage_index=int(view.stage_index),
            view_prob=float(view.view_prob),
            applied=bool(view.applied),
            meta=view.meta,
            nuisance=nuisance,
            nuisance_valid=nuisance_valid,
            nuisance_fields=view.nuisance_fields,
        )
