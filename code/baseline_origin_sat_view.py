from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from cvsrffi.phase1_fcr_types import (
    FCR_V2_ETA_FIELDS,
    FCR_V2_ETA_SCALES,
    FCR_V2_ETA_SCHEMA_VERSION,
    FCR_V2_ETA_UNITS,
)


ApplySatFn = Callable[..., tuple]

CRRA_NUISANCE_FIELDS = FCR_V2_ETA_FIELDS
CRRA_NUISANCE_UNITS = FCR_V2_ETA_UNITS

CRRA_NUISANCE_SCALES = dict(zip(FCR_V2_ETA_FIELDS, FCR_V2_ETA_SCALES))


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
    eta_schema_version: str = FCR_V2_ETA_SCHEMA_VERSION
    eta_fields: tuple[str, ...] = FCR_V2_ETA_FIELDS
    eta_units: tuple[str, ...] = FCR_V2_ETA_UNITS
    eta_scales: tuple[float, ...] = FCR_V2_ETA_SCALES
    eta: Optional[torch.Tensor] = None
    eta_valid_mask: Optional[torch.Tensor] = None
    scenario_by_sample: tuple[str, ...] = ()
    applied_mask: Optional[torch.Tensor] = None


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
) -> tuple[
    Optional[dict[str, Any]],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    tuple[str, ...],
    Optional[torch.Tensor],
]:
    if not isinstance(meta, Mapping):
        return {"scenario": str(scenario), "valid": False}, None, None, (), None
    raw = dict(meta)
    raw.setdefault("scenario", str(scenario))
    if "residual_cfo_hz" not in raw and "cfo_hz" in raw:
        raw["residual_cfo_hz"] = raw["cfo_hz"]
    columns = []
    valid_columns = []
    missing_fields = []
    for field in CRRA_NUISANCE_FIELDS:
        column = _batch_meta_column(raw.get(field), int(batch_size), device)
        if column is None:
            missing_fields.append(field)
            column = torch.zeros(int(batch_size), device=device, dtype=torch.float32)
            valid_columns.append(torch.zeros(int(batch_size), dtype=torch.bool, device=device))
        else:
            valid_columns.append(torch.isfinite(column))
        columns.append(column / float(CRRA_NUISANCE_SCALES[field]))
    nuisance = torch.stack(columns, dim=1)
    eta_valid_mask = torch.stack(valid_columns, dim=1)
    finite_valid = torch.isfinite(nuisance)
    eta_valid_mask = eta_valid_mask & finite_valid
    valid = eta_valid_mask.all(dim=1)
    nuisance = torch.nan_to_num(nuisance, nan=0.0, posinf=0.0, neginf=0.0)
    raw["valid"] = bool(valid.any().item())
    raw["missing_fields"] = tuple(missing_fields)
    return raw, nuisance, valid, CRRA_NUISANCE_FIELDS, eta_valid_mask


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

    def _sample_generator(
        self,
        device: torch.device,
        *,
        epoch: int,
        physical_sample_id: str,
        view_type: str,
    ) -> torch.Generator:
        """Return a stateless per-record stream, independent of loader order."""

        try:
            gen = torch.Generator(device=device)
        except Exception:
            gen = torch.Generator()
        payload = (
            f"{self.seed}:{int(epoch)}:{physical_sample_id}:{view_type}"
        ).encode("utf-8")
        sample_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        gen.manual_seed(sample_seed % (2**63 - 1))
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
        if physical_sample_id is not None:
            return self._transform_per_sample(
                x,
                args=args,
                epoch=epoch,
                physical_sample_id=physical_sample_id,
                crop_offset=crop_offset,
                stage_index=stage_index,
                stage=stage,
            )
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
                scenario_by_sample=("clean_duplicate",) * clean_bsz,
                applied_mask=torch.zeros(clean_bsz, dtype=torch.bool, device=x.device),
            )
        scenario = self._select_scenario(stage, gen, x.device)
        x_sat, raw_meta = self.apply_fn(x, scenario, args, gen=gen, return_meta=True)
        meta, nuisance, nuisance_valid, nuisance_fields, eta_valid_mask = _normalize_nuisance_meta(
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
            eta=nuisance,
            eta_valid_mask=eta_valid_mask,
            scenario_by_sample=(scenario,) * clean_bsz,
            applied_mask=torch.ones(clean_bsz, dtype=torch.bool, device=x.device),
        )

    def _transform_per_sample(
        self,
        x: torch.Tensor,
        *,
        args: Any,
        epoch: int,
        physical_sample_id: tuple[str, ...],
        crop_offset: Optional[torch.Tensor],
        stage_index: int,
        stage: SatViewStage,
    ) -> SatViewTransform:
        clean_bsz = int(x.size(0))
        p = _clamp_prob(stage.view_prob)
        rows: list[torch.Tensor] = []
        eta_rows: list[torch.Tensor] = []
        eta_mask_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        scenarios: list[str] = []
        applied_rows: list[bool] = []
        raw_rows: list[dict[str, Any]] = []
        for index, sample_id in enumerate(physical_sample_id):
            gen = self._sample_generator(
                x.device,
                epoch=int(epoch),
                physical_sample_id=str(sample_id),
                view_type="satellite_view",
            )
            apply_row = p > 0.0 and (
                p >= 1.0 or float(torch.rand((), device=x.device, generator=gen).item()) <= p
            )
            if not apply_row:
                rows.append(x[index : index + 1].clone())
                eta_rows.append(x.new_zeros((1, len(CRRA_NUISANCE_FIELDS))))
                eta_mask_rows.append(torch.zeros((1, len(CRRA_NUISANCE_FIELDS)), dtype=torch.bool, device=x.device))
                valid_rows.append(torch.zeros(1, dtype=torch.bool, device=x.device))
                scenarios.append("clean_duplicate")
                applied_rows.append(False)
                raw_rows.append({"scenario": "clean_duplicate", "valid": False})
                continue
            scenario = self._select_scenario(stage, gen, x.device)
            x_sat, raw_meta = self.apply_fn(
                x[index : index + 1], scenario, args, gen=gen, return_meta=True
            )
            normalized, nuisance, nuisance_valid, nuisance_fields, eta_mask = _normalize_nuisance_meta(
                raw_meta,
                scenario=scenario,
                batch_size=1,
                device=x.device,
            )
            if nuisance is None or eta_mask is None or nuisance_valid is None:
                nuisance = x.new_zeros((1, len(CRRA_NUISANCE_FIELDS)))
                eta_mask = torch.zeros_like(nuisance, dtype=torch.bool)
                nuisance_valid = torch.zeros(1, dtype=torch.bool, device=x.device)
            if nuisance_fields and tuple(nuisance_fields) != CRRA_NUISANCE_FIELDS:
                raise ValueError("per-sample CRRA eta field order drift")
            rows.append(x_sat.to(device=x.device, dtype=x.dtype))
            eta_rows.append(nuisance)
            eta_mask_rows.append(eta_mask)
            valid_rows.append(nuisance_valid)
            scenarios.append(scenario)
            applied_rows.append(True)
            raw_rows.append(dict(normalized or {}))
        scenario_summary = scenarios[0] if len(set(scenarios)) == 1 else "mixed_per_sample"
        return SatViewTransform(
            x=torch.cat(rows, dim=0),
            scenario=scenario_summary,
            stage_start_epoch=int(stage.start_epoch),
            stage_index=int(stage_index),
            view_prob=p,
            applied=bool(any(applied_rows)),
            clean_batch_size=clean_bsz,
            meta={
                "scenario": scenario_summary,
                "scenario_by_sample": tuple(scenarios),
                "per_sample": tuple(raw_rows),
                "eta_schema_version": FCR_V2_ETA_SCHEMA_VERSION,
                "eta_fields": CRRA_NUISANCE_FIELDS,
                "eta_units": CRRA_NUISANCE_UNITS,
                "eta_scales": FCR_V2_ETA_SCALES,
            },
            nuisance=torch.cat(eta_rows, dim=0),
            nuisance_valid=torch.cat(valid_rows, dim=0),
            nuisance_fields=CRRA_NUISANCE_FIELDS,
            pair_id=physical_sample_id,
            physical_sample_id=physical_sample_id,
            crop_offset=crop_offset,
            eta=torch.cat(eta_rows, dim=0),
            eta_valid_mask=torch.cat(eta_mask_rows, dim=0),
            scenario_by_sample=tuple(scenarios),
            applied_mask=torch.tensor(applied_rows, dtype=torch.bool, device=x.device),
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
