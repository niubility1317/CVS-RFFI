from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Optional

import torch


ApplySatFn = Callable[..., tuple]


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
    ) -> SatViewTransform:
        clean_bsz = int(x.size(0))
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
            )
        scenario = self._select_scenario(stage, gen, x.device)
        x_sat, _ = self.apply_fn(x, scenario, args, gen=gen, return_meta=False)
        return SatViewTransform(
            x=x_sat.to(device=x.device, dtype=x.dtype),
            scenario=scenario,
            stage_start_epoch=int(stage.start_epoch),
            stage_index=int(stage_index),
            view_prob=p,
            applied=True,
            clean_batch_size=clean_bsz,
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
        )
