"""Finite, source-only convergence primitives for PairBiCAD-CV2.

The objects in this module consume scalar observations supplied by a caller.
They deliberately have no data-loader, checkpoint, filesystem, or training
entrypoint dependency.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class CoverageLedger:
    """Track cumulative U coverage and minimum L group exposure.

    ``u_coverage`` may exceed one after a no-replacement sampler starts a new
    cycle. ``u_unique_coverage`` remains bounded by one. ``l_coverage`` is the
    minimum exposure count over the declared TX x receiver x day groups.
    """

    def __init__(
        self,
        *,
        u_sample_ids: Iterable[Hashable],
        l_groups: Iterable[Hashable],
    ) -> None:
        u_values = tuple(u_sample_ids)
        l_values = tuple(l_groups)
        if not u_values:
            raise ValueError("u_sample_ids must be non-empty")
        if not l_values:
            raise ValueError("l_groups must be non-empty")
        if len(set(u_values)) != len(u_values):
            raise ValueError("u_sample_ids must be unique")
        if len(set(l_values)) != len(l_values):
            raise ValueError("l_groups must be unique")
        self._u_ids = frozenset(u_values)
        self._l_groups = tuple(l_values)
        self._u_visits = 0
        self._u_seen: set[Hashable] = set()
        self._l_exposures: Counter[Hashable] = Counter()

    def record_u(self, sample_ids: Iterable[Hashable]) -> None:
        values = tuple(sample_ids)
        unknown = set(values) - self._u_ids
        if unknown:
            raise ValueError("U observation contains an undeclared sample ID")
        self._u_visits += len(values)
        self._u_seen.update(values)

    def record_l(self, groups: Iterable[Hashable]) -> None:
        values = tuple(groups)
        unknown = set(values) - set(self._l_groups)
        if unknown:
            raise ValueError("L observation contains an undeclared group")
        self._l_exposures.update(values)

    @property
    def u_coverage(self) -> float:
        return self._u_visits / len(self._u_ids)

    @property
    def u_unique_coverage(self) -> float:
        return len(self._u_seen) / len(self._u_ids)

    @property
    def l_coverage(self) -> float:
        return float(min(self._l_exposures[group] for group in self._l_groups))

    @property
    def l_group_exposures(self) -> Mapping[Hashable, int]:
        return MappingProxyType(
            {group: self._l_exposures[group] for group in self._l_groups}
        )


@dataclass(frozen=True)
class DGObservation:
    """One V_cal convergence observation on the 0--1 metric scale."""

    updates: int
    coverage_u: float
    s_dg: float
    learning_rate: float
    d_logit: float
    d_theta: float
    margin_q10: float
    elapsed_hours: float
    gradient_ratios: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or not isinstance(self.updates, int):
            raise TypeError("updates must be a positive integer")
        if self.updates <= 0:
            raise ValueError("updates must be a positive integer")
        nonnegative = (
            "coverage_u",
            "learning_rate",
            "d_logit",
            "d_theta",
            "elapsed_hours",
        )
        for name in (
            "coverage_u",
            "s_dg",
            "learning_rate",
            "d_logit",
            "d_theta",
            "margin_q10",
            "elapsed_hours",
        ):
            result = _finite_float(getattr(self, name), name=name)
            if name in nonnegative and result < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, result)
        if not 0.0 <= self.s_dg <= 1.0:
            raise ValueError("s_dg must be on the 0--1 scale")
        ratios: dict[str, float] = {}
        for name, value in self.gradient_ratios.items():
            ratio = _finite_float(value, name=f"gradient_ratios[{name!r}]")
            if ratio < 0.0:
                raise ValueError("gradient ratios must be non-negative")
            ratios[str(name)] = ratio
        object.__setattr__(self, "gradient_ratios", MappingProxyType(ratios))


@dataclass(frozen=True)
class ConvergenceDecision:
    status: str
    should_stop: bool
    scientifically_converged: bool
    activation_age: float
    plateau_slope: float | None
    lr_reduction_count: int
    reason: str


class ConvergenceController:
    """Apply the preregistered scientific and technical stop conditions."""

    def __init__(
        self,
        *,
        last_mechanism_activation_coverage: float,
        gradient_ratio_targets: Mapping[str, tuple[float, float]] | None = None,
        min_activation_age: float = 3.0,
        window_size: int = 6,
        slope_limit: float = 0.0015,
        significant_best_gain: float = 0.0030,
        required_lr_reductions: int = 2,
        safety_coverage: float = 12.0,
        safety_hours: float = 24.0,
    ) -> None:
        self.last_mechanism_activation_coverage = _finite_float(
            last_mechanism_activation_coverage,
            name="last_mechanism_activation_coverage",
        )
        self.min_activation_age = _finite_float(
            min_activation_age, name="min_activation_age"
        )
        self.slope_limit = _finite_float(slope_limit, name="slope_limit")
        self.significant_best_gain = _finite_float(
            significant_best_gain, name="significant_best_gain"
        )
        self.safety_coverage = _finite_float(
            safety_coverage, name="safety_coverage"
        )
        self.safety_hours = _finite_float(safety_hours, name="safety_hours")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
            raise ValueError("window_size must be an integer of at least two")
        if (
            isinstance(required_lr_reductions, bool)
            or not isinstance(required_lr_reductions, int)
            or required_lr_reductions < 0
        ):
            raise ValueError("required_lr_reductions must be a non-negative integer")
        self.window_size = window_size
        self.required_lr_reductions = required_lr_reductions
        targets: dict[str, tuple[float, float]] = {}
        for name, bounds in (gradient_ratio_targets or {}).items():
            if len(bounds) != 2:
                raise ValueError("each gradient ratio target needs two bounds")
            low = _finite_float(bounds[0], name=f"{name}.low")
            high = _finite_float(bounds[1], name=f"{name}.high")
            if low < 0.0 or high < low:
                raise ValueError("gradient ratio target bounds are invalid")
            targets[str(name)] = (low, high)
        self.gradient_ratio_targets = MappingProxyType(targets)
        self._history: list[DGObservation] = []
        self._significant_best: list[bool] = []
        self._lr_reduction_count = 0

    @staticmethod
    def _slope(observations: Sequence[DGObservation]) -> float:
        count = len(observations)
        x_mean = (count - 1) / 2.0
        y_mean = sum(item.s_dg for item in observations) / count
        numerator = sum(
            (index - x_mean) * (item.s_dg - y_mean)
            for index, item in enumerate(observations)
        )
        denominator = sum((index - x_mean) ** 2 for index in range(count))
        return numerator / denominator

    def _ratios_are_stable(self) -> bool:
        if not self.gradient_ratio_targets:
            return True
        if len(self._history) < 4:
            return False
        for observation in self._history[-4:]:
            for name, (low, high) in self.gradient_ratio_targets.items():
                value = observation.gradient_ratios.get(name)
                if value is None or not low <= value <= high:
                    return False
        return True

    def observe(self, observation: DGObservation) -> ConvergenceDecision:
        if not isinstance(observation, DGObservation):
            raise TypeError("observation must be a DGObservation")
        if self._history:
            previous = self._history[-1]
            if observation.updates <= previous.updates:
                raise ValueError("observation updates must increase")
            if observation.coverage_u < previous.coverage_u:
                raise ValueError("coverage_u must not decrease")
            if observation.elapsed_hours < previous.elapsed_hours:
                raise ValueError("elapsed_hours must not decrease")
            if observation.learning_rate < previous.learning_rate:
                self._lr_reduction_count += 1
            previous_best = max(item.s_dg for item in self._history)
            significant_best = (
                observation.s_dg > previous_best + self.significant_best_gain
            )
        else:
            significant_best = False

        self._history.append(observation)
        self._significant_best.append(significant_best)
        activation_age = (
            observation.coverage_u - self.last_mechanism_activation_coverage
        )
        window = self._history[-self.window_size :]
        slope = self._slope(window) if len(window) == self.window_size else None
        plateau = slope is not None and abs(slope) < self.slope_limit
        no_large_new_best = (
            len(window) == self.window_size
            and not any(self._significant_best[-self.window_size :])
        )
        margin_not_down = (
            len(window) == self.window_size
            and window[-1].margin_q10 >= window[0].margin_q10
        )
        scientific = all(
            (
                activation_age >= self.min_activation_age,
                plateau,
                no_large_new_best,
                self._lr_reduction_count >= self.required_lr_reductions,
                observation.d_logit < 0.01,
                observation.d_theta < 1.0e-3,
                self._ratios_are_stable(),
                margin_not_down,
            )
        )
        if scientific:
            return ConvergenceDecision(
                status="SCIENTIFICALLY_CONVERGED",
                should_stop=True,
                scientifically_converged=True,
                activation_age=activation_age,
                plateau_slope=slope,
                lr_reduction_count=self._lr_reduction_count,
                reason="all scientific convergence conditions satisfied",
            )
        if (
            observation.coverage_u >= self.safety_coverage
            or observation.elapsed_hours >= self.safety_hours
        ):
            return ConvergenceDecision(
                status="NOT_CONVERGED_SAFETY_STOP",
                should_stop=True,
                scientifically_converged=False,
                activation_age=activation_age,
                plateau_slope=slope,
                lr_reduction_count=self._lr_reduction_count,
                reason="technical coverage or wall-clock limit reached",
            )
        return ConvergenceDecision(
            status="CONTINUE",
            should_stop=False,
            scientifically_converged=False,
            activation_age=activation_age,
            plateau_slope=slope,
            lr_reduction_count=self._lr_reduction_count,
            reason="scientific convergence conditions are not yet satisfied",
        )
