from typing import Iterable

from baseline_origin_sat_view import (
    ApplySatFn,
    BaselineOriginSatViewAugment,
    BaselineOriginSatViewBatch as ConcatSatBatch,
)


class ConcatSatChannelAugment(BaselineOriginSatViewAugment):
    """Baseline-style clean+satellite supervised view expansion."""

    def __init__(
        self,
        *,
        scenarios: Iterable[str],
        p: float,
        seed: int,
        apply_fn: ApplySatFn,
        schedule: str = "",
    ) -> None:
        super().__init__(
            scenarios=scenarios,
            schedule=schedule,
            p=p,
            seed=seed,
            apply_fn=apply_fn,
        )
