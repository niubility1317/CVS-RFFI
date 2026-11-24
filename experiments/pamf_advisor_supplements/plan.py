"""Predeclared Phase1 matrix for the advisor's data controls."""
from itertools import combinations

RUN = "20261124-phase1-pamf-advisor-data-m32-r01"
MODEL_SEED = 2026092701
SPLIT_SEED = 392005
SOURCE = (1, 3, 4, 6, 8)
SOURCE_POOL = SOURCE + (11,)
TARGET = (0, 2, 5, 7, 9, 10)
TARGET_MAIN = TARGET + (11,)
DAYS = (0, 1, 2, 3)
ROLES = ("L_s", "U_s", "V")


def matrix():
    rows = []

    def add(setting, family, input_cache, rxs, days, target, roles, methods, endpoint=None):
        for method in methods:
            rows.append(dict(
                row_id=f"{setting}-{method}-s{MODEL_SEED}",
                setting=setting, family=family, input_cache=input_cache,
                source_rxs=list(rxs), source_days=list(days), target_handle=target,
                target_rxs=list(TARGET if target == "m6-natural" else TARGET_MAIN),
                counts=dict(zip(ROLES, roles)), method=method, endpoint=endpoint,
                n_source_rx=len(rxs), model_seed=MODEL_SEED,
            ))

    for budget, labels, val, cache in (
        (630, 450, 180, "label01"),
        (3150, 2520, 630, "label05"),
        (6300, 5040, 1260, "F0"),
    ):
        add(f"budget{budget}", "label_budget", cache, SOURCE, DAYS[1:], "F0",
            (labels, 63000 - budget, val), ("ssl", "pl"))

    for size in range(1, 7):
        # Endpoints of the lexicographically ordered combinations and its
        # midpoint. Every choice is fixed by receiver index alone.
        groups = list(combinations(SOURCE_POOL, size))
        indices = sorted({0, (len(groups) - 1) // 2, len(groups) - 1})
        for group, index in enumerate(indices, 1):
            add(f"receivers{size}-group{group}", "receiver_count", "m6-natural",
                groups[index], DAYS[1:], "m6-natural",
                (1080, 10800, 1080), ("ssl",))

    for day in DAYS:
        source_days = tuple(d for d in DAYS if d != day)
        add(f"leaveD{day}", "leave_date_out", "by_date", SOURCE, source_days,
            "F0", (6300, 56700, 27000), ("ssl", "pl"), day)
    add("forwardD2", "forward_date", "by_date", SOURCE, DAYS[:2],
        "F0", (4200, 37800, 18000), ("ssl", "pl"), 2)

    assert len(rows) == 32
    return rows
