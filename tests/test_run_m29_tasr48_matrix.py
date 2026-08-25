from __future__ import annotations

import concurrent.futures

import pytest

from scripts import run_m29_tasr48_matrix as runner


def test_execute_fail_fast_does_not_dispatch_after_first_failure() -> None:
    visited: list[int] = []

    def worker(value: int) -> int:
        visited.append(value)
        if value == 0:
            raise RuntimeError("first row failed")
        return value

    with pytest.raises(RuntimeError, match="first row failed"):
        runner._execute_fail_fast(
            list(range(30)),
            worker=worker,
            max_workers=1,
            executor_factory=concurrent.futures.ThreadPoolExecutor,
        )
    assert visited == [0]
