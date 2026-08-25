from __future__ import annotations

import pytest

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_m29_d92 import IDENTITY_ONLY
from cvsrffi.stage2_m29_row_executor import execute_m29_row


def test_row_executor_rejects_non_p2_min_v1_before_output(tmp_path) -> None:
    cache = {
        "manifest": {
            "receiver": "3-19",
            "method_seed": 7282101,
            "protocol_schema": "legacy",
            "phase2_data_status": "VALIDATED_ONCE",
        },
        "old_classes": tuple(f"old{i}" for i in range(6)),
        "new_classes": tuple(f"new{i}" for i in range(5)),
        "scenario_payloads": {name: {} for name in FORMAL_LEO_WEAK_SCENARIOS},
    }
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="protocol identity drift"):
        execute_m29_row(
            arm=IDENTITY_ONLY,
            row_id="bad-schema",
            receiver="3-19",
            base_cache=cache,
            output_root=output,
            seed=7282101,
            bundle=None,
            base_cache_bytes=0,
        )
    assert not output.exists()
