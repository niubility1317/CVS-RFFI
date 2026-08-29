"""Frozen seven-receiver Phase2 confirmation matrix for ADV3B02 FastTrust."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


TARGET_RECEIVERS = ("1-1", "14-7", "2-1", "20-1", "7-14", "7-7", "8-8")
FORMAL_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
CAPSULE_ID = "536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2"
SPLIT_ID = "260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25"
FROZEN_CHECKPOINT_PATH = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1/"
    "S713104_ADV3B02_FASTTRUST_EFF/final_ssdg.pth"
)


def build_receiver_matrix(
    *,
    run_root: str,
    checkpoint_path: str,
    seed: int = 713104,
) -> list[dict[str, Any]]:
    """Return the immutable 7x3 prediction-first confirmation matrix."""

    root = PurePosixPath(str(run_root))
    checkpoint = str(checkpoint_path).strip()
    if not str(root).strip() or not checkpoint:
        raise ValueError("run_root and checkpoint_path must be nonempty")
    if checkpoint != FROZEN_CHECKPOINT_PATH:
        raise ValueError("checkpoint_path must bind the frozen FastTrust checkpoint")
    if int(seed) != 713104:
        raise ValueError("the frozen confirmation seed must remain 713104")

    rows: list[dict[str, Any]] = []
    for gpu, receiver in enumerate(TARGET_RECEIVERS):
        receiver_root = root / "receivers" / f"rx{receiver}"
        package_root = receiver_root / "package" / "predictor"
        for scenario in FORMAL_SCENARIOS:
            row_id = f"RX{receiver}_{scenario}_K20_S713104"
            row_root = receiver_root / "rows" / scenario
            rows.append(
                {
                    "row_id": row_id,
                    "receiver": receiver,
                    "scenario": scenario,
                    "gpu": gpu,
                    "seed": 713104,
                    "k_shot": 20,
                    "expected_query_rows": 1352,
                    "state": "DA1_REG1",
                    "protocol_schema": "p2_min_v1",
                    "phase2_data_status": "VALIDATED_ONCE",
                    "capsule_id": CAPSULE_ID,
                    "split_id": SPLIT_ID,
                    "checkpoint_path": checkpoint,
                    "support_input_path": str(package_root / f"support_{scenario}.npz"),
                    "query_input_path": str(package_root / f"query_{scenario}.npz"),
                    "truth_path": str(
                        receiver_root
                        / "package"
                        / "scorer"
                        / f"truth_{scenario}_rx{receiver}_k20.json"
                    ),
                    "support_path": str(
                        package_root / f"support_{scenario}_rx{receiver}_k20.npz"
                    ),
                    "query_path": str(
                        package_root / f"query_{scenario}_rx{receiver}_k20.npz"
                    ),
                    "support_audit_path": str(
                        package_root / f"support_{scenario}_rx{receiver}_k20.audit.json"
                    ),
                    "prototype_path": str(
                        row_root
                        / "input"
                        / f"prototypes_{scenario}_rx{receiver}_k20.npz"
                    ),
                    "prediction_output_dir": str(row_root / "prediction"),
                    "score_path": str(row_root / "score.json"),
                }
            )
    return rows


__all__ = [
    "CAPSULE_ID",
    "FORMAL_SCENARIOS",
    "FROZEN_CHECKPOINT_PATH",
    "SPLIT_ID",
    "TARGET_RECEIVERS",
    "build_receiver_matrix",
]
