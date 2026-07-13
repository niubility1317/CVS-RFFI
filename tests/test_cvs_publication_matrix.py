from __future__ import annotations

import csv
import json
from pathlib import Path

from paper_reproduction.scripts.run_cvs_publication_matrix import (
    DEFAULT_K,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    PHASE_METHODS,
    _artifact_status,
    build_rows,
)


def test_full_stage2_matrices_cover_methods_receivers_k_and_seeds(tmp_path: Path) -> None:
    for phase in ("stage2b", "stage2c"):
        rows = build_rows(
            phase=phase,
            methods=PHASE_METHODS[phase],
            receivers=DEFAULT_RECEIVERS,
            k_grid=DEFAULT_K,
            seeds=DEFAULT_SEEDS,
            output_root=tmp_path / phase / "runs",
            log_root=tmp_path / phase / "logs",
        )
        assert len(rows) == 3 * 5 * 5 * 5
        assert len({row.experiment_id for row in rows}) == len(rows)
        assert {row.receiver for row in rows} == set(DEFAULT_RECEIVERS)
        assert {row.k_shot for row in rows} == set(DEFAULT_K)
        assert {row.seed for row in rows} == set(DEFAULT_SEEDS)


def test_artifact_contract_requires_satellite_scores_details_and_loss_trace(tmp_path: Path) -> None:
    row = build_rows(
        phase="stage2b",
        methods=("protonet_cda",),
        receivers=("20-1",),
        k_grid=(5,),
        seeds=(713101,),
        output_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
    )[0]
    run_dir = Path(row.run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(
        json.dumps({"support_query_overlap": False, "all_tests_satellite_augmented": True}),
        encoding="utf-8",
    )
    (run_dir / "detailed_metrics.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "loss_trace.json").write_text('[{"loss":1.0}]\n', encoding="utf-8")
    with (run_dir / "score_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        for index in range(360):
            writer.writerow({"sample_id": index})
    with (run_dir / "detailed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_type"])
        writer.writeheader()
        for group_type in (
            "per_receiver",
            "per_transmitter",
            "per_receiver_transmitter",
            "per_receiver_transmitter_day",
        ):
            writer.writerow({"group_type": group_type})
    with (run_dir / "loss_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loss"])
        writer.writeheader()
        writer.writerow({"loss": 1.0})

    assert _artifact_status(row)["complete"] is True
    (run_dir / "loss_trace.csv").write_text("loss\nnan\n", encoding="utf-8")
    status = _artifact_status(row)
    assert status["complete"] is False
    assert "nonfinite_loss_trace" in status["errors"]
