from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi.stage2_ablation_scoring_sidecars import (
    LEGACY_PREDICTOR_TRUTH_SIDECAR_SCHEMA,
    Stage2AblationScoringSidecarError,
    publish_stage2a_scoring_sidecar,
)
from cvsrffi.stage2_metric_scorer import (
    FORMAL_LEO_WEAK_SCENARIOS,
    TRUTH_SIDECAR_SCHEMA,
    load_verified_scoring_sidecar,
    sha256_file,
)


def _source_truth(schema: str) -> dict[str, object]:
    rows = []
    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows.append(
            {
                "scenario": scenario,
                "query_token": f"qid_{index + 1:032x}",
                "true_class_index": 0,
                "true_class_handle": f"cls_{1:032x}",
                "transmitter_label": "14-10",
                "evaluation_role": "target_old",
                "receiver_label": "20-1",
                "day_label": f"day-{index}",
                "signal_label": f"signal-{index}",
                "physical_sample_id": f"physical-{index}",
            }
        )
    return {
        "schema": schema,
        "stage": "stage2b",
        "receiver": "20-1",
        "seed": 7283101,
        "rows": rows,
    }


def _write_source(path: Path, schema: str) -> dict[str, object]:
    payload = _source_truth(schema)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


@pytest.mark.parametrize(
    "source_schema",
    [LEGACY_PREDICTOR_TRUTH_SIDECAR_SCHEMA, TRUTH_SIDECAR_SCHEMA],
)
def test_publish_stage2a_accepts_supported_source_truth_schemas(
    tmp_path: Path,
    source_schema: str,
) -> None:
    source = tmp_path / "source_truth.json"
    original = _write_source(source, source_schema)
    receipt = publish_stage2a_scoring_sidecar(
        source_stage2b_truth_path=source,
        expected_source_truth_sha256=sha256_file(source),
        predictor_package_root_sha256="a" * 64,
        predictor_package_seal_sha256="b" * 64,
        output_root=tmp_path / "published",
    )

    published = json.loads(
        Path(receipt["truth_sidecar_path"]).read_text(encoding="utf-8")
    )
    assert published["schema"] == TRUTH_SIDECAR_SCHEMA
    assert published["stage"] == "stage2a"
    assert published["rows"] == original["rows"]
    assert receipt["source_truth_schema"] == source_schema
    assert receipt["published_truth_schema"] == TRUTH_SIDECAR_SCHEMA
    assert receipt["truth_rows_reused_without_data_revalidation"] is True
    verified_truth, _, _ = load_verified_scoring_sidecar(
        receipt["scoring_manifest_path"],
        expected_scoring_manifest_sha256=receipt[
            "scoring_manifest_sha256"
        ],
    )
    assert verified_truth == published


def test_publish_stage2a_rejects_unknown_source_truth_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_truth.json"
    _write_source(source, "cvs.phase2.query_truth_sidecar.v1")

    with pytest.raises(
        Stage2AblationScoringSidecarError,
        match="source truth is not a Stage2-B scorer sidecar",
    ):
        publish_stage2a_scoring_sidecar(
            source_stage2b_truth_path=source,
            expected_source_truth_sha256=sha256_file(source),
            predictor_package_root_sha256="a" * 64,
            predictor_package_seal_sha256="b" * 64,
            output_root=tmp_path / "published",
        )


def test_publish_stage2a_rejects_extra_source_truth_key_before_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_truth.json"
    payload = _source_truth(LEGACY_PREDICTOR_TRUTH_SIDECAR_SCHEMA)
    payload["unexpected"] = "must-not-propagate"
    source.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "published"

    with pytest.raises(
        Stage2AblationScoringSidecarError,
        match="source truth is not a Stage2-B scorer sidecar",
    ):
        publish_stage2a_scoring_sidecar(
            source_stage2b_truth_path=source,
            expected_source_truth_sha256=sha256_file(source),
            predictor_package_root_sha256="a" * 64,
            predictor_package_seal_sha256="b" * 64,
            output_root=output,
        )
    assert not output.exists()
