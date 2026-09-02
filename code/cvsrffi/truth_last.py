from __future__ import annotations

import hashlib
import hmac
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCENARIOS = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


def stable_sample_id(physical_id: str, *, split_binding: str) -> str:
    """Return an opaque ID stable across independent predictor/scorer processes."""

    binding = str(split_binding).strip()
    physical = str(physical_id).strip()
    if not binding or not physical:
        raise ValueError("split_binding and physical_id must be non-empty")
    key = hashlib.sha256(("cvs.phase1.truth-last|" + binding).encode("utf-8")).digest()
    return "sample:" + hmac.new(key, physical.encode("utf-8"), hashlib.sha256).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite truth-last artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_truth_sidecar(
    identities: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    split_binding: str,
) -> dict[str, Any]:
    records = []
    seen: set[str] = set()
    for identity in identities:
        sample_id = stable_sample_id(
            str(identity.get("physical_id", "")), split_binding=split_binding
        )
        if sample_id in seen:
            raise ValueError("truth sidecar contains duplicate sample IDs")
        label = identity.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError("truth sidecar label must be a non-negative integer")
        seen.add(sample_id)
        records.append({"sample_id": sample_id, "label": int(label)})
    if not records:
        raise ValueError("truth sidecar cannot be empty")
    payload = {
        "schema": "cvs.phase1.truth_sidecar.v1",
        "split_binding": str(split_binding),
        "record_count": len(records),
        "records": records,
    }
    _write_new_json(Path(output_path), payload)
    return payload


def score_predictions(
    prediction_path: str | Path,
    truth_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    truth_records = truth.get("records", [])
    truth_by_id = {str(row["sample_id"]): int(row["label"]) for row in truth_records}
    if len(truth_by_id) != len(truth_records) or not truth_by_id:
        raise ValueError("truth sidecar contains duplicate or empty coverage")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction.get("records", []):
        grouped[str(row.get("scenario", ""))].append(row)
    if set(grouped) != set(SCENARIOS):
        raise ValueError("prediction scenario coverage does not match the registered four scenarios")

    metrics: dict[str, Any] = {}
    for scenario in SCENARIOS:
        rows = grouped[scenario]
        ids = [str(row.get("sample_id", "")) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(truth_by_id):
            raise ValueError(f"prediction coverage mismatch for scenario {scenario}")
        correct = sum(
            int(row.get("predicted_class", -1)) == truth_by_id[str(row["sample_id"])]
            for row in rows
        )
        per_class_total: dict[int, int] = defaultdict(int)
        per_class_correct: dict[int, int] = defaultdict(int)
        for row in rows:
            label = truth_by_id[str(row["sample_id"])]
            per_class_total[label] += 1
            per_class_correct[label] += int(int(row.get("predicted_class", -1)) == label)
        metrics[scenario] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "per_class_accuracy": {
                str(label): per_class_correct[label] / total
                for label, total in sorted(per_class_total.items())
            },
        }
    result = {
        "schema": "cvs.phase1.truth_last_score.v1",
        "prediction_path": str(prediction_path),
        "truth_path": str(truth_path),
        "record_count": sum(item["total"] for item in metrics.values()),
        "metrics": metrics,
    }
    if output_path is not None:
        _write_new_json(Path(output_path), result)
    return result
