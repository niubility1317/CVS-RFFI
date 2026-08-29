#!/usr/bin/env python3
"""Prepare and execute the historical D92 E0 BiSAGE Target125 lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_bisage_runner import (  # noqa: E402
    SCENARIOS,
    adapt_stage_a,
    adapt_stage_b_and_predict,
    frozen_checkpoint,
    joint_stage_a_gate,
    score_truth_last,
)
from cvsrffi.stage2_bisage_target125 import (  # noqa: E402
    build_bisage_target125_manifest,
    validate_bisage_target125_manifest,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _new_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output root exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(args: argparse.Namespace) -> Mapping[str, Any]:
    config = _load_json(args.config)
    source = _load_json(args.source_manifest)
    expected = str(config["historical_source"]["matrix_manifest_sha256"])
    actual = _sha256(args.source_manifest)
    if actual != expected:
        raise ValueError("historical D92 E0 full Target125 manifest SHA drift")
    manifest = build_bisage_target125_manifest(config, source, str(args.output_root))
    _write_json_new(args.output, manifest)
    return {"status": "PREPARED", **validate_bisage_target125_manifest(manifest, config)}


def _pilot_job(manifest: Mapping[str, Any], pilot_key: str) -> Mapping[str, Any]:
    rows = [job for job in manifest["jobs"] if job["outer_key"] == pilot_key]
    if len(rows) != 1:
        raise ValueError("pilot outer key coverage drift")
    return rows[0]


def _pilot_auto(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = _new_root(args.output_root)
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, args.pilot_outer_key)
    model = frozen_checkpoint(args.checkpoint, args.device)
    stage_a_receipts: dict[str, Any] = {}
    for scenario in SCENARIOS:
        root = destination / scenario
        stage_a_receipts[scenario] = dict(adapt_stage_a(
            job, scenario, model, root / "stage_a", args.device, steps=args.stage_a_steps
        ))
    global_pass = joint_stage_a_gate(stage_a_receipts)
    _write_json_new(destination / "stage_a_group_result.json", {
        "schema": "cvs.phase2.bisage_d92.stage_a_group.v1",
        "outer_key": job["outer_key"],
        "stage_a_all_scenarios_passed": global_pass,
        "scenarios": list(SCENARIOS),
    })
    prediction_receipts: dict[str, Any] = {}
    for scenario in SCENARIOS:
        root = destination / scenario
        prediction_receipts[scenario] = dict(adapt_stage_b_and_predict(
            job,
            scenario,
            model,
            root / "stage_a",
            root / "prediction",
            args.device,
            steps=args.stage_b_steps,
            enable_stage_b=global_pass,
        ))
    status = (
        "STAGE_A_GATE_PASSED_STAGE_B_AUTO_CONTINUED"
        if global_pass
        else "STOPPED_SCIENTIFIC_GATE_STAGE_B_NOT_RUN"
    )
    result = {
        "schema": "cvs.phase2.bisage_d92.pilot_auto.v1",
        "status": status,
        "pilot_outer_key": job["outer_key"],
        "stage_a_all_scenarios_passed": global_pass,
        "stage_a": stage_a_receipts,
        "predictions": prediction_receipts,
        "truth_opened": False,
        "full_target125_authorized": global_pass,
    }
    _write_json_new(destination / "pilot_auto_result.json", result)
    return result


def _score_units(
    manifest: Mapping[str, Any], prediction_root: Path, output_root: Path,
    jobs: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    destination = _new_root(output_root)
    rows = []
    for job in jobs:
        for scenario in SCENARIOS:
            source = prediction_root / job["outer_key"] / scenario / "prediction"
            if not source.exists() and len(jobs) == 1:
                source = prediction_root / scenario / "prediction"
            output = destination / job["outer_key"] / scenario / "score.json"
            result = score_truth_last(
                source / "predictions.npz",
                source / "prediction_receipt.json",
                Path(job["truth_sidecar"]),
                output,
            )
            rows.append(dict(result))
    summary = {
        "schema": "cvs.phase2.bisage_d92.score_collection.v1",
        "status": "ANALYZED",
        "scene_unit_count": len(rows),
        "rows": rows,
        "truth_join_after_prediction_only": True,
    }
    _write_json_new(destination / "score_collection.json", summary)
    return summary


def _score_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(args.manifest)
    job = _pilot_job(manifest, args.pilot_outer_key)
    return _score_units(manifest, args.prediction_root, args.output_root, [job])


def _load_pilot_gate(path: Path) -> None:
    payload = _load_json(path)
    if payload.get("status") != "STAGE_A_GATE_PASSED_STAGE_B_AUTO_CONTINUED":
        raise RuntimeError("full Target125 blocked by Stage A scientific gate")
    if payload.get("full_target125_authorized") is not True:
        raise RuntimeError("full Target125 continuation marker missing")


def _run_shard(args: argparse.Namespace) -> Mapping[str, Any]:
    _load_pilot_gate(args.pilot_result)
    manifest = _load_json(args.manifest)
    jobs = [job for job in manifest["jobs"] if int(job["planned_shard_index"]) == args.shard_index]
    destination = _new_root(args.output_root)
    model = frozen_checkpoint(args.checkpoint, args.device)
    completed = []
    for job in jobs:
        stage_a_by_scene: dict[str, Mapping[str, Any]] = {}
        for scenario in SCENARIOS:
            unit = destination / job["outer_key"] / scenario
            stage_a_by_scene[scenario] = adapt_stage_a(
                job, scenario, model, unit / "stage_a", args.device, steps=args.stage_a_steps
            )
        enable_stage_b = joint_stage_a_gate(stage_a_by_scene)
        _write_json_new(destination / job["outer_key"] / "stage_a_group_result.json", {
            "schema": "cvs.phase2.bisage_d92.stage_a_group.v1",
            "outer_key": job["outer_key"],
            "stage_a_all_scenarios_passed": enable_stage_b,
            "scenarios": list(SCENARIOS),
        })
        for scenario in SCENARIOS:
            unit = destination / job["outer_key"] / scenario
            prediction = adapt_stage_b_and_predict(
                job, scenario, model, unit / "stage_a", unit / "prediction",
                args.device, steps=args.stage_b_steps, enable_stage_b=enable_stage_b,
            )
            completed.append({
                "outer_key": job["outer_key"],
                "scenario": scenario,
                "stage_a_gate_passed": enable_stage_b,
                "stage_b_ran": prediction["stage_b_ran"],
                "selected_mode": prediction["selected_mode"],
            })
            _write_json_new(unit / "unit_complete.json", completed[-1])
    result = {
        "schema": "cvs.phase2.bisage_d92.shard.v1",
        "status": "ARTIFACTS_COMPLETE",
        "shard_index": args.shard_index,
        "scene_unit_count": len(completed),
        "units": completed,
        "truth_opened": False,
    }
    _write_json_new(destination / "shard_result.json", result)
    return result


def _score_shard(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(args.manifest)
    jobs = [job for job in manifest["jobs"] if int(job["planned_shard_index"]) == args.shard_index]
    return _score_units(manifest, args.prediction_root, args.output_root, jobs)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    pilot = commands.add_parser("pilot-auto")
    pilot.add_argument("--manifest", type=Path, required=True)
    pilot.add_argument("--pilot-outer-key", default="rx_3_19__seed_713102__k_10__new_5")
    pilot.add_argument("--checkpoint", required=True)
    pilot.add_argument("--output-root", type=Path, required=True)
    pilot.add_argument("--device", required=True)
    pilot.add_argument("--stage-a-steps", type=int, default=3000)
    pilot.add_argument("--stage-b-steps", type=int, default=2000)
    score_pilot = commands.add_parser("score-pilot")
    score_pilot.add_argument("--manifest", type=Path, required=True)
    score_pilot.add_argument("--pilot-outer-key", default="rx_3_19__seed_713102__k_10__new_5")
    score_pilot.add_argument("--prediction-root", type=Path, required=True)
    score_pilot.add_argument("--output-root", type=Path, required=True)
    for name in ("run-shard", "score-shard"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--shard-index", type=int, choices=range(8), required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "run-shard":
            command.add_argument("--pilot-result", type=Path, required=True)
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--device", required=True)
            command.add_argument("--stage-a-steps", type=int, default=3000)
            command.add_argument("--stage-b-steps", type=int, default=2000)
        else:
            command.add_argument("--prediction-root", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    handlers = {
        "prepare": _prepare,
        "pilot-auto": _pilot_auto,
        "score-pilot": _score_pilot,
        "run-shard": _run_shard,
        "score-shard": _score_shard,
    }
    result = handlers[args.command](args)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
