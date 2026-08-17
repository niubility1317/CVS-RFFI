#!/usr/bin/env python3
"""Run the frozen MRIOR-SDA -> ERTB-IDR DA1_REG1 comparison matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
METHOD = "mrior_sda_then_ertb_idr"
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
K_SHOTS = (1, 5, 10, 20)
NEW_COUNTS = (5, 10, 20)
SHARD_COUNT = 8
GROUND_MANIFEST_SHA256 = (
    "15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _contract_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exception_fingerprint(exc: BaseException) -> str:
    message = str(exc).splitlines()[-1] if str(exc).splitlines() else ""
    message = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    message = re.sub(r"\b\d+\b", "<n>", message)
    message = re.sub(r"[/\\][^\s:]+", "<path>", message)
    return hashlib.sha256(f"{type(exc).__name__}:{message}".encode("utf-8")).hexdigest()


def _run_json(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        list(command), cwd=str(cwd), text=True, capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(command)}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command returned no JSON: {' '.join(command)}")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command returned invalid JSON: {proc.stdout}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("command JSON result is not an object")
    return value


def _source_packages(source_plan: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    payload = _read(source_plan)
    if payload.get("schema") != "cvs.phase2.adv3b02_paper_full_ci_plan.v1":
        raise ValueError("source v7 plan schema drift")
    rows = payload.get("packages")
    if not isinstance(rows, list):
        raise ValueError("source v7 package list missing")
    packages: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source v7 package row drift")
        key = (str(row["receiver"]), int(row["seed"]), int(row["new_class_count"]))
        if key in packages:
            raise ValueError("source v7 duplicate package")
        if key[2] in NEW_COUNTS:
            packages[key] = row
    if len(packages) != len(RECEIVERS) * len(SEEDS) * len(NEW_COUNTS):
        raise ValueError("source v7 package coverage drift")
    return packages


def _make_plan(args: argparse.Namespace) -> dict[str, Any]:
    source_plan = Path(args.source_plan).resolve(strict=True)
    packages = _source_packages(source_plan)
    source_sha = _sha256(source_plan)
    cells: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot in K_SHOTS:
                for new_count in NEW_COUNTS:
                    package = packages[(receiver, seed, new_count)]
                    package_root = Path(str(package["predictor_package_root"])).resolve(strict=True)
                    seal_path = Path(str(package["detached_seal"])).resolve(strict=True)
                    build_receipt = Path(str(package["build_receipt"])).resolve(strict=True)
                    old_labels = package.get("old_class_labels")
                    if not isinstance(old_labels, list) or len(old_labels) != 6:
                        raise ValueError("source v7 old-class registry drift")
                    binding = (
                        Path(str(args.mrior_root)).resolve()
                        / "cells"
                        / f"rx_{receiver.replace('-', '_')}__seed_{seed}__new_{new_count}"
                        f"__mrior_sda_then_csil_paper_full__k_{k_shot}"
                        / "mrior_preadapt_bindings.json"
                    )
                    cell_id = (
                        f"rx_{receiver.replace('-', '_')}__seed_{seed}__k_{k_shot}"
                        f"__new_{new_count}__{METHOD}"
                    )
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "receiver": receiver,
                            "seed": seed,
                            "k_shot": k_shot,
                            "new_class_count": new_count,
                            "method": METHOD,
                            "package_id": str(package["package_id"]),
                            "package_root": str(package_root),
                            "seal_path": str(seal_path),
                            "seal_sha256": _sha256(seal_path),
                            "build_receipt": str(build_receipt),
                            "scorer_root": str(package["scorer_root"]),
                            "mrior_bindings": str(binding),
                            "expected_total_capacity": len(old_labels) + new_count,
                            "output_root": str(Path(args.run_root).resolve() / "cells" / cell_id),
                        }
                    )
    cells.sort(key=lambda value: value["cell_id"])
    payload: dict[str, Any] = {
        "schema": "cvs.phase2.adv3b02_mrior_ertb_da1_reg1_ci_plan.v1",
        "method": METHOD,
        "source_plan": str(source_plan),
        "source_plan_sha256": source_sha,
        "mrior_root": str(Path(args.mrior_root).resolve()),
        "ground_component_dir": str(Path(args.ground_component_dir).resolve()),
        "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
        "run_root": str(Path(args.run_root).resolve()),
        "shard_count": SHARD_COUNT,
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "k_shots": list(K_SHOTS),
        "new_class_counts": list(NEW_COUNTS),
        "counts": {"cells": len(cells), "scenario_rows": len(cells) * len(SCENARIOS)},
        "smoke_cell_id": "rx_20_1__seed_713102__k_5__new_5__" + METHOD,
        "cells": cells,
    }
    payload["plan_contract_sha256"] = _contract_sha(payload)
    return payload


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _read(path)
    if plan.get("schema") != "cvs.phase2.adv3b02_mrior_ertb_da1_reg1_ci_plan.v1":
        raise ValueError("ERTB DA1_REG1 plan schema drift")
    contract = {key: value for key, value in plan.items() if key != "plan_contract_sha256"}
    if plan.get("plan_contract_sha256") != _contract_sha(contract):
        raise ValueError("ERTB DA1_REG1 plan contract drift")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != 300:
        raise ValueError("ERTB DA1_REG1 cell count drift")
    if plan.get("counts") != {"cells": 300, "scenario_rows": 900}:
        raise ValueError("ERTB DA1_REG1 counts drift")
    if len({str(cell.get("cell_id")) for cell in cells}) != len(cells):
        raise ValueError("ERTB DA1_REG1 duplicate cell")
    return plan


def _run_cell(plan: Mapping[str, Any], cell: Mapping[str, Any], *, project_root: Path, device: str) -> dict[str, Any]:
    output_root = Path(str(cell["output_root"])).resolve()
    receipt_path = output_root / "cell_receipt.json"
    if receipt_path.is_file():
        receipt = _read(receipt_path)
        if receipt.get("status") == "PASS" and receipt.get("cell_id") == cell["cell_id"]:
            return receipt
        raise RuntimeError("existing ERTB cell receipt is not PASS")
    if output_root.exists():
        raise RuntimeError("partial ERTB cell output exists; refusing overwrite")
    output_root.mkdir(parents=True, exist_ok=False)
    build_receipt = _read(Path(str(cell["build_receipt"])).resolve(strict=True))
    scoring_sha = str(build_receipt.get("scoring_manifest_sha256", ""))
    if len(scoring_sha) != 64:
        raise ValueError("source scoring manifest SHA missing")
    predictor_output = output_root / "predictor"
    predictor_command = [
        sys.executable,
        str(project_root / "paper_reproduction/scripts/run_adv3b02_mrior_ertb_predictor.py"),
        "--package-root", str(cell["package_root"]),
        "--detached-seal", str(cell["seal_path"]),
        "--expected-seal-sha256", str(cell["seal_sha256"]),
        "--mrior-bindings", str(cell["mrior_bindings"]),
        "--ground-component-dir", str(plan["ground_component_dir"]),
        "--ground-manifest-sha256", str(plan["ground_manifest_sha256"]),
        "--old-class-count", "6",
        "--expected-total-capacity", str(cell["expected_total_capacity"]),
        "--k-shot", str(cell["k_shot"]),
        "--seed", str(cell["seed"]),
        "--row-id", str(cell["cell_id"]),
        "--output-dir", str(predictor_output),
        "--device", device,
    ]
    predictor = _run_json(predictor_command, cwd=project_root)
    scoring_output = output_root / "scoring"
    scoring_command = [
        sys.executable,
        str(project_root / "code/scripts/score_cvs_stage2_sealed_prediction.py"),
        "--prediction-artifact", str(predictor["prediction_artifact"]),
        "--expected-prediction-artifact-sha256", str(predictor["prediction_artifact_sha256"]),
        "--expected-prediction-seal-sha256", str(predictor["prediction_seal_sha256"]),
        "--scoring-manifest", str(Path(str(cell["scorer_root"])) / "scoring_manifest.json"),
        "--expected-scoring-manifest-sha256", scoring_sha,
        "--formal-rows", str(scoring_output / "formal_rows.json"),
        "--formal-predictions", str(scoring_output / "formal_predictions.json"),
        "--scoring-receipt", str(scoring_output / "scoring_receipt.json"),
    ]
    scoring = _run_json(scoring_command, cwd=project_root)
    rows = _read(scoring_output / "formal_rows.json")
    if rows.get("schema") != "cvs.phase2.formal_metric_rows.v1" or len(rows.get("rows", [])) != 3:
        raise ValueError("ERTB cell formal-row closure drift")
    if scoring.get("status") != "PASS":
        raise ValueError("ERTB cell scorer did not PASS")
    receipt = {
        "schema": "cvs.phase2.adv3b02_mrior_ertb_da1_reg1_cell_receipt.v1",
        "status": "PASS",
        "cell_id": cell["cell_id"],
        "method": METHOD,
        "four_state_before_registration": "DA1_REG0",
        "four_state_after_registration": "DA1_REG1",
        "receiver": cell["receiver"],
        "seed": int(cell["seed"]),
        "k_shot": int(cell["k_shot"]),
        "new_class_count": int(cell["new_class_count"]),
        "predictor_receipt": str(predictor["predictor_receipt"]),
        "prediction_artifact": str(predictor["prediction_artifact"]),
        "scoring_receipt": str(scoring_output / "scoring_receipt.json"),
        "formal_rows": str(scoring_output / "formal_rows.json"),
        "prediction_artifact_sha256": str(predictor["prediction_artifact_sha256"]),
        "prediction_seal_sha256": str(predictor["prediction_seal_sha256"]),
        "scoring_status": scoring["status"],
        "scenario_row_count": 3,
    }
    _write_new(receipt_path, receipt)
    return receipt


def _finalize(plan: Mapping[str, Any]) -> dict[str, Any]:
    receipts = []
    rows_total = 0
    for cell in plan["cells"]:
        path = Path(str(cell["output_root"])).resolve() / "cell_receipt.json"
        receipt = _read(path)
        if receipt.get("status") != "PASS" or receipt.get("cell_id") != cell["cell_id"]:
            raise ValueError("ERTB final cell receipt drift")
        rows = _read(Path(str(receipt["formal_rows"])).resolve())
        if rows.get("schema") != "cvs.phase2.formal_metric_rows.v1" or len(rows.get("rows", [])) != 3:
            raise ValueError("ERTB final formal-row drift")
        receipts.append(receipt)
        rows_total += 3
    if len(receipts) != 300 or rows_total != 900:
        raise ValueError("ERTB final matrix closure drift")
    result = {
        "schema": "cvs.phase2.adv3b02_mrior_ertb_da1_reg1_final_receipt.v1",
        "status": "PASS",
        "method": METHOD,
        "four_state_before_registration": "DA1_REG0",
        "four_state_after_registration": "DA1_REG1",
        "counts": {"cells": 300, "predictions": 300, "scores": 300, "scenario_rows": 900},
        "plan_contract_sha256": plan["plan_contract_sha256"],
        "cell_receipt_sha256": hashlib.sha256(
            "".join(sorted(_sha256(Path(str(cell["output_root"])).resolve() / "cell_receipt.json") for cell in plan["cells"])).encode("ascii")
        ).hexdigest(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "shard", "finalize"), required=True)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--mrior-root", type=Path)
    parser.add_argument("--ground-component-dir", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    if args.stage == "prepare":
        if args.source_plan is None or args.mrior_root is None or args.ground_component_dir is None:
            raise ValueError("prepare requires source-plan, mrior-root and ground-component-dir")
        if run_root.exists():
            raise ValueError("ERTB run root already exists; refusing overwrite")
        run_root.mkdir(parents=True, exist_ok=False)
        plan = _make_plan(args)
        _write_new(run_root / "plan.json", plan)
        print(json.dumps({"status": "PASS", "plan": str(run_root / "plan.json"), "plan_contract_sha256": plan["plan_contract_sha256"]}, sort_keys=True))
        return
    plan_path = Path(args.plan or (run_root / "plan.json")).resolve(strict=True)
    plan = _load_plan(plan_path)
    if args.stage == "smoke":
        cell = next(cell for cell in plan["cells"] if cell["cell_id"] == plan["smoke_cell_id"])
        receipt = _run_cell(plan, cell, project_root=Path(args.project_root).resolve(), device=args.device)
        print(json.dumps({"status": receipt["status"], "cell_id": receipt["cell_id"]}, sort_keys=True))
        return
    if args.stage == "shard":
        if int(args.shard_count) != SHARD_COUNT or not 0 <= int(args.shard_index) < SHARD_COUNT:
            raise ValueError("ERTB shard contract drift")
        selected = [cell for index, cell in enumerate(plan["cells"]) if index % SHARD_COUNT == int(args.shard_index)]
        failures: list[tuple[str, str]] = []
        seen: set[str] = set()
        for cell in selected:
            try:
                _run_cell(plan, cell, project_root=Path(args.project_root).resolve(), device=args.device)
            except Exception as exc:  # preserve the cell failure and stop only on repeated deterministic faults
                fingerprint = _exception_fingerprint(exc)
                failures.append((str(cell["cell_id"]), fingerprint))
                if fingerprint in seen:
                    _write_new(
                        Path(str(plan["run_root"])) / f"health_stop_shard_{args.shard_index}.json",
                        {"status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE", "fingerprint": fingerprint, "cell_id": cell["cell_id"]},
                    )
                    raise
                seen.add(fingerprint)
        if failures:
            raise RuntimeError(f"ERTB shard had failures: {failures}")
        print(json.dumps({"status": "PASS", "shard_index": int(args.shard_index), "cells": len(selected)}, sort_keys=True))
        return
    result = _finalize(plan)
    _write_new(run_root / "final_receipt.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
