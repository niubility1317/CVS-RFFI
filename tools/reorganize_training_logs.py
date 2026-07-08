"""Move training logs into CVS and reproduction/comparison roots.

The tool preserves files by moving whole historical log directories and writing
an explicit manifest. It avoids overwriting existing destinations.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPRO_TOKENS = {
    "baseline_paper",
    "comparison",
    "drift",
    "feature_separation",
    "fedbase",
    "fjmp",
    "paper",
    "paper_repro",
    "paper_reproduction",
    "phycon",
    "protonet",
    "repro",
    "riei",
    "sgc",
    "tifs",
    "wisig_paper",
}

CVS_TOKENS = {
    "adv3",
    "bex",
    "cen",
    "cvs",
    "cvsrffi",
    "dg",
    "dgleo",
    "epoc",
    "fed_fewshot",
    "fl82",
    "lac",
    "local_dryrun",
    "next",
    "phase1",
    "phase2",
    "safd",
    "sat",
    "spaceborne",
    "stage2",
}

SKIP_NAMES = {"cvs", "old_logs", "_organization_manifests"}


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    bucket: str
    reason: str


def classify(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if any(token in lowered for token in REPRO_TOKENS):
        return "reproduction", "matched reproduction/comparison token"
    if any(token in lowered for token in CVS_TOKENS):
        return "cvs", "matched CVS token"
    return "cvs", "default CVS bucket for training logs"


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.name
    index = 1
    while True:
        candidate = parent / f"{stem}__dup{index}"
        if not candidate.exists():
            return candidate
        index += 1


def add_plan(plans: list[MovePlan], source: Path, dest_root: Path, bucket: str, reason: str) -> None:
    if not source.exists():
        return
    destination = unique_destination(dest_root / source.name)
    if source.resolve() == destination.resolve():
        return
    plans.append(MovePlan(source=source, destination=destination, bucket=bucket, reason=reason))


def build_local_plan(project_root: Path) -> list[MovePlan]:
    plans: list[MovePlan] = []
    logs_root = project_root / "logs"
    cvs_root = logs_root / "cvs"
    repro_root = project_root / "paper_reproduction" / "logs"

    if logs_root.exists():
        for child in sorted(logs_root.iterdir(), key=lambda p: p.name.lower()):
            if child.name in SKIP_NAMES or not child.is_dir():
                continue
            bucket, reason = classify(child.name)
            dest_root = repro_root / "current" if bucket == "reproduction" else cvs_root / "current"
            add_plan(plans, child, dest_root, bucket, reason)

    old_root = logs_root / "old_logs"
    if old_root.exists():
        for batch in sorted(old_root.iterdir(), key=lambda p: p.name.lower()):
            top_level = batch / "top_level"
            if not top_level.is_dir():
                continue
            for child in sorted(top_level.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir():
                    continue
                bucket, reason = classify(child.name)
                if bucket == "reproduction":
                    dest_root = repro_root / "history" / batch.name
                else:
                    dest_root = cvs_root / "history" / batch.name
                add_plan(plans, child, dest_root, bucket, reason)

    for legacy_name in ["5.14-logs", "5.15logs", "5.16logs", "5.17logs", "5.18logs", "5.19logs", "5.20-adapt-logs"]:
        legacy = project_root / legacy_name
        if legacy.exists():
            add_plan(plans, legacy, cvs_root / "history" / "legacy_dated_roots", "cvs", "legacy dated CVS log root")

    for tmp_name in ["tmp_logs", "tmp_cli_logs", "tmp_dry_logs"]:
        tmp_root = project_root / tmp_name
        if not tmp_root.exists():
            continue
        for child in sorted(tmp_root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir() and child.suffix.lower() not in {".log", ".out"}:
                continue
            bucket, reason = classify(child.name)
            dest_root = repro_root / "tmp" / tmp_name if bucket == "reproduction" else cvs_root / "tmp" / tmp_name
            add_plan(plans, child, dest_root, bucket, reason)

    return plans


def write_manifest(manifest: Path, plans: list[MovePlan], executed: bool) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["executed", "bucket", "source", "destination", "reason"])
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "executed": "true" if executed else "false",
                    "bucket": plan.bucket,
                    "source": str(plan.source),
                    "destination": str(plan.destination),
                    "reason": plan.reason,
                }
            )


def execute_plan(plans: list[MovePlan]) -> None:
    for plan in plans:
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(plan.source), str(plan.destination))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate CVS logs from reproduction/comparison logs.")
    parser.add_argument("--project-root", default=".", help="Project root.")
    parser.add_argument("--manifest-dir", default="", help="Directory for manifest CSV.")
    parser.add_argument("--execute", action="store_true", help="Move files. Omit for dry-run.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else project_root / "analysis" / f"training_log_reorg_{stamp}"
    plans = build_local_plan(project_root)
    if args.execute:
        execute_plan(plans)
    write_manifest(manifest_dir / ("executed_manifest.csv" if args.execute else "dry_run_manifest.csv"), plans, args.execute)
    counts = {"cvs": 0, "reproduction": 0}
    for plan in plans:
        counts[plan.bucket] = counts.get(plan.bucket, 0) + 1
    print(f"planned_moves={len(plans)}")
    for bucket, count in sorted(counts.items()):
        print(f"{bucket}={count}")
    print(f"manifest_dir={manifest_dir}")
    print(f"executed={'true' if args.execute else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
