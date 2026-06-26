#!/usr/bin/env python3
"""Build a bounded CVS-RFFI GitHub release snapshot from the local workspace.

The source checkout at E:/type10-7 is not a Git repository. This script copies
auditable source, protocol, and selected experiment evidence into the Git-backed
release repository, writes manifests, and can commit/push the result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SOURCE_ROOT = Path("E:/type10-7")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".claude",
    ".codex-remote-attachments",
    ".tmp",
    ".local",
}

EXCLUDE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pth",
    ".pt",
    ".ckpt",
    ".npz",
    ".npy",
    ".pkl",
    ".pickle",
    ".onnx",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
    ".7z",
    ".rar",
    ".pptx",
    ".docx",
    ".xlsx",
    ".pdf",
    ".log",
    ".out",
}

EXCLUDE_FILE_NAMES = {
    "nohup.out",
    "n607_ssh_config",
}

CORE_TREE_MIRRORS = [
    ("code", "code"),
    ("baselines", "baselines"),
    ("paper_reproduction", "paper_reproduction"),
    ("tests", "tests"),
]

ROOT_MARKDOWN_FILES = [
    "AUDIT_CVS_RFFI_GROUND_TO_SPACE_FSL.md",
    "EXPERIMENT_DESIGN.md",
    "findings.md",
    "progress.md",
    "task_plan.md",
    "metrics_summary.csv",
    "evidence_map.csv",
    "missing_experiments.md",
]

IMPORTANT_REPORT_FILE_NAMES = {
    "report.md",
    "clean_cvs_vs_riei_drift.md",
    "missing_artifacts.md",
}

IMPORTANT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".tsv",
}

IMPORTANT_ARTIFACT_NAME_PATTERNS = [
    "*score_table*.csv",
    "*metrics*.json",
    "*manifest*.json",
    "*summary*.json",
    "*summary*.csv",
    "*matrix*.json",
    "*matrix*.csv",
    "*validation*.json",
    "*inventory*.json",
    "*inventory*.csv",
    "*separability*.json",
    "*separability*.csv",
    "*registry*.jsonl",
    "*current*view*.json",
    "*state*.json",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_git(repo_root: Path, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, completed.args, completed.stdout, completed.stderr)
    return completed


def should_skip_file(path: Path, source_root: Path, max_bytes: int) -> Optional[str]:
    name = path.name
    suffix = path.suffix.lower()
    parts = set(path.relative_to(source_root).parts)
    if name in EXCLUDE_FILE_NAMES:
        return "excluded file name"
    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return "excluded directory"
    if any(part.startswith("tmp") for part in parts):
        return "temporary directory"
    if suffix in EXCLUDE_FILE_SUFFIXES:
        return f"excluded suffix {suffix}"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"stat failed: {exc}"
    if size > max_bytes:
        return f"larger than {max_bytes} bytes"
    return None


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root, topdown=False):
        p = Path(current)
        if p == root:
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
        except OSError:
            pass


def copy_file(
    src: Path,
    dst: Path,
    source_root: Path,
    repo_root: Path,
    max_bytes: int,
    manifest: Dict[str, Any],
    category: str,
) -> bool:
    reason = should_skip_file(src, source_root, max_bytes)
    src_rel = src.relative_to(source_root).as_posix()
    dst_rel = dst.relative_to(repo_root).as_posix()
    if reason:
        manifest["skipped"].append({"source": src_rel, "dest": dst_rel, "reason": reason})
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src)
    old_hash = sha256_file(dst) if dst.exists() and dst.is_file() else None
    if old_hash != src_hash:
        shutil.copy2(src, dst)
        action = "copied"
    else:
        action = "unchanged"
    manifest["included"].append(
        {
            "source": src_rel,
            "dest": dst_rel,
            "size": src.stat().st_size,
            "sha256": src_hash,
            "category": category,
            "action": action,
        }
    )
    return True


def mirror_tree(
    source_dir: Path,
    dest_dir: Path,
    source_root: Path,
    repo_root: Path,
    max_bytes: int,
    manifest: Dict[str, Any],
    category: str,
    allowed_suffixes: Optional[Iterable[str]] = None,
    allow_patterns: Optional[Iterable[str]] = None,
) -> None:
    included_rel: set[str] = set()
    allowed_suffix_set = {s.lower() for s in allowed_suffixes} if allowed_suffixes else None
    allow_pattern_list = list(allow_patterns or [])
    if not source_dir.exists():
        manifest["missing"].append(source_dir.relative_to(source_root).as_posix())
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for current, dirs, files in os.walk(source_dir):
        current_path = Path(current)
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDE_DIR_NAMES
            and not d.startswith("tmp")
            and d not in {"runs", "logs", "outputs", "remote_artifacts", "server_log_backups"}
        ]
        for file_name in files:
            src = current_path / file_name
            rel = src.relative_to(source_dir)
            rel_str = rel.as_posix()
            if allowed_suffix_set and src.suffix.lower() not in allowed_suffix_set:
                if not any(fnmatch.fnmatch(file_name, pat) for pat in allow_pattern_list):
                    manifest["skipped"].append(
                        {
                            "source": src.relative_to(source_root).as_posix(),
                            "dest": (dest_dir / rel).relative_to(repo_root).as_posix(),
                            "reason": "suffix not allowlisted",
                        }
                    )
                    continue
            if copy_file(src, dest_dir / rel, source_root, repo_root, max_bytes, manifest, category):
                included_rel.add(rel_str)
    if dest_dir.exists():
        for current, dirs, files in os.walk(dest_dir):
            for file_name in files:
                target = Path(current) / file_name
                rel = target.relative_to(dest_dir).as_posix()
                if rel not in included_rel:
                    target.unlink()
                    manifest["removed"].append(target.relative_to(repo_root).as_posix())
        remove_empty_dirs(dest_dir)


def copy_core_sources(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    source_root: Path = args.source_root
    repo_root: Path = args.repo_root
    max_bytes = args.max_file_mb * 1024 * 1024
    for src_rel, dst_rel in CORE_TREE_MIRRORS:
        mirror_tree(
            source_root / src_rel,
            repo_root / dst_rel,
            source_root,
            repo_root,
            max_bytes,
            manifest,
            category=f"core:{src_rel}",
        )
    mirror_tree(
        source_root / "tools",
        repo_root / "tools",
        source_root,
        repo_root,
        max_bytes,
        manifest,
        category="tools",
        allowed_suffixes={".py", ".md", ".ps1"},
    )
    mirror_tree(
        source_root / "docs",
        repo_root / "docs" / "source_workspace_docs",
        source_root,
        repo_root,
        max_bytes,
        manifest,
        category="source_docs",
        allowed_suffixes={".md", ".txt", ".csv", ".json"},
    )
    launchers = sorted(source_root.glob("run_*.sh"))
    dest_dir = repo_root / "scripts" / "launchers"
    dest_dir.mkdir(parents=True, exist_ok=True)
    included = set()
    for launcher in launchers:
        dst = dest_dir / launcher.name
        if copy_file(launcher, dst, source_root, repo_root, max_bytes, manifest, "launchers"):
            included.add(launcher.name)
    for target in dest_dir.glob("run_*.sh"):
        if target.name not in included:
            target.unlink()
            manifest["removed"].append(target.relative_to(repo_root).as_posix())
    for file_name in ROOT_MARKDOWN_FILES:
        src = source_root / file_name
        if src.exists():
            copy_file(
                src,
                repo_root / "docs" / "source_notes" / file_name,
                source_root,
                repo_root,
                max_bytes,
                manifest,
                "source_notes",
            )
    protocol_dir = repo_root / "docs" / "source_controls"
    if (source_root / "AGENTS.md").exists():
        copy_file(
            source_root / "AGENTS.md",
            protocol_dir / "AGENTS.full.md",
            source_root,
            repo_root,
            max_bytes,
            manifest,
            "source_controls",
        )
    if (source_root / "项目.md").exists():
        copy_file(
            source_root / "项目.md",
            protocol_dir / "PROJECT_PROTOCOL.full.md",
            source_root,
            repo_root,
            max_bytes,
            manifest,
            "source_controls",
        )


def is_important_artifact(path: Path) -> bool:
    name = path.name
    suffix = path.suffix.lower()
    if name in IMPORTANT_REPORT_FILE_NAMES:
        return True
    if suffix not in IMPORTANT_ARTIFACT_SUFFIXES:
        return False
    return any(fnmatch.fnmatch(name, pat) for pat in IMPORTANT_ARTIFACT_NAME_PATTERNS)


def latest_report_dirs(report_root: Path, keep: int) -> List[Path]:
    if not report_root.exists():
        return []
    candidates = []
    for child in report_root.iterdir():
        if not child.is_dir():
            continue
        report = child / "report.md"
        if report.exists():
            try:
                mtime = max(child.stat().st_mtime, report.stat().st_mtime)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, child))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [p for _, p in candidates[:keep]]


def write_tail_file(src: Path, dst: Path, line_count: int) -> Dict[str, Any]:
    lines: List[str] = []
    with src.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            if len(lines) > line_count:
                lines.pop(0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(lines), encoding="utf-8")
    return {
        "source": str(src),
        "dest": str(dst),
        "source_size": src.stat().st_size,
        "source_sha256": sha256_file(src),
        "tail_lines": len(lines),
    }


def short_component(value: str, max_len: int = 90) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[: max_len - 11]}_{digest}"


def experiment_record_dest(latest_root: Path, report_dir: Path, src: Path) -> Path:
    report_name = short_component(report_dir.name, max_len=90)
    rel = src.relative_to(report_dir)
    if rel.as_posix() == "report.md":
        return latest_root / report_name / "report.md"
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:12]
    return latest_root / report_name / "artifacts" / f"{digest}_{short_component(src.name, max_len=80)}"


def flatten_json_metrics(data: Any, prefix: str = "", limit: int = 80) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(out) >= limit:
        return out
    if isinstance(data, dict):
        for key, value in data.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[child_key] = value
            elif isinstance(value, dict):
                out.update(flatten_json_metrics(value, child_key, limit))
            if len(out) >= limit:
                break
    return out


def score_table_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    columns = reader.fieldnames or []
    metric_candidates = [
        "H_old_new",
        "hmean",
        "old_acc",
        "seen_new_acc",
        "unknown_far",
        "unknown_FAR",
        "auroc",
        "AUROC",
        "fpr95",
        "FPR95",
    ]
    best: Dict[str, Any] = {}
    for metric in metric_candidates:
        if metric not in columns:
            continue
        values: List[Tuple[float, Dict[str, str]]] = []
        for row in rows:
            try:
                values.append((float(row.get(metric, "")), row))
            except (TypeError, ValueError):
                continue
        if values:
            reverse = "far" not in metric.lower() and "fpr" not in metric.lower()
            value, row = sorted(values, key=lambda item: item[0], reverse=reverse)[0]
            best[metric] = {
                "value": value,
                "row": {k: row.get(k) for k in columns[:20]},
            }
    return {"rows": len(rows), "columns": columns, "best": best}


def collect_metrics_inventory(records_root: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not records_root.exists():
        return rows
    for path in sorted(records_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if path.name == "metrics.json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                rows.append(
                    {
                        "artifact": rel,
                        "kind": "metrics_json",
                        "rows": "",
                        "columns": "",
                        "summary": json.dumps(flatten_json_metrics(data), ensure_ascii=False, sort_keys=True),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"artifact": rel, "kind": "metrics_json_error", "rows": "", "columns": "", "summary": str(exc)})
        elif path.name == "score_table.csv":
            try:
                summary = score_table_summary(path)
                rows.append(
                    {
                        "artifact": rel,
                        "kind": "score_table_csv",
                        "rows": summary["rows"],
                        "columns": "|".join(summary["columns"]),
                        "summary": json.dumps(summary["best"], ensure_ascii=False, sort_keys=True),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"artifact": rel, "kind": "score_table_error", "rows": "", "columns": "", "summary": str(exc)})
    return rows


def collect_experiment_records(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    source_root: Path = args.source_root
    repo_root: Path = args.repo_root
    max_bytes = args.max_artifact_mb * 1024 * 1024
    report_root = source_root / "automation_reports" / "CV-SincNet"
    records_root = repo_root / "experiment_records" / "CV-SincNet"
    latest_root = records_root / "latest"
    current_root = records_root / "current"
    if latest_root.exists():
        shutil.rmtree(latest_root)
    latest_root.mkdir(parents=True, exist_ok=True)
    current_root.mkdir(parents=True, exist_ok=True)

    state_files = [
        "stage2_optimizer_state.json",
        "current_state_view_latest_for_automation.json",
        ".current_oldrisk_run_id.txt",
    ]
    for name in state_files:
        src = report_root / name
        if src.exists():
            copy_file(src, current_root / name, source_root, repo_root, max_bytes, manifest, "experiment_state")

    registry = report_root / "optimizer_execution_registry.jsonl"
    if registry.exists():
        tail_meta = write_tail_file(registry, current_root / "optimizer_execution_registry.tail.jsonl", args.registry_tail_lines)
        (current_root / "optimizer_execution_registry.fingerprint.json").write_text(
            json.dumps(tail_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["generated"].append((current_root / "optimizer_execution_registry.tail.jsonl").relative_to(repo_root).as_posix())
        manifest["generated"].append((current_root / "optimizer_execution_registry.fingerprint.json").relative_to(repo_root).as_posix())

    selected_reports = latest_report_dirs(report_root, args.keep_reports)
    manifest["selected_reports"] = [p.name for p in selected_reports]
    for report_dir in selected_reports:
        copied_count = 0
        for current, dirs, files in os.walk(report_dir):
            dirs[:] = [
                d
                for d in dirs
                if d not in EXCLUDE_DIR_NAMES
                and d not in {"logs", "checkpoints"}
                and not d.startswith("tmp")
            ]
            for file_name in files:
                src = Path(current) / file_name
                if not is_important_artifact(src):
                    continue
                if copied_count >= args.max_artifacts_per_report:
                    manifest["skipped"].append(
                        {
                            "source": src.relative_to(source_root).as_posix(),
                            "dest": "",
                            "reason": "max artifacts per report reached",
                        }
                    )
                    continue
                dst = experiment_record_dest(latest_root, report_dir, src)
                if copy_file(src, dst, source_root, repo_root, max_bytes, manifest, "experiment_records"):
                    copied_count += 1

    inventory = collect_metrics_inventory(records_root, repo_root)
    inventory_path = records_root / "metrics_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "kind", "rows", "columns", "summary"])
        writer.writeheader()
        writer.writerows(inventory)
    manifest["generated"].append(inventory_path.relative_to(repo_root).as_posix())


def write_markdown_outputs(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    repo_root: Path = args.repo_root
    records_root = repo_root / "experiment_records" / "CV-SincNet"
    status_path = records_root / "LATEST_SNAPSHOT.md"
    prompt_dir = repo_root / "docs" / "analysis_requests"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = manifest["timestamp_local"]

    selected_reports = manifest.get("selected_reports", [])
    included_count = len(manifest["included"])
    skipped_count = len(manifest["skipped"])
    status_body = [
        "# CVS十二小时项目快照",
        "",
        f"- 生成时间：{manifest['timestamp_utc']} UTC",
        f"- 本地时间戳：`{timestamp}`",
        f"- 来源工作区：`{args.source_root}`",
        f"- GitHub发布工作区：`{args.repo_root}`",
        f"- 同步文件数：{included_count}",
        f"- 跳过文件数：{skipped_count}",
        "- 生成文件清单：`experiment_records/CV-SincNet/snapshot_manifest_latest.json`",
        "",
        "## 已同步范围",
        "",
        "- 核心代码：`code/`、`baselines/`、`paper_reproduction/`、`tests/`。",
        "- 项目控制文件：`docs/source_controls/AGENTS.full.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`。",
        "- 自动化工具：`tools/`中允许公开的`.py`、`.md`、`.ps1`文件。",
        "- 启动脚本：`scripts/launchers/run_*.sh`。",
        "- 最近实验证据：`experiment_records/CV-SincNet/latest/`下的报告、metrics、score table、manifest、matrix、validation和summary类小文件。",
        "",
        "## 最近报告目录",
        "",
    ]
    if selected_reports:
        status_body.extend([f"- `{name}`" for name in selected_reports])
    else:
        status_body.append("- 未找到带`report.md`的自动化报告目录。")
    status_body.extend(
        [
            "",
            "## 边界",
            "",
            "- 未上传WiSig/ManySig数据集、模型权重、checkpoint、原始大日志、N607凭据或本地密钥。",
            "- `optimizer_execution_registry.jsonl`只上传tail和fingerprint，避免十二小时提交持续膨胀。",
            "- 指标解释必须绑定同一run或同一candidate row，不能把不同row的单项最大值拼成结论。",
            "- clean view只能作为control/reference；Stage2-A/B不能声明seen-new identity accuracy。",
        ]
    )
    status_path.write_text("\n".join(status_body) + "\n", encoding="utf-8")
    manifest["generated"].append(status_path.relative_to(repo_root).as_posix())

    prompt_body = [
        "# ChatGPT Pro网页GPT审查提示",
        "",
        f"快照时间：`{timestamp}`",
        "",
        "你正在审查CVS-RFFI/CV-SincNet项目。请先读取GitHub仓库中的以下文件，不要只看README：",
        "",
        "1. `README.md`",
        "2. `docs/source_controls/AGENTS.full.md`",
        "3. `docs/source_controls/PROJECT_PROTOCOL.full.md`",
        "4. `docs/PROJECT_PROTOCOL.md`",
        "5. `experiment_records/CV-SincNet/LATEST_SNAPSHOT.md`",
        "6. `experiment_records/CV-SincNet/metrics_inventory.csv`",
        "7. `experiment_records/CV-SincNet/current/stage2_optimizer_state.json`",
        "8. `experiment_records/CV-SincNet/current/current_state_view_latest_for_automation.json`（如果存在）",
        "9. `experiment_records/CV-SincNet/latest/`下最新报告的`report.md`、`metrics.json`、`score_table.csv`和`manifest.json`。",
        "",
        "请用中文输出，并严格遵守这些边界：",
        "",
        "- 区分startup PASS、landed submit、artifact-complete、runner completion、negative diagnostic和deployment success。",
        "- 不要把Stage2-A/B的unknown rejection写成seen-new identity accuracy。",
        "- 不要把clean view成功写成satellite/LEO deployment success。",
        "- 不要用孤立最大值/最小值拼结论；指标必须来自同一candidate/run row，或明确标为marginal statistics。",
        "- 如果仓库缺少某个文件或指标，写成缺口，不要猜测。",
        "",
        "输出结构必须包含：",
        "",
        "## 证据边界",
        "## 当前主要成果",
        "## 主矛盾",
        "## 次要矛盾",
        "## 必须解决的问题",
        "## 修改建议",
        "## 文件级落地建议",
        "## 下一轮实验矩阵建议",
        "## 不能写入论文/报告的声明",
        "",
        "修改建议必须落到具体文件或模块，例如`code/cvsrffi/spaceborne_fewshot.py`、`tools/spaceborne_fewshot_da_matrix.py`、`tools/optimizer_validate_matrix.py`、`paper_reproduction/cvs_aligned/`、`docs/PROJECT_PROTOCOL.md`或具体报告路径。",
        "",
        "审查完成后，把结果保存为`docs/ai_review/<timestamp>/chatgpt_pro_review.md`，或将正文交给Codex写入该路径并提交。",
    ]
    prompt_path = prompt_dir / f"chatgpt_pro_prompt_{timestamp}.md"
    latest_prompt = prompt_dir / "latest_chatgpt_pro_prompt.md"
    prompt_text = "\n".join(prompt_body) + "\n"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    latest_prompt.write_text(prompt_text, encoding="utf-8")
    manifest["generated"].append(prompt_path.relative_to(repo_root).as_posix())
    manifest["generated"].append(latest_prompt.relative_to(repo_root).as_posix())


def write_manifest(args: argparse.Namespace, manifest: Dict[str, Any]) -> Path:
    repo_root: Path = args.repo_root
    records_root = repo_root / "experiment_records" / "CV-SincNet"
    records_root.mkdir(parents=True, exist_ok=True)
    manifest_path = records_root / f"snapshot_manifest_{manifest['timestamp_local']}.json"
    latest_path = records_root / "snapshot_manifest_latest.json"
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return manifest_path


def maybe_commit_and_push(args: argparse.Namespace, manifest_path: Path) -> None:
    repo_root: Path = args.repo_root
    status = run_git(repo_root, ["status", "--porcelain"], check=True).stdout.strip()
    if not status:
        print("No Git changes to commit.")
        return
    print(status)
    if not args.commit:
        print("Changes left uncommitted because --commit was not set.")
        return
    run_git(repo_root, ["add", "-A"], check=True)
    commit_message = f"自动整理CVS项目快照 {manifest_path.stem.replace('snapshot_manifest_', '')}"
    commit = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if commit.returncode != 0:
        print(commit.stdout)
        print(commit.stderr, file=sys.stderr)
        raise SystemExit(commit.returncode)
    print(commit.stdout)
    if args.push:
        push = subprocess.run(
            ["git", "push"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(push.stdout)
        if push.stderr:
            print(push.stderr, file=sys.stderr)
        if push.returncode != 0:
            raise SystemExit(push.returncode)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--max-file-mb", type=int, default=20)
    parser.add_argument("--max-artifact-mb", type=int, default=20)
    parser.add_argument("--keep-reports", type=int, default=12)
    parser.add_argument("--max-artifacts-per-report", type=int, default=160)
    parser.add_argument("--registry-tail-lines", type=int, default=800)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.source_root = args.source_root.resolve()
    args.repo_root = args.repo_root.resolve()
    if not args.source_root.exists():
        raise SystemExit(f"source root does not exist: {args.source_root}")
    if not (args.repo_root / ".git").exists():
        raise SystemExit(f"repo root is not a Git repository: {args.repo_root}")

    manifest: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "timestamp_local": local_stamp(),
        "source_root": str(args.source_root),
        "repo_root": str(args.repo_root),
        "included": [],
        "skipped": [],
        "removed": [],
        "missing": [],
        "generated": [],
        "selected_reports": [],
    }
    copy_core_sources(args, manifest)
    collect_experiment_records(args, manifest)
    write_markdown_outputs(args, manifest)
    manifest_path = write_manifest(args, manifest)
    print(f"Snapshot manifest: {manifest_path}")
    print(f"Included: {len(manifest['included'])}; skipped: {len(manifest['skipped'])}; generated: {len(manifest['generated'])}")
    maybe_commit_and_push(args, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
