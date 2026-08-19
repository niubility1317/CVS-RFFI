#!/usr/bin/env python3
"""Build a bounded CVS-RFFI GitHub release snapshot from the local workspace.

The source checkout at E:/type10-7 is not a Git repository. This script copies
auditable CVS source, protocol, tests, and launchers into the Git-backed release
repository, writes a compact manifest, and can commit/push the result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


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
    "baseline_runs",
    "paper_resnet",
    "snapshots",
    "superpowers",
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
    "fedbase_paper_trainer.py",
    "make_repro_snapshot.py",
    "nohup.out",
    "n607_ssh_config",
    "paper_checklist.md",
    "paper_original_matrix.md",
    "repro_gap.md",
    "test_baseline_paper_launchers.py",
    "test_drift_table1_paper_parity.py",
    "test_fedbase_launcher.py",
    "test_fedbase_method_isolation.py",
    "test_fedbase_paper_methods.py",
    "test_fedbase_paper_trainer.py",
    "test_paper_reproduction_feature_separation.py",
    "test_paper_reproduction_paper_parity.py",
    "test_paper_reproduction_protonet.py",
}

CORE_TREE_MIRRORS = [
    ("code", "code"),
    ("baselines", "baselines"),
    ("paper_reproduction", "paper_reproduction"),
    ("tests", "tests"),
]

EXCLUDE_ROOT_LAUNCHERS = {
    "run_fedbase_paper_queue.sh",
    "run_paper_reproduction_wisig_n607_smoke.sh",
    "run_rafl_input_versions_queue.sh",
    "run_riei_drift_core16_queue.sh",
    "run_riei_drift_core_next10_queue.sh",
    "run_riei_drift_fixed_paper_core_queue.sh",
    "run_riei_original_table3_queue.sh",
    "run_wisig_paper_scope_queue.sh",
}

PRESERVE_RELEASE_DESTS = {
    "baselines/PAPER_CODE_AUDIT.md",
    "baselines/README.md",
    "paper_reproduction/README.md",
    "scripts/launchers/run_cvs_baseline_queue.sh",
    "scripts/launchers/run_cvsrffi_riei_drift_r010_queue.sh",
    "tests/test_cvs_rffi_launcher.py",
}

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
    rel = path.relative_to(source_root).as_posix()
    if name.startswith("SYNC_MANIFEST"):
        return "sync manifest excluded"
    if rel.startswith("paper_reproduction/configs/") and "cvs_stage2c" not in name:
        return "non-CVS paper reproduction config"
    if rel in {
        "paper_reproduction/feature_separation_crossrx/train.py",
    }:
        return "paper-only training entrypoint"
    if rel.startswith("code/scripts/") and any(token in name.lower() for token in ("paper", "repro")):
        return "non-CVS paper/reproduction script"
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
            dst = dest_dir / rel
            dst_rel = dst.relative_to(repo_root).as_posix()
            if dst_rel in PRESERVE_RELEASE_DESTS:
                included_rel.add(rel_str)
                manifest["skipped"].append(
                    {
                        "source": src.relative_to(source_root).as_posix(),
                        "dest": dst_rel,
                        "reason": "preserved release-maintained file",
                    }
                )
                continue
            if allowed_suffix_set and src.suffix.lower() not in allowed_suffix_set:
                if not any(fnmatch.fnmatch(file_name, pat) for pat in allow_pattern_list):
                    manifest["skipped"].append(
                        {
                            "source": src.relative_to(source_root).as_posix(),
                            "dest": dst_rel,
                            "reason": "suffix not allowlisted",
                        }
                    )
                    continue
            if copy_file(src, dst, source_root, repo_root, max_bytes, manifest, category):
                included_rel.add(rel_str)
    if dest_dir.exists():
        for current, dirs, files in os.walk(dest_dir):
            for file_name in files:
                target = Path(current) / file_name
                rel = target.relative_to(dest_dir).as_posix()
                target_rel = target.relative_to(repo_root).as_posix()
                if target_rel in PRESERVE_RELEASE_DESTS:
                    continue
                if rel not in included_rel:
                    target.unlink()
                    manifest["removed"].append(target_rel)
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
    launchers = sorted(source_root.glob("run_*.sh"))
    dest_dir = repo_root / "scripts" / "launchers"
    dest_dir.mkdir(parents=True, exist_ok=True)
    included = set()
    for launcher in launchers:
        dst = dest_dir / launcher.name
        dst_rel = dst.relative_to(repo_root).as_posix()
        if launcher.name in EXCLUDE_ROOT_LAUNCHERS:
            manifest["skipped"].append(
                {
                    "source": launcher.relative_to(source_root).as_posix(),
                    "dest": dst_rel,
                    "reason": "non-CVS paper launcher excluded from GitHub release",
                }
            )
            continue
        if dst_rel in PRESERVE_RELEASE_DESTS:
            included.add(launcher.name)
            manifest["skipped"].append(
                {
                    "source": launcher.relative_to(source_root).as_posix(),
                    "dest": dst_rel,
                    "reason": "preserved release-maintained launcher",
                }
            )
            continue
        if copy_file(launcher, dst, source_root, repo_root, max_bytes, manifest, "launchers"):
            included.add(launcher.name)
    for target in dest_dir.glob("run_*.sh"):
        target_rel = target.relative_to(repo_root).as_posix()
        if target_rel in PRESERVE_RELEASE_DESTS:
            continue
        if target.name not in included:
            target.unlink()
            manifest["removed"].append(target_rel)
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


def write_markdown_outputs(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    repo_root: Path = args.repo_root
    status_path = repo_root / "docs" / "RELEASE_SNAPSHOT.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = manifest["timestamp_local"]

    included_count = len(manifest["included"])
    skipped_count = len(manifest["skipped"])
    status_body = [
        "# CVS GitHub发布快照",
        "",
        f"- 生成时间：{manifest['timestamp_utc']} UTC",
        f"- 本地时间戳：`{timestamp}`",
        f"- 来源工作区：`{args.source_root}`",
        f"- GitHub发布工作区：`{args.repo_root}`",
        f"- 同步文件数：{included_count}",
        f"- 跳过文件数：{skipped_count}",
        "- 生成文件清单：`docs/release_manifest_latest.json`",
        "",
        "## 已同步范围",
        "",
        "- 核心代码：`code/`、`baselines/`、`paper_reproduction/`、`tests/`。",
        "- 项目控制文件：`docs/source_controls/AGENTS.full.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`。",
        "- 自动化工具：`tools/`中允许公开的`.py`、`.md`、`.ps1`文件。",
        "- 启动脚本：`scripts/launchers/run_*.sh`。",
        "- 公开协议文档：`docs/PROJECT_PROTOCOL.md`、`docs/GROUND_TRAINING.md`、`docs/DEPLOYMENT_PHASES.md`、`docs/PUBLISH_SCOPE.md`。",
    ]
    status_body.extend(
        [
            "",
            "## 本轮落实",
            "",
            "- GitHub发布范围已收敛为CVS-only：只同步CVS相关源码、协议、工具、launcher、测试和发布说明。",
            "- 已阻断实验记录、AI审查提示/输出、source notes、baseline历史运行产物进入发布仓库。",
            "- RIEI/DRIFT仅作为CVS对照baseline保留，不上传paper-only队列、paper parity测试或Fedbase paper材料。",
            "",
            "## 边界",
            "",
            "- 不上传`experiment_records/`、`docs/source_notes/`、`docs/source_workspace_docs/`、`docs/analysis_requests/`或`docs/ai_review/`。",
            "- 不上传WiSig/ManySig数据集、模型权重、checkpoint、原始大日志、N607凭据或本地密钥。",
            "- 不上传baseline历史运行产物、自动化报告、服务器日志或本地snapshot。",
            "- 不上传paper-only复现队列、paper parity测试、Fedbase paper训练器、`paper_resnet/`或非CVS paper reproduction配置。",
            "- 指标解释必须绑定同一run或同一candidate row，不能把不同row的单项最大值拼成结论。",
            "- clean view只能作为control/reference；Stage2-A/B不能声明seen-new identity accuracy。",
        ]
    )
    status_path.write_text("\n".join(status_body) + "\n", encoding="utf-8")
    manifest["generated"].append(status_path.relative_to(repo_root).as_posix())


def write_manifest(args: argparse.Namespace, manifest: Dict[str, Any]) -> Path:
    repo_root: Path = args.repo_root
    manifest_dir = repo_root / "docs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"release_manifest_{manifest['timestamp_local']}.json"
    latest_path = manifest_dir / "release_manifest_latest.json"

    public_manifest = {
        "timestamp_utc": manifest["timestamp_utc"],
        "timestamp_local": manifest["timestamp_local"],
        "release_scope": "CVS-only",
        "included_count": len(manifest["included"]),
        "skipped_count": len(manifest["skipped"]),
        "removed_count": len(manifest["removed"]),
        "missing_count": len(manifest["missing"]),
        "generated": manifest["generated"],
        "missing": manifest["missing"],
        "included": [
            {
                "dest": item["dest"],
                "size": item["size"],
                "sha256": item["sha256"],
                "category": item["category"],
                "action": item["action"],
            }
            for item in manifest["included"]
        ],
        "excluded_scope": [
            "experiment_records/",
            "automation_reports/",
            "code/snapshots/",
            "baselines/baseline_runs/",
            "docs/source_notes/",
            "docs/source_workspace_docs/",
            "docs/analysis_requests/",
            "docs/ai_review/",
            "paper parity tests",
            "paper_resnet/",
            "fedbase paper trainer",
            "non-CVS paper reproduction configs",
            "paper-only launchers and rerun templates",
        ],
    }

    payload = json.dumps(public_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
    commit_message = f"自动整理CVS项目快照 {manifest_path.stem.replace('release_manifest_', '')}"
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
            ["git", "push", "--set-upstream", "origin", "HEAD"],
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
    }
    copy_core_sources(args, manifest)
    write_markdown_outputs(args, manifest)
    manifest_path = write_manifest(args, manifest)
    print(f"Snapshot manifest: {manifest_path}")
    print(f"Included: {len(manifest['included'])}; skipped: {len(manifest['skipped'])}; generated: {len(manifest['generated'])}")
    maybe_commit_and_push(args, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
