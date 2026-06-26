from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from paper_reproduction.common.config import contains_unresolved_placeholder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return "git unavailable"
    if completed.returncode != 0:
        return completed.stderr.strip()
    return completed.stdout.strip()


def sha256_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        hashes[rel] = sha256_file(path)
    return hashes


def contains_unspecified(value: object) -> bool:
    if isinstance(value, str):
        return "paper-unspecified" in value
    if isinstance(value, list):
        return any(contains_unspecified(item) for item in value)
    if isinstance(value, dict):
        return any(contains_unspecified(item) for item in value.values())
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an immutable local reproduction snapshot manifest.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the resolved experiment config.")
    parser.add_argument("--command", required=True, help="Exact formal experiment command to run later.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Snapshot output directory.")
    parser.add_argument("--formal", action="store_true", help="Reject unspecified configs and existing output dirs.")
    args = parser.parse_args()

    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    config_data = json.loads(config.read_text(encoding="utf-8"))
    if args.formal and contains_unspecified(config_data):
        raise ValueError("formal snapshot config still contains paper-unspecified")
    if args.formal and contains_unresolved_placeholder(config_data):
        raise ValueError("formal snapshot config still contains unresolved placeholder")

    out_dir = args.out_dir.resolve()
    if args.formal and out_dir.exists():
        raise FileExistsError(f"snapshot output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=not args.formal)

    copied_config = out_dir / config.name
    if args.formal and copied_config.exists():
        raise FileExistsError(f"snapshot config already exists: {copied_config}")
    copied_config.write_bytes(config.read_bytes())
    repo_root = Path.cwd()
    code_root = repo_root / "paper_reproduction"
    code_hashes = sha256_tree(code_root)
    commands = config_data.get("commands") if isinstance(config_data, dict) else None
    if not commands:
        commands = [args.command]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config),
        "config_snapshot": str(copied_config),
        "config_sha256": sha256_file(copied_config),
        "command": args.command,
        "commands": commands,
        "code_sha256": code_hashes,
        "python_executable": os.fspath(Path(sys.executable).resolve()),
        "git_status_short": git_status(Path.cwd()),
    }
    manifest_path = out_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
