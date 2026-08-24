"""Launch or dry-run the frozen Task8 Phase1 meta-adapter entry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_phase1_entry import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_meta_phase1_config,
)


def _resolve_path(value: str | os.PathLike[str], *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _resolve_config_asset(value: str | os.PathLike[str], *, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    beside_config = (config_path.parent / path).resolve()
    if beside_config.exists():
        return beside_config
    return (PROJECT_ROOT / path).resolve()


def _require_readable_file(path: Path, *, field_name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{field_name} is not a readable regular file: {path}")
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise OSError(f"{field_name} is not readable: {path}") from exc


def _build_command(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    base_checkpoint: Path,
    wisig_pkl: Path,
    output_root: Path,
    python_executable: str,
    gpu: str,
) -> list[str]:
    command = [
        python_executable,
        str(PROJECT_ROOT / "code" / "train.py"),
        "--dataset",
        "wisig",
        "--wisig_pkl",
        str(wisig_pkl),
        "--init_checkpoint",
        str(base_checkpoint),
        "--use_cvs_meta_adapter",
        "--meta_config",
        str(config_path),
        "--meta_adapter_rank",
        str(config["adapter"]["rank"]),
        "--meta_adapter_sites",
        ",".join(config["adapter"]["sites"]),
        "--meta_inner_steps",
        str(config["adapter"]["inner_steps"]),
        "--meta_inner_max_steps",
        str(config["adapter"]["deployment_max_steps"]),
        "--meta_output_root",
        str(output_root),
        "--seed",
        str(config["seed"]),
        "--wisig_equalized",
        str(config["wisig_equalized"]),
        "--wisig_out_len",
        str(config["wisig_out_len"]),
        "--wisig_domain",
        str(config["wisig_domain"]),
        "--wisig_max_day123_per_combo",
        str(config["wisig_max_day123_per_combo"]),
        "--wisig_train_rxs",
        ",".join(str(item) for item in config["source_receiver_ids"]),
        "--wisig_train_days",
        ",".join(str(item) for item in config["source_days"]),
    ]
    if gpu.strip():
        command.extend(["--device", "cuda"])
    else:
        command.extend(["--device", "cpu"])
    return command


def build_launch_plan(
    config_path: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    python_executable: str | None = None,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Validate config and resolve an immutable run plan without mutating disk."""

    config_path_abs = _resolve_path(config_path, base=PROJECT_ROOT).resolve()
    config = load_meta_phase1_config(config_path_abs)
    resolved_checkpoint = _resolve_config_asset(str(config["base_checkpoint"]), config_path=config_path_abs)
    resolved_wisig = _resolve_config_asset(str(config["wisig_pkl"]), config_path=config_path_abs)
    _require_readable_file(resolved_checkpoint, field_name="base_checkpoint")
    _require_readable_file(resolved_wisig, field_name="wisig_pkl")
    resolved_output = (
        _resolve_path(output_root, base=PROJECT_ROOT)
        if output_root is not None and str(output_root).strip()
        else PROJECT_ROOT / "runs" / str(config["run_id"])
    ).resolve()
    if resolved_output.exists():
        raise FileExistsError(f"immutable meta Phase1 output root already exists: {resolved_output}")
    selected_gpu = str(gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    selected_python = str(python_executable or sys.executable)
    command = _build_command(
        config_path=config_path_abs,
        config=config,
        base_checkpoint=resolved_checkpoint,
        wisig_pkl=resolved_wisig,
        output_root=resolved_output,
        python_executable=selected_python,
        gpu=selected_gpu,
    )
    expected_artifacts = (
        "logs.jsonl",
        "metrics.csv",
        "selected_meta_bundle.pt",
        "run_summary.json",
        "config_snapshot.json",
        "source_adaptation_curve.json",
    )
    return {
        "schema": config["schema"],
        "run_id": config["run_id"],
        "base_checkpoint": str(resolved_checkpoint),
        "wisig_pkl": str(resolved_wisig),
        "source_receiver_ids": list(config["source_receiver_ids"]),
        "gpu": selected_gpu or "cpu",
        "python": selected_python,
        "command": command,
        "output_root": str(resolved_output),
        "expected_artifacts": list(expected_artifacts),
        "config_path": str(config_path_abs),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--python", dest="python_executable", type=str, default="")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating a process or run root")
    return parser


def _print_plan(plan: Mapping[str, Any]) -> None:
    print(f"run_id={plan['run_id']}")
    print(f"base_checkpoint={plan['base_checkpoint']}")
    print(f"wisig_pkl={plan['wisig_pkl']}")
    print(f"source_receiver_ids={plan['source_receiver_ids']}")
    print(f"GPU={plan['gpu']}")
    print(f"output_root={plan['output_root']}")
    print("command=" + subprocess.list2cmdline([str(item) for item in plan["command"]]))
    print("expected_artifacts=" + json.dumps(plan["expected_artifacts"], ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    plan = build_launch_plan(
        args.config,
        output_root=args.output_root or None,
        python_executable=args.python_executable or None,
        gpu=args.gpu,
    )
    _print_plan(plan)
    if args.dry_run:
        print("dry_run=true")
        return 0

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    completed = subprocess.run(plan["command"], cwd=str(PROJECT_ROOT), env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
