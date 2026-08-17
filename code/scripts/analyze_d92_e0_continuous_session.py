#!/usr/bin/env python3
"""Truth-last CLI for D92 E0 continuous-session artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


class D92ContinuousSessionAnalyzerError(RuntimeError):
    """Raised when truth-last analysis cannot be started safely."""


AnalysisEntry = Callable[..., Mapping[str, Any] | None]


def _load_analysis_entry() -> AnalysisEntry:
    # The analyzer is deliberately imported only after the runner has closed
    # prediction artifacts.  This keeps the prediction CLI truth-free.
    from cvsrffi.stage2_d92_continuous_session_analysis import (  # noqa: PLC0415
        analyze_continuous_session_run,
    )

    return analyze_continuous_session_run


def _ensure_new_analysis_root(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"analysis output already exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def analyze(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    truth_root: str | Path,
    analysis_root: str | Path,
    analysis_entry: AnalysisEntry | None = None,
) -> Mapping[str, Any] | None:
    """Call the public truth-last analyzer exactly once.

    This wrapper does not inspect truth itself and does not expose truth to the
    prediction runner.  It only reserves a fresh analysis destination before
    handing the four paths to the dedicated analysis implementation.
    """

    destination = Path(analysis_root).resolve()
    _ensure_new_analysis_root(destination)
    entry = analysis_entry or _load_analysis_entry()
    arguments = {
        "manifest_path": Path(manifest_path),
        "output_root": Path(output_root),
        "truth_root": Path(truth_root),
        "analysis_root": destination,
    }
    try:
        signature = inspect.signature(entry)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        arguments = {
            name: value
            for name, value in arguments.items()
            if name in signature.parameters
        }
    try:
        result = entry(**arguments)
    except TypeError as error:
        raise D92ContinuousSessionAnalyzerError(
            f"analysis entry signature/API mismatch: {error}"
        ) from error
    if result is not None and not isinstance(result, Mapping):
        raise D92ContinuousSessionAnalyzerError("analysis entry must return a mapping")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D92 E0 continuous-session truth-last analyzer"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(
        manifest_path=args.manifest,
        output_root=args.output_root,
        truth_root=args.truth_root,
        analysis_root=args.analysis_root,
    )
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI help
    raise SystemExit(main())


__all__ = [
    "D92ContinuousSessionAnalyzerError",
    "analyze",
    "build_parser",
    "main",
]
