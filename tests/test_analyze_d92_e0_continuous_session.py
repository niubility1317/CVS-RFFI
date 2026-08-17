from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_d92_e0_continuous_session as analyzer


def test_analyzer_forwards_truth_last_arguments_without_runner_inputs(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path, Path, Path]] = []

    def fake_analysis(
        manifest_path: Path,
        output_root: Path,
        truth_root: Path,
        analysis_root: Path,
    ) -> dict[str, object]:
        calls.append((manifest_path, output_root, truth_root, analysis_root))
        return {"status": "ANALYZED_TRUTH_LAST", "session_count": 5}

    manifest = tmp_path / "matrix_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "outputs"
    truth_root = tmp_path / "truth"
    analysis_root = tmp_path / "analysis"

    result = analyzer.analyze(
        manifest_path=manifest,
        output_root=output_root,
        truth_root=truth_root,
        analysis_root=analysis_root,
        analysis_entry=fake_analysis,
    )

    assert result["status"] == "ANALYZED_TRUTH_LAST"
    assert calls == [(manifest, output_root, truth_root, analysis_root)]


def test_analyzer_refuses_nonempty_analysis_output(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir()
    (root / "existing.json").write_text("{}", encoding="utf-8")

    try:
        analyzer.analyze(
            manifest_path=tmp_path / "manifest.json",
            output_root=tmp_path / "outputs",
            truth_root=tmp_path / "truth",
            analysis_root=root,
            analysis_entry=lambda **_: {},
        )
    except FileExistsError as error:
        assert "analysis" in str(error)
    else:  # pragma: no cover - assertion keeps the RED contract explicit
        raise AssertionError("non-empty analysis root must be immutable")


def test_cli_help_is_truth_last_only() -> None:
    help_text = analyzer.build_parser().format_help()
    for name in ("manifest", "output-root", "truth-root", "analysis-root"):
        assert name in help_text
