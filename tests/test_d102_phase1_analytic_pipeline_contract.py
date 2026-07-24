from __future__ import annotations

from pathlib import Path


PIPELINE = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d102_phase1_analytic_held_pipeline.sh"
)


def test_pipeline_leaves_atomic_writer_subdirectories_absent() -> None:
    source = PIPELINE.read_text(encoding="utf-8")
    assert 'mkdir "$OUTPUT_ROOT"\n' in source
    assert 'mkdir "$OUTPUT_ROOT/tap"' not in source
    assert 'mkdir "$OUTPUT_ROOT/held"' not in source
    assert 'mkdir "$OUTPUT_ROOT/bundle"' not in source
    assert '--output-dir "$OUTPUT_ROOT/tap"' in source
    assert '--output "$OUTPUT_ROOT/held/phase1_analytic_held.json"' in source
    assert '--output-dir "$OUTPUT_ROOT/bundle"' in source


def test_pipeline_exclusively_creates_pid_and_exit_markers() -> None:
    source = PIPELINE.read_text(encoding="utf-8")
    assert '[[ ! -e "$STATUS_PATH" && ! -e "$PID_PATH" ]]' in source
    assert 'mv "$temporary" "$PID_PATH"' in source
    assert 'mv "$temporary" "$STATUS_PATH"' in source
