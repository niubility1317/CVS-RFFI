from __future__ import annotations

from pathlib import PurePosixPath

from scripts.run_cvs_stage2_landlock_pinned import (
    _audit_ledger,
    _predictor_failure_message,
)


def _ledger(*paths: str) -> list[dict[str, object]]:
    return [
        {"path": path, "successful_open_count": 1, "syscalls": ["openat"]}
        for path in paths
    ]


def test_audit_does_not_match_raw_inside_benign_module_names() -> None:
    audit = _audit_ledger(
        _ledger(
            "/runtime/code/predictor.py",
            "/output/prediction.cvspred",
            "/opt/python/site-packages/sympy/expressionrawdomain.pyc",
            "/opt/python/site-packages/torch/graph_drawer.pyc",
        ),
        runtime_root=PurePosixPath("/runtime/code"),
        output_root=PurePosixPath("/output"),
        system_roots=[PurePosixPath("/opt/python")],
        forbidden_roots=[PurePosixPath("/forbidden")],
        package_root=PurePosixPath("/package"),
    )

    assert audit["status"] == "PASS"
    assert audit["violations"] == []


def test_audit_still_rejects_sensitive_path_tokens() -> None:
    audit = _audit_ledger(
        _ledger(
            "/runtime/code/predictor.py",
            "/output/prediction.cvspred",
            "/opt/python/cache/clean_cache/data.npz",
        ),
        runtime_root=PurePosixPath("/runtime/code"),
        output_root=PurePosixPath("/output"),
        system_roots=[PurePosixPath("/opt/python")],
        forbidden_roots=[PurePosixPath("/forbidden")],
        package_root=PurePosixPath("/package"),
    )

    assert audit["status"] == "FAIL"
    assert audit["violations"] == [
        {"path": "/opt/python/cache/clean_cache/data.npz", "reason": "sensitive_path_opened"}
    ]


def test_predictor_failure_message_keeps_only_bounded_stderr_tail() -> None:
    stderr = "prefix" + "x" * 5000 + "TRACEBACK_END"
    message = _predictor_failure_message(1, stderr)

    assert message.startswith("Landlock predictor failed with return code 1; stderr_tail=")
    assert message.endswith("TRACEBACK_END")
    assert "prefix" not in message
    assert len(message.split("stderr_tail=", 1)[1]) == 4000
