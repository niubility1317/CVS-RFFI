from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_d104_support_geometry_audit_cli_has_no_query_surface() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "audit_d104_angq_support_geometry_local.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tap-archive" in result.stdout
    assert "--output-json" in result.stdout
    assert "query" not in result.stdout.lower()
    assert "truth" not in result.stdout.lower()


def test_d104_support_geometry_audit_binds_historical_exposure() -> None:
    script_root = Path(__file__).resolve().parents[1] / "code" / "scripts"
    sys.path.insert(0, str(script_root))
    try:
        import audit_d104_angq_support_geometry_local as audit

        assert audit.EXPECTED_TAP_ROWS == 8400
        assert audit.HISTORICAL_DIAGNOSTIC_QUERY_COUNT == 2478
        assert len(audit.HISTORICAL_DIAGNOSTIC_QUERY_ID_ROOT_SHA256) == 64
    finally:
        sys.path.remove(str(script_root))
