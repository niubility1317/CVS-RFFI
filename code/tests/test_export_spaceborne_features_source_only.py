from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import export_spaceborne_features as exporter  # noqa: E402


def test_source_only_cli_is_explicit_and_has_no_target_requirement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_spaceborne_features.py",
            "--out_npz",
            str(tmp_path / "source.npz"),
            "--source_tx_ids",
            "1-1,1-2",
            "--source_only_export",
        ],
    )
    args = exporter.parse_args()
    assert args.source_only_export is True
    assert args.target_old_tx_ids is None
    assert args.new_tx_ids is None


@pytest.mark.parametrize("field", ["target_old_tx_ids", "new_tx_ids", "unknown_tx_ids", "proxy_unknown_tx_ids"])
def test_source_only_role_arguments_are_rejected_before_data_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
):
    argv = [
        "export_spaceborne_features.py",
        "--ckpt",
        str(tmp_path / "missing.pth"),
        "--out_npz",
        str(tmp_path / "source.npz"),
        "--source_tx_ids",
        "1-1,1-2",
        "--source_only_export",
        f"--{field}",
        "9-9",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="source_only_export forbids role arguments"):
        exporter.main()
