from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = (
    REPO_ROOT
    / "code"
    / "release_bootstrap"
    / "d127"
    / "cvsrffi"
    / "__init__.py"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def test_bootstrap_extends_two_package_roots_and_imports_both_modules(tmp_path, monkeypatch):
    d127_root = tmp_path / "d127_root"
    d106_root = tmp_path / "d106_root"
    _write(
        d127_root / "cvsrffi" / "__init__.py",
        BOOTSTRAP.read_text(encoding="utf-8"),
    )
    _write(
        d127_root / "cvsrffi" / "stage2_d127_marker.py",
        "SOURCE = 'd127'\n",
    )
    _write(
        d106_root / "cvsrffi" / "stage2_d106_phase1_tap.py",
        "SOURCE = 'd106'\n",
    )

    monkeypatch.syspath_prepend(str(d106_root))
    monkeypatch.syspath_prepend(str(d127_root))
    for name in (
        "cvsrffi",
        "cvsrffi.stage2_d127_marker",
        "cvsrffi.stage2_d106_phase1_tap",
    ):
        sys.modules.pop(name, None)

    d127 = importlib.import_module("cvsrffi.stage2_d127_marker")
    d106 = importlib.import_module("cvsrffi.stage2_d106_phase1_tap")

    assert d127.SOURCE == "d127"
    assert d106.SOURCE == "d106"
    package = importlib.import_module("cvsrffi")
    package_paths = {Path(item).resolve() for item in package.__path__}
    assert (d127_root / "cvsrffi").resolve() in package_paths
    assert (d106_root / "cvsrffi").resolve() in package_paths


def test_bootstrap_has_no_project_or_scientific_imports():
    source = BOOTSTRAP.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BOOTSTRAP))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("pkgutil", ["extend_path"])
    ]
    assert "stage2_" not in source
