import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("package_name", "init_path"),
    (
        ("cvsrffi", PROJECT_ROOT / "code" / "cvsrffi" / "__init__.py"),
        ("SSDG", PROJECT_ROOT / "code" / "SSDG" / "__init__.py"),
    ),
)
def test_release_overlay_package_can_import_changed_and_base_modules(
    tmp_path: Path,
    package_name: str,
    init_path: Path,
):
    overlay_root = tmp_path / "overlay"
    base_root = tmp_path / "base"
    overlay_package = overlay_root / package_name
    base_package = base_root / package_name
    overlay_package.mkdir(parents=True)
    base_package.mkdir(parents=True)
    (overlay_package / "__init__.py").write_text(init_path.read_text(encoding="utf-8"), encoding="utf-8")
    (overlay_package / "changed_module.py").write_text("VALUE = 'changed'\n", encoding="utf-8")
    (base_package / "__init__.py").write_text("\n", encoding="utf-8")
    (base_package / "existing_module.py").write_text("VALUE = 'existing'\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(overlay_root), str(base_root)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"from {package_name}.changed_module import VALUE as changed; "
                f"from {package_name}.existing_module import VALUE as existing; "
                "assert (changed, existing) == ('changed', 'existing')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
