import unittest
from pathlib import Path


class SSDGPackageLayoutTest(unittest.TestCase):
    def test_ssdg_implementation_lives_under_ssdg_package(self):
        root = Path(__file__).resolve().parents[1]

        self.assertTrue((root / "SSDG" / "__init__.py").is_file())
        self.assertTrue((root / "SSDG" / "train_ssdg.py").is_file())

    def test_project_uses_ssdg_package_path_without_root_level_copy(self):
        root = Path(__file__).resolve().parents[1]
        package_files = [
            root / "SSDG" / "train_ssdg.py",
            root / "scripts" / "run_sgc_ssdg_6gpu.sh",
            root / "tests" / "test_post_stage_trainers.py",
        ]

        for path in package_files:
            text = path.read_text(encoding="utf-8")
            self.assertIn("SSDG", text, str(path))

        self.assertFalse((root / "train_ssdg.py").exists())


if __name__ == "__main__":
    unittest.main()
