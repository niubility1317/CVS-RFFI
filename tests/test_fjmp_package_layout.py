import unittest
from pathlib import Path


class FJMPPackageLayoutTest(unittest.TestCase):
    def test_fjmp_implementation_lives_under_fjmp_package(self):
        root = Path(__file__).resolve().parents[1]

        self.assertTrue((root / "FJMP" / "__init__.py").is_file())
        self.assertTrue((root / "FJMP" / "frozen_joint_prototype_head.py").is_file())
        self.assertTrue((root / "FJMP" / "experiment_manifest.py").is_file())
        self.assertTrue((root / "FJMP" / "summarize_experiments.py").is_file())

    def test_project_imports_fjmp_package_path(self):
        root = Path(__file__).resolve().parents[1]
        files_to_check = [
            root / "train_fjmp.py",
            root / "scripts" / "run_fjmp_v2_8gpu.sh",
            root / "tests" / "test_frozen_joint_prototype_head.py",
            root / "tests" / "test_fjmp_experiment_design_v2.py",
        ]

        for path in files_to_check:
            text = path.read_text(encoding="utf-8")
            self.assertIn("FJMP", text, str(path))
            self.assertNotIn("from frozen_joint_prototype_head import", text, str(path))
            self.assertNotIn("from fjmp_experiment_manifest import", text, str(path))


if __name__ == "__main__":
    unittest.main()
