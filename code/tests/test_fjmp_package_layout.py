import unittest
from pathlib import Path


class FJMPPackageLayoutTest(unittest.TestCase):
    def test_fjmp_implementation_lives_under_fjmp_package(self):
        root = Path(__file__).resolve().parents[1]

        self.assertTrue((root / "FJMP" / "__init__.py").is_file())
        self.assertTrue((root / "FJMP" / "train_fjmp.py").is_file())
        self.assertTrue((root / "FJMP" / "frozen_joint_prototype_head.py").is_file())
        self.assertTrue((root / "FJMP" / "experiment_manifest.py").is_file())
        self.assertTrue((root / "FJMP" / "prototype_metrics.py").is_file())
        self.assertTrue((root / "FJMP" / "summarize_experiments.py").is_file())

    def test_project_imports_fjmp_package_path_and_has_no_duplicate_compatibility_shims(self):
        root = Path(__file__).resolve().parents[1]
        package_files = [
            root / "FJMP" / "train_fjmp.py",
            root / "FJMP" / "prototype_metrics.py",
            root / "scripts" / "run_fjmp_v2_8gpu.sh",
            root / "scripts" / "run_fjmp_sgv_bp_8gpu.sh",
            root / "tests" / "test_fjmp_experiment_design_v2.py",
        ]

        for path in package_files:
            text = path.read_text(encoding="utf-8")
            self.assertIn("FJMP", text, str(path))
            self.assertNotIn("from frozen_joint_prototype_head import", text, str(path))
            self.assertNotIn("from fjmp_experiment_manifest import", text, str(path))

        duplicate_files = [
            root / "train_fjmp.py",
            root / "fjmp_experiment_manifest.py",
            root / "frozen_joint_prototype_head.py",
            root / "summarize_fjmp_experiments.py",
            root / "utils" / "prototype_metrics.py",
            root / "models" / "fjmp_v2_proto_head.py",
            root / "losses" / "fjmp_v2_losses.py",
        ]
        for path in duplicate_files:
            self.assertFalse(path.exists(), f"remove duplicate FJMP compatibility file: {path}")


if __name__ == "__main__":
    unittest.main()
