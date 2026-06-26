import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CVSDataImportPathTest(unittest.TestCase):
    def test_cvs_data_prefers_code_dataset_wisig_over_cwd_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            shadow_dir = Path(tmp)
            (shadow_dir / "dataset_wisig.py").write_text(
                "raise RuntimeError('shadow dataset_wisig imported')\n",
                encoding="utf-8",
            )
            script = textwrap.dedent(
                """
                import baselines.common.cvs_data  # noqa: F401
                import dataset_wisig
                print(dataset_wisig.__file__)
                """
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [
                    str(shadow_dir),
                    str(ROOT / "code"),
                    str(ROOT),
                    env.get("PYTHONPATH", ""),
                ]
            )

            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(shadow_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(
            Path(proc.stdout.strip()).resolve(),
            (ROOT / "code" / "dataset_wisig.py").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
