import os
import unittest
from unittest.mock import patch


class D93125StabilityThreadTest(unittest.TestCase):
    def test_child_cpu_thread_caps_override_uncapped_libraries(self):
        from scripts import run_d93_125_stability as launcher

        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "128",
                "MKL_NUM_THREADS": "128",
                "OPENBLAS_NUM_THREADS": "128",
            },
            clear=True,
        ):
            configured = launcher._configure_child_cpu_threads(2)
            for key in launcher._CPU_THREAD_ENV_VARS:
                self.assertEqual(os.environ[key], "2")
                self.assertEqual(configured[key], "2")
            self.assertEqual(os.environ["CVSRFFI_CPU_THREADS"], "2")
            self.assertEqual(os.environ["CVSRFFI_CPU_INTEROP_THREADS"], "1")

    def test_parser_locks_candidate_to_d93(self):
        from scripts import run_d93_125_stability as launcher

        action = next(
            item for item in launcher.parser()._actions if item.dest == "candidate"
        )
        self.assertEqual(tuple(action.choices), launcher.CANDIDATES_D93)


if __name__ == "__main__":
    unittest.main()
