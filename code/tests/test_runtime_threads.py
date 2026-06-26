import os
import unittest
from unittest.mock import patch


class RuntimeThreadConfigTest(unittest.TestCase):
    def test_default_training_thread_env_caps_cpu_libraries(self):
        from cvsrffi.runtime_threads import configure_cpu_thread_env

        clear_keys = [
            "CVSRFFI_CPU_THREADS",
            "CVSRFFI_CPU_INTEROP_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ]
        env = {k: v for k, v in os.environ.items() if k not in clear_keys}
        with patch.dict(os.environ, env, clear=True):
            info = configure_cpu_thread_env()

            self.assertEqual(info["cpu_threads"], 4)
            self.assertEqual(info["cpu_interop_threads"], 1)
            for key in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            ]:
                self.assertEqual(os.environ[key], "4")
            self.assertEqual(os.environ["CVSRFFI_CPU_THREADS"], "4")
            self.assertEqual(os.environ["CVSRFFI_CPU_INTEROP_THREADS"], "1")

    def test_existing_thread_env_is_respected_by_default(self):
        from cvsrffi.runtime_threads import configure_cpu_thread_env

        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "6",
            },
            clear=True,
        ):
            info = configure_cpu_thread_env()

            self.assertEqual(info["cpu_threads"], 8)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "8")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "6")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "8")

    def test_explicit_thread_limit_forces_all_cpu_libraries(self):
        from cvsrffi.runtime_threads import configure_cpu_thread_env

        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "32",
                "MKL_NUM_THREADS": "32",
                "OPENBLAS_NUM_THREADS": "32",
            },
            clear=True,
        ):
            info = configure_cpu_thread_env(cpu_threads=2, cpu_interop_threads=1, force=True)

            self.assertEqual(info["cpu_threads"], 2)
            self.assertEqual(info["cpu_interop_threads"], 1)
            for key in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            ]:
                self.assertEqual(os.environ[key], "2")
            self.assertEqual(os.environ["CVSRFFI_CPU_THREADS"], "2")
            self.assertEqual(os.environ["CVSRFFI_CPU_INTEROP_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
