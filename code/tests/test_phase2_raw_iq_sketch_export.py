import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2RawIqSketchExportTest(unittest.TestCase):
    def test_fft_logmag_sketch_is_phase_rotation_invariant_and_normalized(self):
        from phase2_raw_iq_sketch_export import _sketch_batch_with_method

        rng = np.random.default_rng(607)
        raw = rng.normal(size=(3, 2, 128)).astype(np.float32)
        theta = 1.37
        complex_raw = raw[:, 0] + 1j * raw[:, 1]
        rotated = complex_raw * np.exp(1j * theta)
        rotated_iq = np.stack([rotated.real, rotated.imag], axis=1).astype(np.float32)

        sketch = _sketch_batch_with_method(raw, dim=32, method="fft_logmag", projection=None)
        rotated_sketch = _sketch_batch_with_method(rotated_iq, dim=32, method="fft_logmag", projection=None)

        self.assertEqual(sketch.shape, (3, 32))
        np.testing.assert_allclose(np.linalg.norm(sketch, axis=1), np.ones(3), atol=1e-5)
        np.testing.assert_allclose(sketch, rotated_sketch, atol=1e-5)

    def test_random_projection_requires_projection_and_records_existing_shape(self):
        from phase2_raw_iq_sketch_export import _projection_matrix, _sketch_batch_with_method

        raw = np.arange(2 * 2 * 16, dtype=np.float32).reshape(2, 2, 16)
        projection = _projection_matrix(32, 8, 11)

        sketch = _sketch_batch_with_method(raw, dim=8, method="random_projection", projection=projection)

        self.assertEqual(sketch.shape, (2, 8))
        np.testing.assert_allclose(np.linalg.norm(sketch, axis=1), np.ones(2), atol=1e-5)
        with self.assertRaises(ValueError):
            _sketch_batch_with_method(raw, dim=8, method="random_projection", projection=None)


if __name__ == "__main__":
    unittest.main()
