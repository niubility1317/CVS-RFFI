import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_lora_main import shared_practical_mse_run as run


class IdentityAction:
    def __call__(self, value):
        return value

    def adjoint(self, value):
        return value


class PracticalMseTest(unittest.TestCase):
    def test_real_training_step_never_constructs_teacher(self):
        rng = np.random.default_rng(17)
        clean = (rng.normal(size=(32, 256)) + 1j*rng.normal(size=(32, 256))).astype(np.complex64)
        noisy = clean + .2*(rng.normal(size=clean.shape) + 1j*rng.normal(size=clean.shape)).astype(np.complex64)
        action = IdentityAction()
        def r17(value):
            return value, action
        def experts(_rt, _dataset):
            return clean, np.arange(len(clean)), r17, lambda value: .7*value
        def pool(_source, _ids, _dataset, smoke):
            smoke(clean[:16], noisy[:16])
            return clean, noisy, {'clean': 0, 'test': len(clean)}
        def as_complex(value):
            return torch.tensor(value.tolist(), dtype=torch.complex64)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'wisig').mkdir()
            with patch.object(run, 'DEVICE', 'cpu'), patch.object(run, 'STEPS', 1), \
                 patch.object(run.shared, 'experts', side_effect=experts), \
                 patch.object(run, 'practical_pool', side_effect=pool), \
                 patch.object(run, 'as_complex', side_effect=as_complex), \
                 patch.object(run, 'source_teacher', side_effect=AssertionError('teacher read')), \
                 patch('torch.cuda.synchronize'), patch('torch.cuda.max_memory_allocated', return_value=0):
                run.fit({'run_root': directory}, 'wisig')
            checkpoint = torch.load(Path(directory) / 'wisig/gain.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['proof']['F_reads'], 0)
            self.assertEqual(checkpoint['proof']['TX_reads'], 0)
            self.assertEqual(checkpoint['proof']['target_reads'], 0)


if __name__ == '__main__':
    unittest.main()
