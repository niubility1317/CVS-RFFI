import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write
from experiments.pamf_unrolled_restore.wisig_practical_supervised import COUNTS, receipt
from experiments.pamf_unrolled_restore.wisig_practical_supervised_fit import (
    ARMS, FIT, HELD, REGIONS, PacketDenoiser, load_arrays, matched_target, objective,
)


class KnownNoiseSupervisionTests(unittest.TestCase):
    def test_direction_target_matches_two_regional_teacher_norms(self):
        received = np.full((2, 256), 1j, dtype=np.complex64)
        teacher = received + np.linspace(.001, 1., 256, dtype=np.float32)
        oracle = received + (1j * np.arange(1, 257, dtype=np.float32))
        oracle[1, REGIONS[1]] = received[1, REGIONS[1]]
        direction = matched_target(received, teacher, oracle)
        for region in REGIONS:
            edit = (direction-received)[0, region]
            target = (oracle-received)[0, region]
            np.testing.assert_allclose(np.linalg.norm(edit), np.linalg.norm((teacher-received)[0, region]), rtol=1e-6)
            np.testing.assert_allclose(edit/np.linalg.norm(edit), target/np.linalg.norm(target), rtol=5e-6, atol=1e-7)
        np.testing.assert_array_equal(direction[1, REGIONS[1]], received[1, REGIONS[1]])

    def test_model_keeps_native_iq_shape_and_objective_trains(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        model = PacketDenoiser()
        received, target, clean = (torch.randn(4, 2, 256) for _ in range(3))
        self.assertEqual(model(received).shape, received.shape)
        objective(model, received, target, clean).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_full_receipt_scores_each_arm_on_clean_as_well(self):
        views = ('clean', 'practical_low_urban')
        complete = receipt(views, 'cross_rx')
        self.assertEqual(complete['route'], 'full')
        self.assertFalse(complete['truth_read'])
        self.assertEqual(complete['inputs'], 'received_iq_only')
        self.assertEqual(complete['active']['clean'], {a: COUNTS['cross_rx'] for a in ARMS})

    def test_fitting_rejects_nonisolated_source_bank_before_opening_arrays(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'complete.json', dict(status='SOURCE_READY', fit=FIT, held=HELD, source_role='U_s',
                                            trace_scope='source_only', target_reads=1, TX_reads=0, F_reads=0))
            with self.assertRaisesRegex(ValueError, 'isolated source bank'):
                load_arrays(root, ARMS[0], 'cpu')


if __name__ == '__main__':
    unittest.main()
