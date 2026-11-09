import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.phase1_cumulant_true_response import train, ground


class SourceAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.n = 64
        rng = np.random.default_rng(17)
        self.received = rng.normal(size=(self.n, 2, 256)).astype(np.float32)
        self.clean = rng.normal(size=(self.n, 2, 256)).astype(np.float32)
        self.kernel = np.zeros((self.n, 2, 64), np.float32)
        self.coef = np.zeros((self.n, 4, 256), np.float32)
        self.roles = np.array(['L_s']*32 + ['U_s']*32)
        self.fit = np.arange(self.n)
        self.values = (torch.tensor(self.received.tolist(), dtype=torch.float32), None, None, None, torch.tensor(self.clean.tolist(), dtype=torch.float32))

    def mock_load(self, path, **kwargs):
        return {'roles.npy': self.roles, 'received.npy': self.received,
                'kernel.npy': self.kernel, 'coefficients.npy': self.coef}[Path(path).name]

    def metadata(self, path):
        return dict(count=self.n, namespace='learned-action/train/epoch21',
                    TX_labels_read=0, F_outputs_read=0, target_reads=0)

    def exercise(self, values):
        class Dura:
            def estimate_context(self, x):
                return None, object(), None
        def delta(clean, start):
            return torch.complex(clean[:, 0], clean[:, 1]) * .01
        def context(dura, changed):
            return changed, torch.zeros_like(changed), torch.ones(len(changed))
        with patch.object(train, 'source_inputs', return_value=(None,)*6),              patch.object(train, 'load_source_training', return_value=(Path('/source'), Path('/fresh'), self.clean, self.fit)),              patch.object(train.np, 'load', side_effect=self.mock_load),              patch.object(train.core, 'COUNT', self.n),              patch.object(train.core, 'read', side_effect=self.metadata),              patch.object(train.core, 'tensor', side_effect=lambda x, device: torch.tensor(np.array(x).tolist(), dtype=torch.float32, device=device)),              patch.object(train, 'true_action', return_value=lambda x: 2*x),              patch.object(train, 'choose_perturbation', side_effect=delta),              patch.object(train, 'extract_context', side_effect=context):
            return train.prepare_true_counterfactual(Path('/root'), Dura(), values, self.fit, True)

    def test_matched_simulator_action_changes_only_training_view(self):
        changed, anchor, direction, variance, delta = self.exercise(self.values)
        torch.testing.assert_close(changed - self.values[0], 2*delta, rtol=1e-5, atol=2e-7)
        torch.testing.assert_close(anchor, changed)
        self.assertEqual(tuple(direction.shape), (64, 2, 256))
        self.assertEqual(tuple(variance.shape), (64,))

    def test_mismatched_observation_rejected(self):
        x = self.values[0].clone(); x[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'must match source physical'):
            self.exercise((x, None, None, None, self.values[4]))


class InferenceContractTest(unittest.TestCase):
    def test_simulator_action_at_inference_rejected(self):
        metadata = dict(schema=train.SCHEMA, commit='test', epochs=20, fit_ids=32768,
            actual_source_view_count=98304, views_per_record=3, width=train.WIDTH, layers=train.LAYERS,
            lags=list(train.LAGS), cumulant_weight=train.CUMULANT_WEIGHT,
            response_weight=train.RESPONSE_WEIGHT, response_epsilon=train.EPSILON,
            response_directions=list(train.DIRECTIONS), response_source_action='matched_source_simulator',
            simulator_action_inference_reads=1, TX_labels_read=0, F_weights_read=0,
            F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
        summary = dict(metadata, status='SOURCE_TRAINED')
        with patch.object(ground.core, 'read', return_value=summary),              patch.object(ground.torch, 'load', return_value=metadata):
            with self.assertRaisesRegex(ValueError, 'matching contract required'):
                ground.load_fixed(Path('/root'), 'source', 'test', 'cpu')


if __name__ == '__main__':
    unittest.main()
