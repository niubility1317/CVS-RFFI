import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_lora_main.lora_r69_w24_source_control import restore_packets, score


class LoRaR69SourceControlTest(unittest.TestCase):
    def test_smoke_forces_real_model_path_even_if_gate_would_bypass(self):
        class ClosedGate:
            def impairment(self, iq):
                return torch.zeros(len(iq))

        packets = np.ones((2, 8192), dtype=np.complex64)
        with patch('experiments.pamf_lora_main.lora_r69_w24_source_control.DEVICE', 'cpu'), patch(
                'experiments.pamf_lora_main.packet_physics.chirp_samples',
                return_value=np.ones((6, 256), dtype=np.complex64)), patch(
                'experiments.pamf_lora_main.full_dura_r69_lora.restored',
                side_effect=lambda gate, model, iq: (iq, iq)) as infer, patch(
                'experiments.pamf_lora_main.w24_native_full.project_delta',
                side_effect=lambda source, delta, scale: source):
            _, activated = restore_packets(packets, ClosedGate(), object(), force=True)
        self.assertEqual(activated, 2)
        infer.assert_called_once()

    def test_scoring_requires_cpu_and_complete_frozen_predictions(self):
        rt = {'project': 'unused', 'run_root': 'unused'}
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'Independent CPU scorer'):
                score(rt)
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), patch(
                'experiments.pamf_lora_main.lora_r69_w24_source_control.read',
                return_value=dict(status='RUNNING', modes=['direct', 'real_gain', 'lora_r69_half'],
                                  views=['clean', 'practical_low_urban'], count=60,
                                  source_truth_reads=0, target_reads=0)):
            with self.assertRaisesRegex(ValueError, 'freeze before source truth'):
                score(rt)


if __name__ == '__main__':
    unittest.main()
