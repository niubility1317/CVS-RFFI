import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_confidence_lora_retry_run as run


class ConfidenceLoRaRetryTest(unittest.TestCase):
    def test_experts_activate_before_teacher_and_retry_is_lora_only(self):
        order = []
        def experts(_rt, _dataset):
            order.append('experts')
            return object()
        def teacher(_rt, _dataset):
            self.assertEqual(order, ['experts'])
            return object()
        with patch.object(run, 'read', return_value={}), \
             patch.object(run.shared, 'experts', side_effect=experts), \
             patch.object(run, 'source_teacher', side_effect=teacher), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict'), patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            run.worker('runtime.json', 'lora')
            with self.assertRaises(ValueError):
                run.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
