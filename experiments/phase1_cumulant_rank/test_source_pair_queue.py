import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.phase1_cumulant_rank import source_pair_queue as queue

class QueueResourceTest(unittest.TestCase):
    def test_same_project_worker_waits(self):
        rows=['4, GPU-TEST','GPU-TEST, 123']
        command=('python --root '+str(Path('/project/root'))+' --gpu 4').encode()
        with patch.object(queue,'gpu_available',return_value=(True,19000,1)), \
             patch.object(queue.subprocess,'check_output',side_effect=rows), \
             patch.object(Path,'read_bytes',return_value=command), \
             patch.object(queue.time,'sleep',side_effect=RuntimeError('waited')):
            with self.assertRaisesRegex(RuntimeError,'waited'):queue.wait_for_project_slot(Path('/project/root'),4)
    def test_other_owner_can_share_with_capacity(self):
        rows=['4, GPU-TEST','GPU-TEST, 123']
        with patch.object(queue,'gpu_available',return_value=(True,19000,1)), \
             patch.object(queue.subprocess,'check_output',side_effect=rows), \
             patch.object(Path,'read_bytes',return_value=b'other owner command'):
            queue.wait_for_project_slot(Path('/project/root'),4)

if __name__=='__main__':unittest.main()
