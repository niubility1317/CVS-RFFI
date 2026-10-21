import unittest
from unittest.mock import patch,MagicMock
from experiments.cric_dura_comparison.multiprocess_control import gpu_admits,command_for,inspect_worker,external_pending


class ResourceOnlyHandover(unittest.TestCase):
    def test_second_own_F_allowed_but_not_third(self):
        jobs={'a':dict(kind='F'),'b':dict(kind='F')};active={'a':dict(gpu=0)}
        self.assertTrue(gpu_admits(0,dict(kind='F'),active,jobs,1,18000))
        active['b']=dict(gpu=0)
        self.assertFalse(gpu_admits(0,dict(kind='F'),active,jobs,2,18000))
    def test_existing_foreign_capacity_memory_and_heavy_peak(self):
        jobs={'a':dict(kind='G')};active={'a':dict(gpu=4)}
        self.assertTrue(gpu_admits(4,dict(kind='F'),active,jobs,2,12000))
        self.assertFalse(gpu_admits(4,dict(kind='F'),active,jobs,3,18000))
        self.assertFalse(gpu_admits(4,dict(kind='F'),active,jobs,1,11999))
        self.assertFalse(gpu_admits(4,dict(kind='G_student'),active,jobs,1,18000))
    def test_external_queue_capacity_reserved_until_complete(self):
        jobs={'a':dict(kind='F')};active={'a':dict(gpu=2)}
        self.assertFalse(gpu_admits(2,dict(kind='F'),active,jobs,1,18000,True))
        self.assertTrue(gpu_admits(2,dict(kind='F'),active,jobs,1,18000,False))
    def test_timing_cannot_be_joined_or_started_shared(self):
        jobs={'a':dict(kind='timing',exclusive=True)};active={'a':dict(gpu=7)}
        self.assertFalse(gpu_admits(7,dict(kind='F'),active,jobs,1,18000))
        self.assertFalse(gpu_admits(0,dict(kind='timing',exclusive=True),{},jobs,1,18000))
        self.assertTrue(gpu_admits(0,dict(kind='timing',exclusive=True),{},jobs,0,18000))
        self.assertFalse(gpu_admits(2,dict(kind='timing',exclusive=True),{},jobs,0,18000))
    def test_scientific_command_and_release_remain_immutable(self):
        rt=dict(python='/CVS/python',worker_release='/release/science3e880')
        job=dict(kind='F',config='/original/row.json')
        cmd,cwd,env=command_for(rt,job,4)
        self.assertEqual(cmd,['/CVS/python','-u','-m','experiments.cric_dura_comparison.pipeline','worker','--config','/original/row.json'])
        self.assertEqual(cwd,'/release/science3e880');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4')
        self.assertEqual(command_for(rt,job,None)[2]['CUDA_VISIBLE_DEVICES'],'')
    def test_worker_finishing_during_identity_check_is_not_a_mismatch(self):
        proc=MagicMock();proc.joinpath.return_value.read_bytes.return_value=b''
        with patch('experiments.cric_dura_comparison.multiprocess_control.Path') as path,patch('experiments.cric_dura_comparison.multiprocess_control.alive',side_effect=[True,False]):
            path.return_value.__truediv__.return_value=proc
            self.assertIsNone(inspect_worker({},dict(id='finished',kind='F'),123))
    def test_external_state_rewrite_reserves_its_slots(self):
        with patch('experiments.cric_dura_comparison.multiprocess_control.Path') as path,patch('experiments.cric_dura_comparison.multiprocess_control.read',side_effect=ValueError('in-progress state write')):
            path.return_value.is_file.return_value=True
            self.assertTrue(external_pending(dict(external_pipeline_state='external.json')))

if __name__=='__main__':unittest.main()
