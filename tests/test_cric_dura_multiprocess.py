import unittest
from unittest.mock import patch,MagicMock
from experiments.cric_dura_comparison.multiprocess_control import gpu_admits,command_for,inspect_worker,external_pending,handover


class ResourceOnlyHandover(unittest.TestCase):
    def test_third_own_F_allowed_but_not_fourth(self):
        jobs={'a':dict(kind='F'),'b':dict(kind='F'),'c':dict(kind='F')};active={'a':dict(gpu=0)}
        self.assertTrue(gpu_admits(0,dict(kind='F'),active,jobs,1,18000))
        active['b']=dict(gpu=0)
        self.assertTrue(gpu_admits(0,dict(kind='F'),active,jobs,2,18000))
        active['c']=dict(gpu=0)
        self.assertFalse(gpu_admits(0,dict(kind='F'),active,jobs,3,18000))
        self.assertFalse(gpu_admits(0,dict(kind='F'),{'a':dict(gpu=0),'b':dict(gpu=0)},jobs,2,18000,False,2,3))
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
    def test_resource_controller_can_be_replaced_without_changing_worker_release(self):
        import signal
        rt=dict(previous_state='state',python='/CVS/python',previous_runtime='/old/runtime.json',
                previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
                previous_controller_cwd='/old/control',worker_release='/original/science')
        argv=[rt['python'],'-u','-m',rt['previous_controller_module'],'--config',rt['previous_runtime']]
        with patch('experiments.cric_dura_comparison.multiprocess_control.read',return_value=dict(pid=123)),patch('experiments.cric_dura_comparison.multiprocess_control.Path') as path,patch('experiments.cric_dura_comparison.multiprocess_control.alive',side_effect=[True,False,False]),patch('experiments.cric_dura_comparison.multiprocess_control.os.readlink',return_value='/old/control'),patch('experiments.cric_dura_comparison.multiprocess_control.os.kill') as kill:
            path.return_value.__truediv__.return_value.joinpath.return_value.read_bytes.return_value=b'\0'.join(v.encode() for v in argv)+b'\0'
            path.return_value.iterdir.return_value=[]
            previous,active=handover(rt,[])
            kill.assert_called_once_with(123,signal.SIGTERM)
            self.assertEqual(previous['pid'],123);self.assertEqual(active,{})
    def test_wrong_controller_cwd_never_receives_signal(self):
        rt=dict(previous_state='state',python='/CVS/python',previous_runtime='/old/runtime.json',
                previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
                previous_controller_cwd='/old/control',worker_release='/original/science')
        argv=[rt['python'],'-u','-m',rt['previous_controller_module'],'--config',rt['previous_runtime']]
        with patch('experiments.cric_dura_comparison.multiprocess_control.read',return_value=dict(pid=123)),patch('experiments.cric_dura_comparison.multiprocess_control.Path') as path,patch('experiments.cric_dura_comparison.multiprocess_control.alive',return_value=True),patch('experiments.cric_dura_comparison.multiprocess_control.os.readlink',return_value='/unrelated'),patch('experiments.cric_dura_comparison.multiprocess_control.os.kill') as kill:
            path.return_value.__truediv__.return_value.joinpath.return_value.read_bytes.return_value=b'\0'.join(v.encode() for v in argv)+b'\0'
            with self.assertRaisesRegex(ValueError,'identity differs'):handover(rt,[])
            kill.assert_not_called()

if __name__=='__main__':unittest.main()
