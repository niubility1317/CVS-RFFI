import unittest
from experiments.cric_dura_comparison import multiprocess_control as mc


class LoadBalancedAdmission(unittest.TestCase):
    def policy(self, gpu_util=70):
        cls = getattr(mc, 'LoadPolicy', None)
        self.assertIsNotNone(cls, 'Load-aware admission is not implemented')
        p = cls(dict(boost_gpus=[0, 6], max_gpu_workers=26))
        host = dict(cpu_busy_pct=54, available_ram_gib=350, iowait_pct=0.1)
        for now in (0, 30, 60, 90, 120):
            p.observe(host, {g:dict(util=gpu_util, total_mib=24576) for g in range(8)}, now)
        return p

    def owned(self, count=3, gpu=0, kind='F'):
        jobs = {str(i):dict(kind=kind) for i in range(count)}
        active = {key:dict(gpu=gpu, pid=100+i) for i,key in enumerate(jobs)}
        return active, jobs

    def test_fourth_F_only_on_approved_low_load_cards(self):
        p = self.policy()
        for gpu, want in ((0, True), (6, True), (1, False), (7, False)):
            active,jobs = self.owned(gpu=gpu)
            self.assertEqual(p.admits(gpu,dict(kind='F'),active,jobs,3,14000,False), want)

    def test_fourth_rejects_G_prediction_and_existing_heavy_worker(self):
        p=self.policy();active,jobs=self.owned()
        self.assertFalse(p.admits(0,dict(kind='G'),active,jobs,3,14000,False))
        self.assertFalse(p.admits(0,dict(kind='predict'),active,jobs,3,14000,False))
        jobs['0']['kind']='G'
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,3,14000,False))

    def test_foreign_process_or_global_capacity_blocks_new_worker(self):
        p=self.policy();active,jobs=self.owned()
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,4,14000,False))
        for i in range(3,26):
            jobs[str(i)]=dict(kind='F');active[str(i)]=dict(gpu=1,pid=100+i)
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,3,14000,False))

    def test_preCUDA_owned_workers_do_not_hide_foreign_GPU_processes(self):
        p=self.policy();active={};jobs={}
        for gpu in range(8):
            p.devices[gpu]['cuda_pids']=[]
            for slot in range(3):
                pid=100+gpu*3+slot;key=str(pid)
                active[key]=dict(gpu=gpu,pid=pid);jobs[key]=dict(kind='F')
                if gpu not in (6,7) or slot<2:p.devices[gpu]['cuda_pids'].append(pid)
            if gpu in (6,7):p.devices[gpu]['cuda_pids'].append(900+gpu)
            p.devices[gpu]['apps']=len(p.devices[gpu]['cuda_pids'])
        # 24 owned PIDs plus 2 foreign PIDs already consume all 26 slots.
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,3,14000,False))

    def test_cpu_RAM_IO_pressure_blocks_all_new_GPU_launches(self):
        for field,value in (('cpu_busy_pct',75),('available_ram_gib',63),('iowait_pct',5)):
            p=self.policy();p.host[field]=value;active,jobs=self.owned(1)
            self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,1,18000,False))

    def test_high_GPU_load_blocks_fourth_but_does_not_kill_or_block_baseline(self):
        p=self.policy(90);active,jobs=self.owned()
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,3,14000,False))
        active.pop('2')
        self.assertTrue(p.admits(0,dict(kind='F'),active,jobs,2,14000,False))
        self.assertEqual(len(active),2)

    def test_warmup_missing_or_stale_telemetry_cannot_admit_fourth(self):
        p=self.policy();active,jobs=self.owned()
        p.history={}
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,3,14000,False))
        p.host={}
        self.assertFalse(p.admits(0,dict(kind='F'),{},jobs,0,24000,False))

    def test_memory_reservation_accounts_for_future_heavy_stage_peak(self):
        p=self.policy();active,jobs=self.owned()
        self.assertFalse(p.admits(0,dict(kind='G'),active,jobs,3,16000,False))
        jobs['0']['kind']='G';active.pop('2')
        self.assertTrue(p.admits(0,dict(kind='F'),active,jobs,2,14000,False))
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,2,11999,False))

    def test_timing_exclusive_and_other_owner_reservation_remain(self):
        p=self.policy();active,jobs=self.owned(1,gpu=2)
        self.assertFalse(p.admits(2,dict(kind='F'),active,jobs,1,18000,True))
        self.assertFalse(p.admits(0,dict(kind='timing',exclusive=True),active,jobs,1,18000,False))
        self.assertTrue(p.admits(0,dict(kind='timing',exclusive=True),{},jobs,0,24000,False))

    def test_lower_load_gpu_is_ranked_first(self):
        p=self.policy();p.history[6]=[(0,60),(120,60)];p.history[0]=[(0,80),(120,80)]
        self.assertEqual(p.order([0,6])[0],6)

    def test_no_throughput_proxy_gain_disables_only_future_fourth_launches(self):
        p=self.policy();p.evaluate_slowdown(0,[1.4,1.5,1.6]);active,jobs=self.owned()
        self.assertFalse(p.admits(0,dict(kind='F'),active,jobs,3,14000,False))
        active.pop('2')
        self.assertTrue(p.admits(0,dict(kind='F'),active,jobs,2,14000,False))
        self.assertIn(0,p.disabled)


if __name__=='__main__':unittest.main()
