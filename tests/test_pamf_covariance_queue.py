import os
import unittest
from experiments.pamf_u_history.covariance_queue import admissible,controller_identity,reconcile,expected_jobs


class QueueTests(unittest.TestCase):
    def test_shared_admission_counts_pending_and_preserves_one_own_slot(self):
        self.assertTrue(admissible(4,{10},{10:4},{},20000))
        self.assertFalse(admissible(4,{10},{10:4,11:4},{},20000))
        self.assertFalse(admissible(4,{10},{10:4},{'own':dict(gpu=4)},20000))
        self.assertFalse(admissible(4,{10},{10:4},{str(g):dict(gpu=g) for g in range(4)},20000))
        self.assertFalse(admissible(4,set(),{}, {},9999))

    def test_exact_original_controller_only(self):
        e=dict(pid=123,cwd='/release/original',argv=['python','-m','owned','dispatch'])
        self.assertTrue(controller_identity(e,123,e['cwd'],e['argv'],1000,1000))
        self.assertFalse(controller_identity(e,124,e['cwd'],e['argv'],1000,1000))
        self.assertFalse(controller_identity(e,123,'/another',e['argv'],1000,1000))
        self.assertFalse(controller_identity(e,123,e['cwd'],['python','other'],1000,1000))
        self.assertFalse(controller_identity(e,123,e['cwd'],e['argv'],2000,1000))

    def test_completed_outputs_reused_and_live_workers_adopted(self):
        jobs=[dict(id=k) for k in ('finished','live','queued')];old={k:dict(status='QUEUED') for k in ('finished','live','queued')};live={'live':dict(pid=7,status='RUNNING',gpu=4)}
        states,active,stop=reconcile(jobs,old,live,lambda j:j['id']=='finished',lambda j:False)
        self.assertEqual(states['finished']['status'],'COMPLETE');self.assertEqual(active,live);self.assertFalse(stop)
        with self.assertRaises(ValueError):reconcile(jobs,old,live,lambda j:j['id']=='finished',lambda j:j['id']=='queued')
        old['queued']['status']='RUNNING';states,_,stop=reconcile(jobs,old,live,lambda j:j['id']=='finished',lambda j:False)
        self.assertTrue(stop);self.assertEqual(states['queued']['status'],'FAILED')
        self.assertEqual(len(expected_jobs()),28)


if __name__=='__main__':unittest.main()
