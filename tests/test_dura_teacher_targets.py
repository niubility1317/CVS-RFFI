import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from experiments.cric_dura_comparison.teacher_targets import EpochTargets,SharedEpochTargets,produce_shared_epoch,exposure_count

torch.set_num_threads(2)
TEMP_ROOT=Path(__file__).resolve().parents[1]/'local_artifacts/dura_training_acceleration_20261022/tests'
TEMP_ROOT.mkdir(parents=True,exist_ok=True)


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.75));self.samples=0;self.calls=0
        self.eval().requires_grad_(False)
    def stage_outputs(self,x):
        self.samples+=len(x);self.calls+=1
        return [x*self.scale,x*self.scale+.1,x*self.scale+.2],None,None,None


class View:
    ids=np.asarray(['L0','L1','U0','U1','U2'])
    def __len__(self):return len(self.ids)
    def ready(self):return True
    def batch(self,ids,epoch):
        x=np.asarray(ids,np.float32)[:,None,None]+epoch/10.
        return dict(received=np.broadcast_to(x,(len(ids),2,256)).copy())


class TeacherReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys,experiments
        cls.original_paths=list(sys.path);cls.original_namespace=list(experiments.__path__)
        root=Path(__file__).resolve().parents[1]/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        roots={k:str(root/k) for k in ('F','AF','BASE','FOLD','G')}
        cls.have_sources=all(Path(v).is_dir() for v in roots.values())
        if cls.have_sources:
            from experiments.cric_dura_comparison.bootstrap import activate
            activate(dict(frozen_sources=roots))
    @classmethod
    def tearDownClass(cls):
        import sys,experiments
        sys.path[:]=cls.original_paths;experiments.__path__=cls.original_namespace
    def test_original_exposures_and_zero_repeat_case(self):
        from experiments.cric_dura_comparison.g_train import batches
        for n in (630,1080,3150,6300,12600,37800,63000,75600):
            plan=list(batches(n,np.random.default_rng(5)))
            self.assertEqual(sum(map(len,plan)),exposure_count(n));self.assertEqual(len(plan),247)
        self.assertEqual(exposure_count(63000),63000)

    def test_unique_input_once_and_exact_requested_order(self):
        t=Teacher();cache=EpochTargets(t,5,1,dict(teacher='fixed',source='L'),storage='cpu')
        for ids in ([3,1,3],[4,1,0],[0,2,4]):
            x=torch.tensor(ids,dtype=torch.float32)[:,None,None].expand(-1,2,256)
            got=cache.get(np.asarray(ids),x,1)
            expected=[x*.75,x*.75+.1,x*.75+.2]
            for a,b in zip(got,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(t.samples,5);self.assertEqual(cache.stats()['unique_cached'],5)
        self.assertTrue(all(not v.requires_grad for v in got))

    def test_epoch_role_index_and_memory_boundaries(self):
        t=Teacher();cache=EpochTargets(t,5,1,{'roles':'L'},max_mib=.001)
        x=torch.zeros(1,2,256)
        with self.assertRaisesRegex(ValueError,'epochs'):cache.get(np.asarray([0]),x,2)
        with self.assertRaisesRegex(ValueError,'source index'):cache.get(np.asarray([-1]),x)
        with self.assertRaisesRegex(ValueError,'source indices'):cache.get(np.asarray([.5]),x)
        with self.assertRaisesRegex(ValueError,'memory budget'):cache.get(np.asarray([0]),x)
        t.train()
        with self.assertRaisesRegex(ValueError,'eval-mode'):EpochTargets(t,5,1,{'roles':'L'}).get(np.asarray([0]),x)

    def test_teacher_does_not_consume_student_rng(self):
        t=Teacher()
        def predict(x):return (x+torch.rand_like(x),)
        cache=EpochTargets(t,1,1,{'source':'same'},predict=predict)
        before=torch.random.get_rng_state().clone()
        cache.get(np.asarray([0]),torch.zeros(1,2,256))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_same_budget_student_updates_and_gradients(self):
        t=Teacher();a=torch.nn.Conv1d(2,2,1);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=1e-3);ob=torch.optim.AdamW(b.parameters(),lr=1e-3)
        cache=EpochTargets(t,5,1,{'source':'same'})
        ids=np.asarray([3,1,4,1,0,2,3,4])
        for offset in range(0,len(ids),2):
            selected=ids[offset:offset+2];x=torch.tensor(selected.tolist(),dtype=torch.float32)[:,None,None].expand(-1,2,256)
            online=tuple(v.detach() for v in t.stage_outputs(x)[0]);cached=cache.get(selected,x)
            for model,opt,desired in ((a,oa,online),(b,ob,cached)):
                opt.zero_grad();values=model(x);loss=sum((values-v).square().mean() for v in desired)/len(desired)
                loss.backward();opt.step()
            for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        self.assertFalse(any(p.grad is not None for p in t.parameters()))

    def test_one_shared_producer_serves_different_student_orders(self):
        view=View();t=Teacher();identity=dict(teacher='R65',source={'roles':'LU'},input='epoch-fixed')
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            produce_shared_epoch(folder,identity,view,t,1,batch_size=2)
            self.assertEqual(t.samples,5)
            for ids in ([4,1,0,2,3],[3,2,1,0,4]):
                reader=SharedEpochTargets(folder,identity,view.ids,1)
                arrays=view.batch(np.asarray(ids),1);x=torch.from_numpy(arrays['received'])
                got=reader.get(np.asarray(ids),x,1)
                torch.testing.assert_close(got[0],x*.75+.1,rtol=0,atol=0)
                torch.testing.assert_close(got[1],x*.75+.2,rtol=0,atol=0)
            self.assertEqual(t.samples,5)
            with self.assertRaisesRegex(ValueError,'binding'):SharedEpochTargets(folder,dict(identity,teacher='OTHER'),view.ids,1)
            with self.assertRaisesRegex(ValueError,'physical IDs'):SharedEpochTargets(folder,identity,view.ids[::-1],1)
            with self.assertRaises(FileExistsError):produce_shared_epoch(folder,identity,view,t,1)
            (Path(folder)/'epoch02').mkdir()
            with self.assertRaises(FileNotFoundError):SharedEpochTargets(folder,identity,view.ids,2)

    def test_real_loss_alignment_and_zero_weight_fft_skip(self):
        if not self.have_sources:self.skipTest('Immutable G source checkout required for original spectral loss')
        from experiments.cric_dura_comparison.g_lineage import objective
        from experiments.pamf_r44_pruned_refine.models import spectral_error
        class Student(torch.nn.Module):
            def __init__(self,count):
                super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.6));self.count=count
            def stage_outputs(self,x):return [x*self.scale+i*.1 for i in range(self.count)],None,None,None
        x=torch.randn(3,2,256);clean=x*.9;t=Teacher()
        for stage,count in (('r38',3),('r44',3),('r46',3),('r65',3),('r69',2)):
            a=Student(count);b=copy.deepcopy(a)
            desired=tuple(t.stage_outputs(x)[0]);desired=desired[1:] if stage=='r69' else desired
            baseline=objective(a,t,stage,x,clean,7)
            with patch('experiments.pamf_r44_pruned_refine.models.spectral_error',wraps=spectral_error) as spec:
                cached=objective(b,t,stage,x,clean,7,desired=desired,skip_zero_spectral=True)
                self.assertEqual(spec.call_count,0 if stage in ('r38','r44') else 1)
            torch.testing.assert_close(baseline,cached,rtol=0,atol=0)
            baseline.mean().backward();cached.mean().backward()
            torch.testing.assert_close(a.scale.grad,b.scale.grad,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'Detached'):
            objective(a,t,'r69',x,clean,1,desired=[x.requires_grad_(),x])

    def test_ablation_shared_tail_and_no_KD_teacher_bypass(self):
        if not self.have_sources:self.skipTest('Immutable G source checkout required for original spectral loss')
        from experiments.cric_dura_comparison.g_ablation import loss
        class Student(torch.nn.Module):
            def __init__(self,count):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.6));self.count=count
            def stage_outputs(self,x):return [x*self.scale+i*.1 for i in range(self.count)],None,None,None
        x=torch.randn(3,2,256);clean=x*.9;t=Teacher()
        for arm,count in (('coarse_only',1),('random_with_KD',2),('copy_no_KD',2),('random_no_KD',2)):
            a=Student(count);b=copy.deepcopy(a);tail=tuple(t.stage_outputs(x)[0][-2:]);before=t.calls
            cached=loss(b,t,arm,x,clean,2,desired=tail)
            self.assertEqual(t.calls,before)
            online=loss(a,t,arm,x,clean,2)
            torch.testing.assert_close(cached,online,rtol=0,atol=0)
            cached.mean().backward();online.mean().backward()
            torch.testing.assert_close(a.scale.grad,b.scale.grad,rtol=0,atol=0)

    def test_producer_rejects_target_and_F_before_opening_inputs(self):
        from experiments.cric_dura_comparison.build_teacher_targets import run
        for key in ('truth','target_truth','target_inputs','tests','reader','F_checkpoint'):
            with self.assertRaisesRegex(ValueError,'rejects F/target'):run({key:'FORBIDDEN'})


if __name__=='__main__':unittest.main()
