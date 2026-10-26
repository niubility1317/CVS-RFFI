import copy
import unittest
import torch
from torch import nn
from experiments.pamf_u_history.covariance_fit import RUN,RELEASE,RECIPE,validate,capture


class FitBoundaryTest(unittest.TestCase):
    def config(self):
        return dict(run_id=RUN,recipe=copy.deepcopy(RECIPE),project='/source/project',
            run_root='/source/project/runs/'+RUN,release_root='/source/project/releases/'+RELEASE,allowed_gpus=list(range(8)))

    def test_no_query_or_unlabeled_mask_fitting(self):
        c=self.config();validate(c);c['truth']='target'
        with self.assertRaises(ValueError):validate(c)
        c=self.config();c['recipe']['source_role']='U_s'
        with self.assertRaises(ValueError):validate(c)

    def test_output_must_be_owned_immutable_root(self):
        c=self.config();c['run_root']='/source/project/runs/another-owner'
        with self.assertRaises(ValueError):validate(c)

    def test_two_actual_temporal_maps_and_hooks_are_removed(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__();self.core=nn.Module();self.core.id_backbone=nn.Module()
                b=self.core.id_backbone;b.time_down=nn.Conv1d(3,4,1);b.t2=nn.Module();b.t2.pool=nn.AvgPool1d(2)
            def features(self,x):
                b=self.core.id_backbone;return b.t2.pool(b.time_down(x)).mean(-1)
        m=Net();z,corr,shapes=capture(m,torch.randn(2,3,8))
        self.assertEqual(shapes,{'time_down':[4,8],'t2.pool':[4,4]})
        self.assertEqual(z.shape,(2,4));self.assertEqual(corr['t2.pool'].shape,(2,4,4))
        self.assertTrue(all(len(layer._forward_hooks)==0 for layer in m.modules()))


if __name__=='__main__':unittest.main()
