"""Analytic population-risk checks, no production data or checkpoint access."""
import unittest
from copy import deepcopy
import torch
from torch.nn import functional as F
from .realized_audit import geometry,group_means,scores,validate_complete_cells
from experiments.cvs_stage2_geometry.receiver import finite_population_loss


class RealizedChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def example(self):
        p=torch.zeros(2,160,dtype=torch.float64);p[0,0]=.2;p[1,0]=-.2
        bank=dict(groups=[dict(potential=p[0],tx=0,day=1,rx=1),dict(potential=p[1],tx=0,day=1,rx=2)],
            blocks=[dict(group_ids=[0,1],fold=0,alpha=.25,block_id=0,tx=0,day=1)],folds=[dict(c=.3)])
        return bank,p,-.25*p

    def test_parent_perfect_and_signed_progress(self):
        bank,p,t=self.example();zero=torch.zeros_like(p)
        for scale,ratio in [(0,1),(.5,.25),(1,0),(-1,4),(2,1)]:
            result=geometry(scale*t,zero,bank)['summary']
            self.assertAlmostEqual(result['progress_coefficient'],scale)
            self.assertAlmostEqual(result['error_over_parent'],ratio)
            self.assertAlmostEqual(result['population_risk'],result['anchor_term']+result['cross_term']+result['constant_term'])

    def test_off_direction_drift_is_measured(self):
        bank,p,t=self.example();drift=torch.zeros_like(p);drift[:,1]=t[:,0]
        result=geometry(t+drift,torch.zeros_like(t),bank)['summary']
        self.assertAlmostEqual(result['progress_coefficient'],1.)
        self.assertAlmostEqual(result['off_target_energy'],result['target_energy'])
        self.assertAlmostEqual(result['error_over_parent'],1.)

    def test_zero_trust_has_no_artificial_progress(self):
        bank,p,t=self.example();bank['folds'][0]['c']=0
        result=geometry(t,torch.zeros_like(t),bank)['summary']
        self.assertIsNone(result['progress_coefficient']);self.assertTrue(result['zero_signal'])
        self.assertEqual(result['population_risk'],0.)

    def test_equals_full_population_training_estimator(self):
        bank,p,t=self.example();g=torch.Generator().manual_seed(83)
        q0=torch.randn(8,160,generator=g,dtype=torch.float64)*.01
        q1=torch.randn(12,160,generator=g,dtype=torch.float64)*.01
        risk,_=finite_population_loss(q0,q1,p[0]-p[1],alpha=.25,c=.3,population_g=8,population_h=12)
        result=geometry(torch.stack((q0.mean(0),q1.mean(0))),torch.zeros(2,160),bank)['summary']
        self.assertAlmostEqual(float(risk),result['population_risk'],places=14)

    def test_no_pooled_weighting_across_blocks(self):
        bank,p,t=self.example();other=deepcopy(bank);other['groups']+=deepcopy(bank['groups'])
        other['blocks'].append(dict(group_ids=[2,3],fold=1,alpha=.25,block_id=1,tx=1,day=1));other['folds'].append(dict(c=.9))
        target=torch.cat((t,t));means=torch.cat((t,torch.zeros_like(t)))
        result=geometry(means,torch.zeros_like(means),other)['summary']
        self.assertAlmostEqual(result['progress_coefficient'],.25)
        self.assertAlmostEqual(result['error_over_parent'],.75)

    def test_cell_assignment_and_missing_packet_rejected(self):
        bank,p,t=self.example();z=F.normalize(torch.arange(4*160,dtype=torch.float64).reshape(4,160)+1,dim=1)
        cache=dict(ids=list('abcd'),y=torch.zeros(4,dtype=torch.long),day=torch.ones(4,dtype=torch.long),rx=torch.tensor([1,2,1,2]))
        means,counts=group_means(z,cache,bank);self.assertEqual(counts,[2,2])
        torch.testing.assert_close(means[0],z[[0,2]].mean(0))
        bad=deepcopy(bank);bad['groups'].append(deepcopy(bad['groups'][0]))
        with self.assertRaises(ValueError):group_means(z,cache,bad)

    def test_classifier_swap_reproduces_identity_scores(self):
        z=torch.eye(6,160,dtype=torch.float64);y=torch.arange(6);rx=torch.tensor([1,1,1,2,2,2])
        same=scores(z,z,y,rx);flipped=scores(z,z.flip(0),y,rx)
        self.assertEqual(same['errors'],0);self.assertEqual(same['margin_mean'],30)
        self.assertEqual(flipped['errors'],6);self.assertEqual(flipped['margin_mean'],-30)

    def test_complete_cells_allow_nonzero_quality_label(self):
        keys=[(y,day,rx) for y in range(6) for day in (1,2,3) for rx in (1,3,4,6,8)]
        metadata=torch.tensor([k for k in keys for _ in range(70)])
        cache=dict(ids=list(range(6300)),y=metadata[:,0],day=metadata[:,1],rx=metadata[:,2])
        bank=dict(groups=[dict(tx=y,day=day,rx=rx,quality_bins=2,quality_bin=1,
            indices=list(range(i*70,(i+1)*70))) for i,(y,day,rx) in enumerate(keys)],blocks=[{}]*18)
        validate_complete_cells(bank,cache)
        wrong=deepcopy(bank);wrong['groups'][0]['indices'][0]=70
        with self.assertRaises(ValueError):validate_complete_cells(wrong,cache)
        wrong=deepcopy(bank);wrong['groups'][1].update(tx=0,day=1,rx=1)
        with self.assertRaises(ValueError):validate_complete_cells(wrong,cache)


if __name__=='__main__':unittest.main()
