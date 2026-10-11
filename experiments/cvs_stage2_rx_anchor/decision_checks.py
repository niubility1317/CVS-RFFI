import unittest
from copy import deepcopy
import torch
from torch.nn import functional as F
from .decision_audit import centered_head,basis,decision_geometry,pair_decisions,target_margins
from experiments.cvs_stage2_stable_rx.realized_audit import geometry


class DecisionChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def sample(self):
        head=torch.eye(6,160,dtype=torch.float64)
        p=torch.zeros(2,160,dtype=torch.float64);p[0,0]=.2;p[0,1]=-.2;p[0,20]=.3;p[1]=-p[0]
        bank=dict(groups=[dict(potential=p[0],tx=0,day=1,rx=1,block_id=0),dict(potential=p[1],tx=0,day=1,rx=2,block_id=0)],
            blocks=[dict(group_ids=[0,1],fold=0,alpha=.25,block_id=0,tx=0,day=1)],folds=[dict(c=.4)])
        return head,p,bank

    def test_head_basis_preserves_logit_differences(self):
        head,p,bank=self.sample();q,info=basis(head);self.assertEqual(info['rank'],5)
        z=torch.randn(13,160,dtype=torch.float64);w=centered_head(head)
        torch.testing.assert_close(z@w.T,z@q@q.T@w.T)

    def test_energy_decomposition_and_population_risk_match(self):
        h,p,b=self.sample();u=-.1*p;zero=torch.zeros_like(p)
        r=decision_geometry(u,zero,b,h)
        for key in ('target','update','alignment','error'):
            self.assertAlmostEqual(r['energies']['full'][key],r['energies']['visible'][key]+r['energies']['remaining'][key])
        self.assertAlmostEqual(.5*r['energies']['full']['error'],geometry(u,zero,b)['summary']['population_risk'])
        self.assertAlmostEqual(r['target_visible_fraction'],.08/.17)

    def test_joint_rotation_invariance(self):
        h,p,b=self.sample();u=-.1*p;zero=torch.zeros_like(p);g=torch.Generator().manual_seed(15)
        rotation=torch.linalg.qr(torch.randn(160,160,dtype=torch.float64,generator=g))[0]
        altered=deepcopy(b)
        for group in altered['groups']:group['potential']=group['potential']@rotation
        a=decision_geometry(u,zero,b,h);c=decision_geometry(u@rotation,zero,altered,h@rotation)
        for scope in ('full','visible','remaining'):
            for key in ('target','update','alignment','error'):self.assertAlmostEqual(a['energies'][scope][key],c['energies'][scope][key])

    def test_rank_zero_and_no_target_signal(self):
        h,p,b=self.sample();q,info=basis(torch.ones_like(h));self.assertEqual(info['rank'],0)
        b['blocks'][0]['alpha']=0
        r=decision_geometry(p,p,b,h);self.assertIsNone(r['target_visible_fraction'])
        self.assertIsNone(r['energies']['visible']['progress'])

    def test_symmetric_features_and_head_decomposition(self):
        h,p,b=self.sample();g=torch.Generator().manual_seed(33);z=F.normalize(torch.randn(20,160,dtype=torch.float64,generator=g),dim=1);y=torch.arange(20)%6
        same=pair_decisions(z,h,z,h,z,h,y)
        self.assertEqual(same['fixed_parent_rival']['total']['rms'],0)
        changed=pair_decisions(z,h.roll(1,0),z,h,z,h,y)
        self.assertEqual(changed['fixed_parent_rival']['feature']['rms'],0)
        self.assertGreater(changed['fixed_parent_rival']['head']['rms'],0)
        self.assertAlmostEqual(changed['fixed_parent_rival']['total']['mean'],changed['fixed_parent_rival']['head']['mean'])

    def test_target_margin_and_missing_cell_rejection(self):
        h,p,b=self.sample();z=h[[0,0]].clone();cache=dict(y=torch.tensor([0,0]),day=torch.tensor([1,1]),rx=torch.tensor([1,2]))
        result=target_margins(z,h,cache,b)
        self.assertAlmostEqual(result['linear_target_margin_shift']['mean'],0)
        self.assertAlmostEqual(result['linear_target_margin_shift']['rms'],3.)
        bad=deepcopy(cache);bad['rx'][1]=3
        with self.assertRaises(ValueError):target_margins(z,h,bad,b)


if __name__=='__main__':unittest.main()
