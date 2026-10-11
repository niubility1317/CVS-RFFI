import unittest
import torch
from .identity_screen import estimate_rx,intervention,performance,logits_from_features,analyze,identity_safe
from . import relation_checks

class IdentityScreenChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_estimator_excludes_held_day_and_nonfit_tx(self):
        means,groups,blocks=relation_checks.RelationChecks.cells()
        one=estimate_rx(means,groups,blocks,0,[2,3,4,5]);other=means.clone()
        for i,g in enumerate(groups):
            if g['day']==0 or g['tx']<2:other[i]+=1000*(g['rx']+1)
        two=estimate_rx(other,groups,blocks,0,[2,3,4,5])
        for rx in one:self.assertTrue(torch.equal(one[rx],two[rx]))

    def test_transform_uses_only_feature_and_source_environment(self):
        z=torch.nn.functional.normalize(torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64),dim=1)
        bank={1:{3:torch.tensor([.1,-.1],dtype=torch.float64)}}
        rx=torch.tensor([3,3]);day=torch.tensor([1,1])
        result=intervention(z,rx,day,bank,.25)
        self.assertTrue(torch.allclose(result.norm(dim=1),torch.ones(2,dtype=torch.float64)))
        self.assertTrue(torch.allclose(intervention(z,rx,day,bank,0),z))
        self.assertTrue(torch.equal(intervention(z.flip(0),rx,day,bank,.25),result.flip(0)))

    def test_score_counts_broken_and_repaired(self):
        parent=torch.zeros(3,6,dtype=torch.float64);current=parent.clone();labels=torch.tensor([0,1,2])
        parent[0,0]=2;parent[1,0]=2;parent[2,2]=2
        current[0,1]=2;current[1,1]=2;current[2,2]=2
        result=performance(current,labels,parent,torch.tensor([1,1,1]),torch.tensor([1,1,1]))
        self.assertEqual(result['overall']['broken'],1);self.assertEqual(result['overall']['repaired'],1)
        self.assertEqual(result['overall']['errors'],1)

    def test_cosine_head_is_scale_thirty(self):
        z=torch.eye(6,dtype=torch.float64);weight=3*z
        self.assertTrue(torch.equal(logits_from_features(z,weight),30*z))

    def test_wrong_head_rejected_before_validation_access(self):
        z=torch.eye(6,160,dtype=torch.float64);state=torch.zeros(6,40);state[:,32]=.5
        cache=dict(parent_feature=z,state=state,y=torch.arange(6),rx=torch.ones(6,dtype=torch.long),
            day=torch.ones(6,dtype=torch.long),ids=[str(i) for i in range(6)],
            parent_margin=torch.full((6,),30.),parent_correct=torch.ones(6,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError,'original cosine head'):
            analyze(cache,None,-z)

    def test_unknown_source_environment_is_rejected(self):
        with self.assertRaises(KeyError):
            intervention(torch.tensor([[1.,0.]]),torch.tensor([99]),torch.tensor([1]),{1:{}},.25)

    def test_integer_overall_boundary(self):
        parent=dict(overall=dict(errors=20,count=1000),receiver={'r':dict(errors=20,count=1000)})
        primary=dict(overall=dict(errors=22,count=1000),receiver={'r':dict(errors=22,count=1000)})
        self.assertTrue(identity_safe(parent,primary))
        primary['overall']['errors']=23;self.assertFalse(identity_safe(parent,primary))

    def test_integer_receiver_boundary(self):
        parent=dict(overall=dict(errors=20,count=1000),receiver={'r':dict(errors=10,count=200)})
        primary=dict(overall=dict(errors=20,count=1000),receiver={'r':dict(errors=11,count=200)})
        self.assertTrue(identity_safe(parent,primary))
        primary['receiver']['r']['errors']=12;self.assertFalse(identity_safe(parent,primary))

if __name__=='__main__':unittest.main()
