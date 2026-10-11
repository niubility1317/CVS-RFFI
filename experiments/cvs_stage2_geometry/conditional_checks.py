"""Algebra, out-of-fold isolation and bounded-potential tests; synthetic only."""
import unittest
import torch
from .conditional_audit import (fit_linear,center_blocks,group_weights,
    tangent_prediction,tangent_oracle,metric,analyze)


class ConditionalChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.generator=torch.Generator().manual_seed(718)

    def test_rotating_own_jacobian_transfers_and_global_does_not(self):
        j=torch.tensor([[[1.],[0.]],[[1.],[0.]],
                        [[0.],[1.]],[[0.],[1.]]],dtype=torch.float64)
        a=torch.tensor([[-1.],[1.],[-1.],[1.]],dtype=torch.float64)
        blocks=[dict(block_id=0,group_ids=[0,1]),dict(block_id=1,group_ids=[2,3])]
        truth,_=tangent_prediction(j,a,blocks)
        x=j*a[:,None,:]
        w=group_weights(blocks[:1],4)
        coef,_=fit_linear(x,truth,w,normalize_columns=True)
        predicted=x@coef
        score=metric(truth,predicted,blocks[1:])
        self.assertLess(score['mse_ratio'],.01)
        global_prediction=torch.cat([truth[:2],truth[:2]],0)
        self.assertEqual(metric(truth,global_prediction,blocks[1:])['mse_ratio'],2.)

    def test_column_scaling_uses_only_fit_and_truth_isolation(self):
        x=torch.randn(6,5,3,generator=self.generator,dtype=torch.float64)
        y=torch.randn(6,5,generator=self.generator,dtype=torch.float64)
        w=torch.tensor([.5,.5,0,0,0,0],dtype=torch.float64)
        a,d1=fit_linear(x,y,w,normalize_columns=True)
        x[2:]*=1e6;y[2:]+=1e6
        b,d2=fit_linear(x,y,w,normalize_columns=True)
        self.assertTrue(torch.equal(a,b));self.assertEqual(d1,d2)

    def test_bounds_center_cycles_and_oracle(self):
        j=torch.randn(4,10,2,generator=self.generator,dtype=torch.float64)
        a=torch.randn(4,2,generator=self.generator,dtype=torch.float64)*10
        blocks=[dict(block_id=0,group_ids=[0,1,2,3])]
        truth,_=tangent_prediction(j,a,blocks)
        bounded,diag=tangent_prediction(j,a,blocks,True)
        self.assertLess(diag[0]['beta'],1)
        self.assertLess(float((a*diag[0]['beta']).abs().max()),1+1e-12)
        self.assertLess(float(bounded.mean(0).abs().max()),1e-12)
        self.assertLess(float(((bounded[0]-bounded[1])+(bounded[1]-bounded[2])+
                              (bounded[2]-bounded[0])).abs().max()),1e-12)
        oracle,_,info=tangent_oracle(j,truth,blocks)
        self.assertLess(metric(truth,oracle,blocks)['mse_ratio'],1e-20)
        self.assertFalse(info['predictive']);self.assertTrue(info['heldout_truth_used'])

    def test_full_fold_heldout_feature_does_not_fit_coefficients(self):
        n=6*2*2*8
        labels=torch.arange(6).repeat_interleave(32)
        rx=torch.tensor([0]*8+[1]*8).repeat(12)
        day=torch.tensor([0]*16+[1]*16).repeat(6)
        state=torch.randn(n,40,generator=self.generator);state[:,32]=.5
        z=torch.nn.functional.normalize(torch.randn(n,160,generator=self.generator),dim=1)
        cache=dict(ids=[str(i) for i in range(n)],y=labels,rx=rx,day=day,state=state,
            parent_margin=torch.zeros(n),parent_correct=torch.ones(n,dtype=torch.bool),parent_feature=z)
        j=torch.randn(n,160,8,generator=self.generator)*.01
        result,fit=analyze(cache,j)
        cache['parent_feature']=z.clone();cache['parent_feature'][labels<2]*=-1
        _,altered=analyze(cache,j)
        for name in ('tangent_state','tangent_environment','global_full','global_environment'):
            self.assertTrue(torch.equal(fit['0'][name],altered['0'][name]),name)
        self.assertEqual(len(result['folds']),3);self.assertEqual(result['identity_updates'],0)
        self.assertFalse(result['target_access'])


if __name__=='__main__':unittest.main()
