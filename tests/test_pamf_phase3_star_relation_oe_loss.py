"""Pure loss mathematics and controlled gradient routes, without data access."""
import importlib
import inspect
import math
import unittest
import torch
from torch.nn import functional as F


class RelationOELossTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_relation_oe_loss')
    def data(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            values=[torch.randn(4,160,dtype=torch.float64,requires_grad=True),torch.randn(3,6,dtype=torch.float64,requires_grad=True),
                torch.randn(4,6,dtype=torch.float64,requires_grad=True),torch.randn(3,6,dtype=torch.float64,requires_grad=True),torch.tensor([0,2,5]),
                F.normalize(torch.randn(4,160,dtype=torch.float64),dim=1).requires_grad_()]
        return values

    def test_manual_CE_T2_KD_offdiagonal_Gram_and_uniform_KL(self):
        m=self.module();a=self.data();main=m.compute(*a,arm='relation_oe');control=m.compute(*a,arm='relation_control')
        self.assertEqual(set(main),{'total','ce','kd','gram_OE','OE_uniform'})
        z,logits,oe,teacher,y,unit=a;u=F.normalize(z,dim=1);v=F.normalize(unit.detach(),dim=1);mask=~torch.eye(4,dtype=torch.bool)
        gram=((u@u.T-v@v.T)[mask].square()).mean();p=F.softmax(teacher.detach()/2,dim=1)
        kd=4*(p*(F.log_softmax(teacher.detach()/2,dim=1)-F.log_softmax(logits/2,dim=1))).sum(1).mean()
        ce=F.cross_entropy(logits,y);uniform=-F.log_softmax(oe,dim=1).mean()-math.log(6)
        for k,v in dict(ce=ce,kd=kd,gram_OE=gram,OE_uniform=uniform,total=ce+kd+gram+uniform).items():torch.testing.assert_close(main[k],v,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(control['total'],ce+kd+gram,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(control['OE_uniform'],main['OE_uniform'],rtol=0,atol=0)

    def test_only_main_backpropagates_OE_default_logits_and_teacher_is_detached(self):
        m=self.module()
        for arm in m.ARMS:
            with self.subTest(arm=arm):
                a=self.data();before=[v.detach().clone() for v in a];out=m.compute(*a,arm=arm)
                gradients=torch.autograd.grad(out['total'],(a[0],a[1],a[2],a[3],a[5]),allow_unused=True)
                self.assertGreater(gradients[0].norm().item(),0);self.assertGreater(gradients[1].norm().item(),0)
                if arm=='relation_control':self.assertIsNone(gradients[2]);self.assertFalse(out['OE_uniform'].requires_grad)
                else:self.assertGreater(gradients[2].norm().item(),0)
                self.assertIsNone(gradients[3]);self.assertIsNone(gradients[4])
                for actual,expected in zip(a,before):torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_Gram_preserves_relations_but_is_invariant_to_global_rotation_and_scale(self):
        m=self.module();a=self.data();base=m.compute(*a,arm='relation_control')
        torch.manual_seed(73);q=torch.linalg.qr(torch.randn(160,160,dtype=torch.float64))[0]
        rotated=list(a);rotated[0]=(a[0]@q)*7
        moved=m.compute(*rotated,arm='relation_control')
        torch.testing.assert_close(base['gram_OE'],moved['gram_OE'],rtol=1e-12,atol=1e-12)
        exact=list(a);exact[0]=a[5]@q
        self.assertLess(m.compute(*exact,arm='relation_oe')['gram_OE'].item(),1e-24)

    def test_uniform_logits_have_zero_KL_and_diagonal_is_not_in_loss(self):
        m=self.module();a=self.data();a[2]=torch.zeros_like(a[2],requires_grad=True);a[0]=a[5].detach().clone().requires_grad_()
        out=m.compute(*a,arm='relation_oe');self.assertAlmostEqual(out['OE_uniform'].item(),0.,places=12);self.assertAlmostEqual(out['gram_OE'].item(),0.,places=12)

    def test_strict_visible_source_inputs_only(self):
        m=self.module()
        for defect in ('OE_single','z','L_logits','OE_logits','teacher','label','dtype','nan','arm'):
            with self.subTest(defect=defect):
                a=self.data();arm='relation_oe'
                if defect=='OE_single':a[0]=a[0][:1];a[2]=a[2][:1];a[5]=a[5][:1]
                elif defect=='z':a[0]=a[0][:,:159]
                elif defect=='L_logits':a[1]=a[1][:,:5]
                elif defect=='OE_logits':a[2]=a[2][:,:5]
                elif defect=='teacher':a[3]=a[3][:2]
                elif defect=='label':a[4][0]=6
                elif defect=='dtype':a[5]=a[5].float()
                elif defect=='nan':a[0]=torch.full_like(a[0],float('nan'))
                else:arm='old-joint'
                with self.assertRaises(ValueError):m.compute(*a,arm=arm)
        self.assertEqual(tuple(inspect.signature(m.compute).parameters),('z_OE','logits_L','logits_OE','teacher_logits_L','y_L','teacher_unit_OE','arm'))
        with self.assertRaises(TypeError):m.compute(*self.data(),arm='relation_oe',y_OE=torch.zeros(4))


if __name__=='__main__':unittest.main()
