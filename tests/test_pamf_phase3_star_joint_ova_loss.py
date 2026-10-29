"""Pure synthetic Torch losses/gradients; no model, IQ, cache or trainer."""
import importlib
import inspect
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


ROOT=Path(__file__).resolve().parents[1]
KEYS={'total','ce','kd','feature_L','feature_OE','feature_total',
      'ova_L_positive','ova_L_hardnegative','ova_OE','ova_total'}


class StarJointOVALossTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_joint_ova_loss.py').is_file(),
                        'Joint OVA loss is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_ova_loss')

    @staticmethod
    def data():
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(62)
            def floating(shape):return torch.randn(shape,dtype=torch.float64).requires_grad_()
            zL,zO=floating((4,160)),floating((3,160))
            logits,teacher=floating((4,6)),floating((4,6))
            tuL=F.normalize(torch.randn(4,160,dtype=torch.float64),dim=1).requires_grad_()
            tuO=F.normalize(torch.randn(3,160,dtype=torch.float64),dim=1).requires_grad_()
            return [zL,zO,logits,teacher,torch.tensor([0,2,5,1]),tuL,tuO,
                    floating((6,160)),floating((6,))]

    def test_both_arms_equal_value_but_only_joint_ova_reaches_encoder_and_both_train_aux(self):
        m=self.module();a=self.data();control=m.compute(*a,joint=False);joint=m.compute(*a,joint=True)
        self.assertEqual(set(joint),KEYS)
        for key in KEYS:
            self.assertEqual(joint[key].ndim,0);torch.testing.assert_close(joint[key],control[key],rtol=0,atol=0)
        selected=(a[0],a[1],a[7],a[8])
        gc=torch.autograd.grad(control['ova_total'],selected,allow_unused=True,retain_graph=True)
        gj=torch.autograd.grad(joint['ova_total'],selected,allow_unused=True,retain_graph=True)
        self.assertIsNone(gc[0]);self.assertIsNone(gc[1])
        for value in gj:self.assertGreater(value.norm().item(),0.)
        for index in (2,3):
            self.assertGreater(gc[index].norm().item(),0.)
            torch.testing.assert_close(gc[index],gj[index],rtol=0,atol=0)
        # Detaching auxiliary input must leave feature preservation and the
        # original identity CE/KD paths differentiable in the control arm.
        gz=torch.autograd.grad(control['total'],(a[0],a[1],a[2]),retain_graph=True)
        for value in gz:self.assertGreater(value.norm().item(),0.)

    def test_teachers_always_detached_and_inputs_preserved(self):
        m=self.module()
        for joint in (False,True):
            with self.subTest(joint=joint):
                a=self.data();before=[v.detach().clone() for v in a]
                result=m.compute(*a,joint=joint);result['total'].backward()
                for index in (3,5,6):self.assertIsNone(a[index].grad)
                for index in (0,1,2,7,8):self.assertGreater(a[index].grad.norm().item(),0.)
                for actual,expected in zip(a,before):torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_true_class_excluded_from_hard_negative_and_OE_max_competes_over_all_six(self):
        m=self.module();z=torch.ones(1,160,dtype=torch.float64,requires_grad=True)
        teacher=F.normalize(z.detach(),dim=1);logits=torch.zeros(1,6,dtype=torch.float64,requires_grad=True)
        weight=torch.zeros(6,160,dtype=torch.float64,requires_grad=True)
        bias=torch.tensor([9.,8.,7.,6.,5.,4.],dtype=torch.float64,requires_grad=True)
        out=m.compute(z,z,logits,logits,torch.tensor([0]),teacher,teacher,weight,bias,joint=True)
        positive=torch.autograd.grad(out['ova_L_positive'],bias,retain_graph=True)[0]
        negative=torch.autograd.grad(out['ova_L_hardnegative'],bias,retain_graph=True)[0]
        oe=torch.autograd.grad(out['ova_OE'],bias)[0]
        self.assertEqual(torch.nonzero(positive).flatten().tolist(),[0]);self.assertLess(positive[0].item(),0.)
        self.assertEqual(torch.nonzero(negative).flatten().tolist(),[1]);self.assertGreater(negative[1].item(),0.)
        self.assertEqual(torch.nonzero(oe).flatten().tolist(),[0]);self.assertGreater(oe[0].item(),0.)
        torch.testing.assert_close(out['ova_L_positive'],F.softplus(-bias[0]))
        torch.testing.assert_close(out['ova_L_hardnegative'],F.softplus(bias[1]))
        torch.testing.assert_close(out['ova_OE'],F.softplus(bias[0]))

    def test_manual_CE_T2_KL_role_equal_feature_and_OVA_weighted_total(self):
        m=self.module();L=torch.eye(160,dtype=torch.float64)[:2].requires_grad_()
        O=torch.eye(160,dtype=torch.float64)[2:3].repeat(3,1).requires_grad_()
        tl=L.detach().clone();tl[1]*=-1;to=O.detach().clone();to[[0,2]]*=-1
        logits=torch.tensor([[3.,-1.,2.,0.,-2.,1.],[0.,1.,2.,3.,-1.,-2.]],dtype=torch.float64,requires_grad=True)
        teacher=torch.tensor([[-1.,2.,0.,1.,3.,-2.],[2.,0.,-1.,-2.,1.,3.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,1]);w=torch.zeros(6,160,dtype=torch.float64,requires_grad=True)
        b=torch.tensor([2.,-1.,0.,-2.,1.,-3.],dtype=torch.float64,requires_grad=True)
        out=m.compute(L,O,logits,teacher,y,tl,to,w,b,joint=False)
        ce=-F.log_softmax(logits,dim=1)[torch.arange(2),y].mean()
        probability=F.softmax(teacher.detach()/2,dim=1)
        kd=4*(probability*(F.log_softmax(teacher.detach()/2,dim=1)-F.log_softmax(logits/2,dim=1))).sum(1).mean()
        positive=F.softplus(-b[y]).mean();negative=(F.softplus(b[4])+F.softplus(b[0]))/2
        oe=F.softplus(b.max());ova=.5*(.5*positive+.5*negative)+.5*oe
        for key,expected in {'ce':ce,'kd':kd,'feature_L':L.new_tensor(1.),'feature_OE':L.new_tensor(4/3),
                             'feature_total':L.new_tensor(7/6),'ova_total':ova,'total':ce+kd+5*(7/6)+ova}.items():
            torch.testing.assert_close(out[key],expected,rtol=1e-12,atol=1e-12)

    def test_unit_feature_scale_invariance_parallel_antipodal_zero_and_dot_clamp(self):
        m=self.module();a=self.data();out=m.compute(*a,joint=True)
        changed=list(a);changed[0]=a[0]*7.;changed[1]=a[1]*.125
        scaled=m.compute(*changed,joint=True)
        for key in ('feature_L','feature_OE','feature_total','ova_total'):
            torch.testing.assert_close(out[key],scaled[key],rtol=1e-12,atol=1e-12)
        unit=torch.eye(160,dtype=torch.float64)[:1]
        for orientation,expected in ((1.,0.),(-1.,2.),(1.000001,0.)):
            with self.subTest(orientation=orientation):
                data=[unit,unit,torch.zeros(1,6,dtype=torch.float64),torch.zeros(1,6,dtype=torch.float64),
                      torch.tensor([0]),orientation*unit,orientation*unit,
                      torch.zeros(6,160,dtype=torch.float64),torch.zeros(6,dtype=torch.float64)]
                result=m.compute(*data,joint=False)
                torch.testing.assert_close(result['feature_total'],unit.new_tensor(expected),rtol=0,atol=0)
        data[0]=torch.zeros_like(unit);data[1]=torch.zeros_like(unit)
        result=m.compute(*data,joint=True)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in result.values()))
        self.assertEqual(result['feature_total'].item(),1.)

    def test_strict_shapes_labels_finiteness_and_no_hidden_roles(self):
        m=self.module()
        for defect in ('z','logits','teacher_logits','unit','aux_weight','aux_bias','empty','label_range','label_dtype','nan','dtype','joint'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                a=self.data();joint=True
                if defect=='z':a[0]=a[0][:,:159]
                elif defect=='logits':a[2]=a[2][:,:5]
                elif defect=='teacher_logits':a[3]=a[3][:3]
                elif defect=='unit':a[6]=a[6][:2]
                elif defect=='aux_weight':a[7]=a[7][:5]
                elif defect=='aux_bias':a[8]=a[8][:5]
                elif defect=='empty':a[1]=a[1][:0]
                elif defect=='label_range':a[4][0]=6
                elif defect=='label_dtype':a[4]=a[4].double()
                elif defect=='nan':a[5]=torch.full_like(a[5],float('nan'))
                elif defect=='dtype':a[0]=a[0].float()
                else:joint=1
                m.compute(*a,joint=joint)
        self.assertEqual(tuple(inspect.signature(m.compute).parameters),
            ('z_L','z_OE','logits_L','teacher_logits_L','y_L','teacher_unit_L','teacher_unit_OE','aux_weight','aux_bias','joint'))
        with self.assertRaises(TypeError):m.compute(*self.data(),joint=True,y_OE=torch.zeros(3))
        with self.assertRaises(TypeError):m.compute(*self.data(),joint=True,V=torch.zeros(1))


if __name__=='__main__':unittest.main()
