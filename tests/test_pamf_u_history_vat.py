import unittest
import torch
from types import SimpleNamespace
from experiments.pamf_u_history.vat_objective import unit_direction,Exposure,Objective,KEYS


class VATTests(unittest.TestCase):
    def test_direction_handles_zero_and_extreme_scales(self):
        x=torch.tensor([[[0.,0.],[0.,0.]],[[1e-30,-1e-30],[2e-30,1e-30]],[[1e20,-1e20],[1e20,1e20]]])
        u=unit_direction(x);norm=u.flatten(1).norm(dim=1)
        self.assertTrue(torch.isfinite(u).all());self.assertEqual(float(norm[0]),0.)
        self.assertTrue(torch.allclose(norm[1:],torch.ones(2),atol=1e-6))

    def test_exposure_rejects_visible_truth_and_foreign_records(self):
        meta={k:torch.tensor([v]) for k,v in zip(KEYS,(1,3,1,10,99))};e=Exposure(meta)
        metadata=dict(meta,tx_label_visible=torch.tensor([False]));batch=(torch.ones(1,2,256),torch.tensor([-1]),torch.tensor([0]),metadata)
        self.assertEqual(e.indices(batch,'U_s',True).tolist(),[0])
        with self.assertRaises(ValueError):e.indices(batch,'V',True)
        metadata['tx_label_visible'][0]=True
        with self.assertRaises(ValueError):e.indices(batch,'U_s',True)
        self.assertEqual(e.seen,0)

    def test_all_low_confidence_U_receives_VAT_without_probe_parameter_gradients(self):
        torch.manual_seed(9)
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm1d(512);self.head=torch.nn.Linear(512,6)
            def forward(self,x):return self.head(self.bn(x.flatten(1)))
            def forward_identity_only(self,x,domain_labels):return dict(tx_logits=self(x))
        model=Student();n=4;x=torch.randn(n,2,256)
        metadata={k:torch.tensor([base+i if k in ('sig_i','base_index') else base for i in range(n)]) for k,base in zip(KEYS,(1,3,1,10,99))}
        batchmeta=dict(metadata,tx_label_visible=torch.zeros(n,dtype=torch.bool))
        batch=(x,torch.full((n,),-1),torch.zeros(n,dtype=torch.long),batchmeta)
        model._cd_U_loader=SimpleNamespace(dataset=SimpleNamespace(role='U_s',metadata=metadata))
        model._cd_U_iter=iter([batch]);model._feature_l_steps=1;model._cd_U_seen=model._cd_U_accepted=0
        objective=Objective(dict(step_offset=4440));loss=objective(model,'cpu')
        self.assertEqual(objective.history.accepted,0)
        self.assertEqual(objective.history.seen,n)
        self.assertEqual(objective.history.nonzero_directions,n)
        self.assertEqual(objective.history.probe_checks,1)
        self.assertEqual(int(model.bn.num_batches_tracked),2)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertLessEqual(objective.history.max_relative_radius,.030003)
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(float(model.head.weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
