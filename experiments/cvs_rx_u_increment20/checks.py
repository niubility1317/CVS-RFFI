"""Focused source-role exclusions, stored norm matching and backward isolation."""
from types import SimpleNamespace
import unittest
import torch
from . import design as d
from .fold import selected_excluding
from .geometry import match_stored_norm
from .observer import backward_L

class Metadata:
    def __init__(self,rx):self.rx_i=rx
    @property
    def tx_i(self):raise AssertionError('U transmitter field is forbidden')
class Dataset:
    def __init__(self,n,hide):self.index=[Metadata(d.RX[i%5]) for i in range(n)];self.selected=list(range(n));self.hide=hide
    def __len__(self):return len(self.index)
    def __getitem__(self,i):raise AssertionError('Role filtering must not fetch IQ')

class Checks(unittest.TestCase):
    def test_source_roles_without_IQ_or_U_truth(self):
        for role,count in (('L_s',6300),('U_s',56700),('V',27000)):
            ds=Dataset(count,role=='U_s');selected=selected_excluding(ds,1,role)
            self.assertEqual(len(selected),count*4//5)
            self.assertFalse(any(ds.index[i].rx_i==1 for i in selected))
    def test_roles_rejected_before_data(self):
        for role in ('target','query','support','V_cal','U_truth'):
            with self.assertRaises(ValueError):selected_excluding(object(),1,role)
    def test_incomplete_or_unmasked_U(self):
        for ds in (Dataset(56700,False),Dataset(56000,True)):
            with self.assertRaises(ValueError):selected_excluding(ds,1,'U_s')
    def test_backward_restores_original_grad_objects_and_None(self):
        x=torch.nn.Parameter(torch.tensor([2.,-3.]));unused=torch.nn.Parameter(torch.tensor([7.]))
        sentinel=torch.tensor([9.,10.]);x.grad=sentinel
        gradients=backward_L((x*x).sum(),(x,unused))
        self.assertTrue(torch.equal(gradients[0],torch.tensor([4.,-6.])))
        self.assertIsNone(gradients[1]);self.assertIs(x.grad,sentinel);self.assertIsNone(unused.grad)
    def test_norm_matches_stored_FP32_positions(self):
        generator=torch.Generator().manual_seed(12)
        base=torch.randn(2048,generator=generator);raw=base+torch.randn(2048,generator=generator)*.0001
        target=base+torch.randn(2048,generator=generator)*.00006
        control,facts=match_stored_norm(base,raw,target)
        self.assertTrue(facts['norm_matched']);self.assertLessEqual(facts['relative_norm_error'],1e-3)
        self.assertTrue(torch.allclose(control,base+facts['scale']*(raw-base),atol=2e-7,rtol=2e-7))
    def test_invalid_positions(self):
        for base,raw,target in ((torch.zeros(2).double(),torch.ones(2),torch.ones(2)),(torch.zeros(2),torch.zeros(2),torch.ones(2))):
            with self.assertRaises(ValueError):match_stored_norm(base,raw,target)
    def test_gate_cannot_promote_main(self):
        self.assertFalse(d.gate([])['main_run_qualification']);self.assertFalse(d.gate([])['full_source_training_qualification'])
    def test_U_must_outperform_L_only_features(self):
        import copy
        record=dict(step=2896,U_loss=.1,geometry=dict(raw_U_F_to_full_F_step=.2,retained_U_F_norm_ratio=.8,
            CONE_RAW_cosine=.8,stored_norm_control=dict(norm_matched=True),rounded_cone_violation_relative=0.),
            paired={control:{readout+'_'+metric:(-.01 if metric=='CE_difference' else 0)
                for readout in ('ACTUAL','PRE_W') for metric in ('CE_difference','error_difference')}
                for control in ('RAW','SHRINK','L_ONLY_F')})
        records=[copy.deepcopy(record) for _ in d.PROBES]
        for step,row in zip(d.PROBES,records):row['step']=step
        self.assertTrue(d.gate(records)['full_source_training_qualification'])
        for row in records:row['paired']['L_ONLY_F']['ACTUAL_CE_difference']=0.
        self.assertFalse(d.gate(records)['full_source_training_qualification'])

if __name__=='__main__':unittest.main(verbosity=2)
