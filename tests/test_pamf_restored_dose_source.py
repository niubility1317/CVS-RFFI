import unittest
import torch
from experiments.pamf_channel_pl.restored_dose_source import soft_loss,decide_dose,BATCHES


class SourceDoseTests(unittest.TestCase):
    def test_soft_supervision_is_masked_over_full_native_batch(self):
        q=torch.tensor([[.8,.2],[.1,.9]])
        z=torch.zeros((2,2),requires_grad=True)
        loss=soft_loss(z,q,torch.tensor([True,False]))
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()),float(torch.log(torch.tensor(2.)))/2)
        self.assertTrue(torch.equal(z.grad[1],torch.zeros(2)))
        with self.assertRaises(ValueError):soft_loss(z,q.requires_grad_(True),torch.tensor([True,True]))

    def test_source_dose_caps_weight_and_reduces_large_extra_gradients(self):
        rows=[dict(primary_norm=2.,restored_extra_norm=1.,primary_restored_cosine=.1) for _ in range(BATCHES)]
        self.assertEqual(decide_dose(rows)['selected_weight'],.25)
        for r in rows:r['restored_extra_norm']=4.
        result=decide_dose(rows)
        self.assertEqual(result['selected_weight'],.125)
        self.assertEqual(result['median_weighted_extra_to_primary'],.25)

    def test_nonfinite_empty_or_strongly_conflicting_probe_does_not_pass(self):
        with self.assertRaises(ValueError):decide_dose([])
        rows=[dict(primary_norm=2.,restored_extra_norm=1.,primary_restored_cosine=-.8) for _ in range(BATCHES)]
        self.assertFalse(decide_dose(rows)['source_gradient_pass'])
        rows[0]['primary_norm']=float('nan')
        with self.assertRaises(ValueError):decide_dose(rows)


if __name__=='__main__':unittest.main()
