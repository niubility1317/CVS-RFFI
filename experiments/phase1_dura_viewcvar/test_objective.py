import unittest
import torch
from experiments.phase1_dura_viewcvar.objective import worst_two_view_loss

class WorstViewsTest(unittest.TestCase):
    def test_selects_two_largest_for_each_record(self):
        x=torch.tensor([1.,10.,2.,8.,3.,0.],requires_grad=True)
        groups=torch.tensor([4,8,4,8,4,8])
        loss=worst_two_view_loss(x,groups)
        self.assertAlmostEqual(loss.item(),5.75)
        grad=torch.autograd.grad(loss,x)[0]
        torch.testing.assert_close(grad,torch.tensor([0.,.25,.25,.25,.25,0.]))

    def test_group_balancing_uses_records_not_views(self):
        x=torch.tensor([1.,2.,3.,1.,2.,3.,1.,2.,3.],requires_grad=True)
        groups=torch.tensor([4,4,8,4,4,8,4,4,8])
        self.assertAlmostEqual(worst_two_view_loss(x,groups).item(),2.25)

    def test_invalid_pairing_is_rejected(self):
        x=torch.ones(6)
        with self.assertRaises(ValueError):
            worst_two_view_loss(x,torch.tensor([4,8,4,4,4,8]))
        with self.assertRaises(ValueError):
            worst_two_view_loss(x[:5],torch.ones(5))

if __name__=='__main__':unittest.main()
