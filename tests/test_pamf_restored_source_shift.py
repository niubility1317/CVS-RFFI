import unittest
import torch
from experiments.pamf_channel_pl.restored_source_shift import phase_aligned_nmse,validate_context


class FrozenSourceShiftTests(unittest.TestCase):
    def test_fidelity_removes_only_global_complex_gain_and_phase(self):
        torch.manual_seed(7);x=torch.randn(4,2,256)
        z=torch.stack((-2*x[:,1],2*x[:,0]),dim=1)
        self.assertTrue(torch.all(phase_aligned_nmse(x,z)<1e-10))
        self.assertTrue(torch.all(phase_aligned_nmse(x,x+torch.randn_like(x))>.1))

    def test_source_only_context_rejects_target_handles(self):
        with self.assertRaises(ValueError):validate_context(dict(target_inputs='target-IQ'))


if __name__=='__main__':unittest.main()
