import unittest
from unittest.mock import patch
import torch
from experiments.phase1_wavelet_lifting.model import WaveletLifting
from experiments.phase1_wavelet_shift.model import ShiftAugmentedWavelet

class ShiftAugmentationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(44)
        self.model=ShiftAugmentedWavelet()
        self.x=torch.randn(2,2,256);self.base=torch.randn(2,2,256);self.lv=torch.randn(2)
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0,.03)
            self.model.gate_output.bias.normal_(0,.03)

    def test_train_rolls_input_and_restores_output(self):
        self.model.train()
        with patch('experiments.phase1_wavelet_shift.model.torch.randint',return_value=torch.tensor(3)):
            result=self.model(self.x,self.base,self.lv)
        shifted=torch.roll(self.x,3,dims=2);anchor=torch.roll(self.base,3,dims=2)
        expected=torch.roll(super(ShiftAugmentedWavelet,self.model).forward(shifted,anchor,self.lv),-3,dims=2)
        torch.testing.assert_close(result,expected,rtol=0,atol=0)

    def test_eval_matches_unmodified_single_origin(self):
        self.model.eval()
        actual=self.model(self.x,self.base,self.lv)
        expected=super(ShiftAugmentedWavelet,self.model).forward(self.x,self.base,self.lv)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_exact_identity_initially_even_with_shift(self):
        model=ShiftAugmentedWavelet().train()
        with patch('experiments.phase1_wavelet_shift.model.torch.randint',return_value=torch.tensor(7)):
            torch.testing.assert_close(model(self.x,self.base,self.lv),self.base,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
