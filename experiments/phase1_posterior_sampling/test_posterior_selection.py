import unittest
import torch
from posterior_selection import choose_consensus,select_waveforms

class PosteriorSelectionTest(unittest.TestCase):
 def test_consensus_uses_mixture_weights_and_returns_actual_wave(self):
  prob=torch.tensor([[[.9,.1],[.1,.9]],[[.9,.1],[.2,.8]]]);logw=torch.tensor([[.8,.2],[.3,.7]]).log();choice,cls=choose_consensus(prob,logw);self.assertEqual(cls.tolist(),[0,1]);self.assertEqual(choice.tolist(),[0,1]);waves=torch.tensor([[[1.],[2.]],[[3.],[4.]]]);self.assertEqual(select_waveforms(waves,choice).flatten().tolist(),[1.,4.])
 def test_bad_probabilities_are_rejected(self):
  with self.assertRaises(ValueError):choose_consensus(torch.tensor([[[.8,.8]]]),torch.zeros(1,1))
if __name__=='__main__':unittest.main()
