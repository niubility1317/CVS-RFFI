import unittest
import torch
from gate_aware_router import GateAwareRouter,select_waveforms

class GateAwareRouterTest(unittest.TestCase):
 def test_selects_actual_candidate_waveform(self):
  net=GateAwareRouter();pred=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]]);gate=torch.tensor([.3,.9]);choice,decision=net.choose(pred,gate);waves=torch.randn(2,5,2,256);chosen=select_waveforms(waves,choice);torch.testing.assert_close(chosen,waves[torch.arange(2),choice]);self.assertTrue(torch.equal(decision,pred[torch.arange(2),choice]))
 def test_continuous_quality_can_change_route(self):
  net=GateAwareRouter()
  with torch.no_grad():
   for p in net.parameters():p.zero_()
   net.network[0].weight[0,30]=10.;net.network[0].bias[0]=-5.;net.network[2].weight[0,0]=1.;net.network[4].weight[0,0]=1.;net.network[4].bias[1]=.1
  pred=torch.tensor([[0,1,2,3,4],[0,1,2,3,4]]);choice,_=net.choose(pred,torch.tensor([0.,1.]));self.assertEqual(choice.tolist(),[1,0])
 def test_rejects_invalid_quality_and_class(self):
  net=GateAwareRouter()
  with self.assertRaises(ValueError):net(torch.tensor([[0,1,2,3,6]]),torch.tensor([.5]))
  with self.assertRaises(ValueError):net(torch.tensor([[0,1,2,3,4]]),torch.tensor([1.1]))
if __name__=='__main__':unittest.main()
