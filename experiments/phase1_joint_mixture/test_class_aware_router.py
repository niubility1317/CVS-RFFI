import importlib,importlib.util,unittest,math
import torch
class ClassAwareRouterTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('class_aware_router'));self.m=importlib.import_module('class_aware_router')
 def test_selects_actual_candidate_and_waveform(self):
  net=self.m.ClassAwareRouter();pred=torch.tensor([[0,1,2,3,4],[4,4,4,4,4]],dtype=torch.long);choice,value=net.choose(pred);self.assertTrue(torch.equal(value,pred[torch.arange(2),choice]));waves=torch.arange(2*5*2*256,dtype=torch.float32).reshape(2,5,2,256);chosen=self.m.select_waveforms(waves,choice);torch.testing.assert_close(chosen,waves[torch.arange(2),choice])
 def test_multi_positive_likelihood_matches_hand_calculation(self):
  scores=torch.zeros(1,5,requires_grad=True);pred=torch.tensor([[0,1,1,2,3]],dtype=torch.long);loss=self.m.multi_positive_loss(scores,pred,torch.tensor([1]),torch.ones(1));self.assertAlmostEqual(loss.item(),-math.log(2/5),places=6);loss.backward();self.assertTrue(torch.isfinite(scores.grad).all());self.assertGreater(scores.grad.abs().sum().item(),0)
 def test_uncovered_identity_is_rejected(self):
  with self.assertRaises(ValueError):self.m.multi_positive_loss(torch.zeros(1,5),torch.tensor([[0,1,2,3,4]]),torch.tensor([5]),torch.ones(1))
 def test_learns_class_dependent_multi_candidate_reliability(self):
  torch.manual_seed(7);target=torch.arange(6,dtype=torch.long).repeat(40);pred=torch.stack(((target+1)%6,torch.where(target%2==0,target,(target+2)%6),torch.where(target%2==1,target,(target+3)%6),(target+4)%6,(target+5)%6),1);net=self.m.ClassAwareRouter();opt=torch.optim.AdamW(net.parameters(),lr=.01)
  for _ in range(160):
   loss=self.m.multi_positive_loss(net(pred),pred,target,torch.ones(len(target)));opt.zero_grad();loss.backward();opt.step()
  _,selected=net.choose(pred);self.assertGreater((selected==target).float().mean().item(),.99)
if __name__=='__main__':unittest.main()
