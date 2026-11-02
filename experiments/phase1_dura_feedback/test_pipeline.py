import unittest
import torch
from experiments.phase1_dura_feedback.pipeline import input_noise,predict_batch

class DummyBlind(torch.nn.Module):
    def forward(self,x):return None,x.new_zeros(len(x))
class DummyGate(torch.nn.Module):
    def forward(self,x):return x.new_full((len(x),1),10.)
class DummyDura(torch.nn.Module):
    def forward(self,x):return 2*x
class DummyReader(torch.nn.Module):
    def features(self,x):return x.flatten(1)[:,:160]
    def classify_features(self,x):return x[:,:6]
class DummyModel(torch.nn.Module):
    def forward(self,x,packed,lv):return packed[:,:2]+.1*x

class FeedbackPipelineTest(unittest.TestCase):
    def setUp(self):
        self.x=torch.randn(3,2,256);self.blind=DummyBlind();self.gate=DummyGate()
        self.dura=DummyDura();self.cond=DummyReader().eval();self.reader=DummyReader().eval()
    def test_training_condition_uses_dura_of_same_record(self):
        packed,lv=input_noise(self.x,self.blind,self.dura,self.cond,None)
        torch.testing.assert_close(packed[:,:2],2*self.x)
        torch.testing.assert_close(packed[:,2,:160],self.cond.features(2*self.x))
        torch.testing.assert_close(packed[:,2,160:166],self.cond.classify_features(self.cond.features(2*self.x)))
        self.assertEqual(tuple(lv.shape),(3,))
    def test_inference_returns_iq_and_three_decisions(self):
        anchor,out,active,classes=predict_batch(self.x,DummyModel(),self.blind,self.gate,self.dura,self.cond,self.reader,None)
        torch.testing.assert_close(anchor,2*self.x)
        torch.testing.assert_close(out,2.1*self.x)
        self.assertTrue(active.all())
        self.assertEqual(len(classes),3)
        self.assertTrue(all(tuple(c.shape)==(3,) for c in classes))
if __name__=='__main__':unittest.main()
