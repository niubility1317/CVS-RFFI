import unittest
from unittest.mock import patch
from types import SimpleNamespace
import experiments
import torch
from torch import nn
from experiments.phase1_dura_physical_fusion.pipeline import input_noise,predict_batch

class Physical(nn.Module):
 def __init__(self):super().__init__();self.calls=[]
 def forward(self,x):self.calls.append(len(x));return x*.5,torch.full((len(x),),-2.)
class Gate(nn.Module):
 def forward(self,x):return torch.tensor([-2.,3.])[:len(x)]
class Reader(nn.Module):
 def features(self,x):return x.flatten(1)
 def classify_features(self,x):return x[:,:6]
class Model(nn.Module):
 def forward(self,x,p,lv):
  assert p.shape==(len(x),4,256)
  return p[:,2:]
class PipelineTest(unittest.TestCase):
 def test_pack_is_same_record_and_forward_gates_physical(self):
  x=torch.arange(1024,dtype=torch.float32).reshape(2,2,256);dura=lambda y:y+10;physical=Physical()
  pack,lv=input_noise(x,None,dura,physical,None)
  self.assertTrue(torch.equal(pack[:,:2],x+10));self.assertTrue(torch.equal(pack[:,2:],x*.5))
  a,out,active,classes=predict_batch(x,Model(),None,Gate(),dura,physical,Reader(),None)
  self.assertEqual(active.tolist(),[False,True]);self.assertEqual(physical.calls,[2,1])
  self.assertTrue(torch.equal(out[0],a[0]));self.assertTrue(torch.equal(out[1],x[1]*.5));self.assertEqual(len(classes),3)

 def test_loader_survives_reader_package_path_reset(self):
  from experiments.phase1_dura_physical_fusion.pipeline import load_components
  deps=SimpleNamespace(SOURCE='blind',PRIOR_RUN='prior',BlindActionNet=nn.Identity,action_matrix=lambda x:x,posterior=lambda *x:x,
   s=SimpleNamespace(GATE_RUN='gate',GATE_COMMIT='g',BlindGate=nn.Identity))
  states=[dict(commit='94693a342bb2daa684c0f21415d7b949f9fb2107',epoch=10,model={}),
          dict(source_ids=8192,means=torch.zeros(1,2),covariances=torch.eye(2)[None],priors=torch.ones(1)),
          dict(commit='g',selected=dict(kind='gated_direct_iq'),gate={})]
  with patch.object(experiments,'__path__',[]), patch('torch.load',side_effect=states), \
       patch('experiments.phase1_dura_physical_fusion.pipeline.frozen_dura',return_value=nn.Identity()):
   blind,gate,dura,physical=load_components(__import__('pathlib').Path('.'),'cpu',deps)
  self.assertIs(physical.blind,blind);self.assertFalse(physical.training);self.assertFalse(gate.training)

if __name__=='__main__':unittest.main()
