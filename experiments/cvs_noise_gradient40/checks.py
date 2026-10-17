"""Synthetic derivative/role negatives;no actual IQ,model weights or target."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from .maths import cosine,head_weight_vectors,conv_vectors,OBJECTIVES
from .criteria import decide
from .protocol import validate_role,validate_config
from .probe import stack
from .analyze import angle,decision
import numpy as np
class Cases(unittest.TestCase):
 def test_each_objective_actual_W_parameter_gradient(self):
  g=torch.Generator().manual_seed(40);j=torch.randn(3,160,generator=g);w=torch.randn(6,160,generator=g,requires_grad=True);y=torch.tensor([0,3,5]);z=30*F.linear(F.normalize(j,dim=1,eps=1e-4),F.normalize(w,dim=1,eps=1e-4));others=z.detach().clone();others[torch.arange(3),y]=-torch.inf;c=others.argmax(1)
  losses=dict(CE0=F.cross_entropy(z,y,reduction='none'),CE01=F.cross_entropy(z,y,reduction='none',label_smoothing=.01),margin=-(z[torch.arange(3),y]-z[torch.arange(3),c])/30)
  for objective in OBJECTIVES:
   actual=torch.stack([torch.autograd.grad(losses[objective][i],w,retain_graph=True)[0].flatten() for i in range(3)])
   self.assertTrue(torch.allclose(head_weight_vectors(j,z,y,w,objective),actual,atol=2e-5,rtol=2e-5),objective)
 def test_gate_parameter_gradient_exact_per_packet(self):
  g=torch.Generator().manual_seed(41);x=torch.randn(3,4,16,generator=g);m=nn.Conv1d(4,2,5,padding=2);z=m(x);up=torch.randn(z.shape,generator=g);actual=[]
  for i in range(3):
   a,b=torch.autograd.grad((z[i]*up[i]).sum(),(m.weight,m.bias),retain_graph=True);actual.append(torch.cat((a.flatten(),b)))
  self.assertTrue(torch.allclose(conv_vectors(x,up,m),torch.stack(actual),atol=2e-5,rtol=2e-5))
 def test_LS_calibration_conflict_not_identity_primary(self):
  j=torch.zeros(1,160);j[0,0]=1;w=torch.zeros(6,160);w[0,:2]=torch.tensor([1.,1.]);w[1:,:2]=torch.tensor([.4,1.]);y=torch.tensor([0]);z=30*F.linear(F.normalize(j,dim=1,eps=1e-4),F.normalize(w,dim=1,eps=1e-4));a=head_weight_vectors(j,z,y,w,'CE0');b=head_weight_vectors(j,z,y,w,'CE01');c,v=cosine(a,b);self.assertTrue(v[0]);self.assertLess(c[0],-.99)
  with self.assertRaises(ValueError):decide([dict(draw=d,objective='CE01',parameter='head_W',N=900,valid_full=900,valid_no_noise=900,negative_full=300,negative_no_noise=50) for d in (0,1)])
 def test_all900_denominator_and_both_draws(self):
  cells=[dict(draw=d,objective='margin',parameter='head_W',N=900,valid_full=900,valid_no_noise=900,negative_full=225,negative_no_noise=90) for d in (0,1)];self.assertTrue(decide(cells)['margin_noise_conflict_evidence']);self.assertTrue(decision(cells)['margin_noise_conflict_evidence']);cells[1]['negative_no_noise']=180;self.assertFalse(decide(cells)['margin_noise_conflict_evidence']);self.assertFalse(decision(cells)['margin_noise_conflict_evidence']);cells[1]['negative_no_noise']=90;cells[1]['valid_full']=800;self.assertFalse(decide(cells)['margin_noise_conflict_evidence'])
  with self.assertRaises(ValueError):decide(cells[:1])
 def test_zero_gradient_kept_and_independent_cosine(self):
  a=torch.tensor([[1.,0.],[1.,0.],[0.,0.]]);b=torch.tensor([[-1.,0.],[0.,1.],[1.,1.]]);c,v=cosine(a,b);cc,vv,_,_=angle(np.asarray(a.tolist()),np.asarray(b.tolist()));self.assertEqual(v.tolist(),[True,True,False]);self.assertTrue(torch.isnan(c[2]));self.assertEqual(vv.tolist(),v.tolist());self.assertEqual(cc[:2].tolist(),c[:2].tolist())
 def test_role_no_V_U_or_RX_holdout(self):
  ids=['L'+str(i) for i in range(6300)];contract=dict(role_ids=dict(L_s=ids,U_s=['U0'],V=['V0']));rx=[(1,3,4,6,8)[i%5] for i in range(6300)];day=[i%3+1 for i in range(6300)];y=[i%6 for i in range(6300)];validate_role('L_s',ids,contract,rx,day,y)
  for role in ['V','U_s','target']:
   with self.assertRaises(ValueError):validate_role(role,ids,contract,rx,day,y)
  with self.assertRaises(ValueError):validate_role('L_s',ids,contract,[3]*6300,day,y)
  ids2=ids.copy();ids2[0]='V0'
  with self.assertRaises(ValueError):validate_role('L_s',ids2,contract,rx,day,y)
 def test_actual_physics_values_not_defaults(self):
  c=dict(processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2462000000.,boundary='reflect',quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True);validate_config(c)
  for k,v in [('equalization_enabled',True),('reference_snr_db',40.),('fc_hz',2450000000.),('quality_adaptation_enabled',False),('processing_route','full')]:
   with self.assertRaises(ValueError):validate_config(dict(c,**{k:v}))
 def test_invalid_head_and_gate_inputs(self):
  for y in [torch.tensor([0,6]),torch.tensor([0.,1.]),torch.tensor([1])]:
   with self.assertRaises(ValueError):head_weight_vectors(torch.ones(2,160),torch.ones(2,6),y,torch.ones(6,160),'margin')
  with self.assertRaises(ValueError):conv_vectors(torch.ones(2,4,16),torch.ones(2,2,8),nn.Conv1d(4,2,5,stride=2,padding=2))
 def test_tensor_record_stack(self):
  z=stack([dict(a=torch.zeros(2,6),b={'x':torch.ones(2,160)}),dict(a=torch.zeros(1,6),b={'x':torch.ones(1,160)})]);self.assertEqual(z['a'].shape,(3,6));self.assertEqual(z['b']['x'].shape,(3,160))
if __name__=='__main__':unittest.main()
