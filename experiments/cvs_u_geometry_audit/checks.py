import argparse,torch
from . import design as d
from .geometry import terms,record,aggregate,gradient_vectors,gradient_summary

def check():
 torch.manual_seed(21);w=torch.randn(6,12,dtype=torch.float64);b=torch.randn(10,12,dtype=torch.float64,requires_grad=True)
 z=(b.detach()+.1*torch.randn_like(b)).requires_grad_(True)
 logits=z@w.T;target=(b@w.T).requires_grad_(True)
 a=terms(logits,z,target,b);shift=terms(logits+13,z,target-7,b)
 assert all(torch.allclose(a[k],shift[k],atol=1e-12) for k in a)
 sum(v.mean() for v in a.values()).backward();assert b.grad is None and torch.isfinite(z.grad).all() and z.grad.norm()>0
 equal=terms(logits.detach(),z.detach(),logits.detach(),z.detach())
 assert all(float(v.abs().max())<1e-12 for v in equal.values())
 r,rank=record(logits.detach(),z.detach(),target.detach(),b.detach(),w)
 assert rank<=5 and torch.allclose(r['feature_energy'],r['row_energy']+r['residual_energy'],atol=1e-12)
 # Huge logit confidence makes probability KL small while centered-margin loss persists.
 x=torch.tensor([[25.,0.,0.]],dtype=torch.float64);t=torch.tensor([[20.,0.,0.]],dtype=torch.float64)
 test=terms(x,torch.ones(1,4),t,torch.ones(1,4));assert test['KL'].item()<1e-6 and test['margin'].item()>1
 for row in d.rows():d.validate(d.config(row))
 return dict(status='PASS',target_stopgrad=True,logit_shift_invariant=True,identity_view_zero=True,exact_energy_partition=True,KL_saturation_counterexample=True,rows=3)

def preflight():
 from .run import load,gradients,forward
 from experiments.cvs_equivariant_identity.precision import numerical_context
 value=check();c=d.config(d.rows()[0])
 with numerical_context(d.parent.FULL_FP32_POLICY):
  m,h,L,U,p,_,_=load(c,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
  x=L['x'][:8];scores,z=forward(m,x)
  expected=30*torch.nn.functional.linear(torch.nn.functional.normalize(z,dim=1,eps=1e-4),torch.nn.functional.normalize(m.head.weight,dim=1,eps=1e-4))
  assert torch.allclose(scores,expected,atol=1e-5)
  rows=gradients(c,m,h,L,U,p,'cpu',count=1)
  assert len(rows)==2 and all(torch.equal(before[k],v) for k,v in m.state_dict().items())
  assert all(row['gradients'][k]['norm']>0 for row in rows for k in ('L_CE','KL','margin','feature'))
 value.update(actual_E200_no_update=True,actual_cosine_head_verified=True,actual_L_U_three_objective_gradients=True);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();torch.set_num_threads(2);d.write(a.output,preflight() if a.remote_preflight else check())
