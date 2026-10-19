"""Independent vector oracle and source paired-CE budget checks."""
import torch
def run():
 try:
  from .core import corrected,surrogate,block_update,contributions,active,strength
 except ModuleNotFoundError as e:raise AssertionError('EXPECTED_RED: missing weak-pair feature gradient mechanism') from e
 torch.set_num_threads(2);a=torch.tensor([1.,0.],dtype=torch.float64);b=torch.tensor([-1.,1.],dtype=torch.float64);gl=torch.tensor([2.,1.],dtype=torch.float64);wanted=torch.tensor([2.5,1.5],dtype=torch.float64);pa=torch.tensor([.5,.5],dtype=torch.float64);pb=torch.tensor([0.,1.],dtype=torch.float64)
 assert (pa@b).abs()<1e-15 and (pb@a).abs()<1e-15
 dp,ip=block_update(gl,a,b,1.,'BLOCK_PAIR');dc,ic=block_update(gl,a,b,1.,'NORM_L_ID');torch.testing.assert_close(gl+dp,wanted,rtol=1e-14,atol=1e-14);torch.testing.assert_close(gl+dc,gl*(wanted.norm()/gl.norm()),rtol=1e-14,atol=1e-14);assert ip['conflict'] and ic['conflict']
 for mode in ('BLOCK_PAIR','NORM_L_ID'):
  for scale in (1/25000,1000.):
   ds,_=block_update(gl*scale,a*scale,b*scale,1.,mode);base,_=block_update(gl,a,b,1.,mode);torch.testing.assert_close(ds,base*scale,rtol=1e-12,atol=1e-12*scale)
  zero,_=block_update(gl,a,a,1.,mode);assert torch.count_nonzero(zero)==0
  opposite,_=block_update(gl,a,-a,1.,mode);assert torch.count_nonzero(opposite)==0
  empty,_=block_update(gl,a,a*0,1.,mode);assert torch.count_nonzero(empty)==0
  stationary,_=block_update(gl*0,a,b,1.,mode);assert torch.count_nonzero(stationary)==0
 d,_=block_update(gl,a,b,.1,'BLOCK_PAIR');torch.testing.assert_close(gl+d,torch.tensor([2.05,1.05],dtype=gl.dtype),rtol=1e-14,atol=1e-14)
 p=torch.tensor([2.,1.],dtype=torch.float64,requires_grad=True);proxy=((p-p.detach())*dp.detach()).sum();assert proxy==0;torch.testing.assert_close(torch.autograd.grad(proxy,p)[0],dp,rtol=0,atol=0)
 first=torch.arange(768,dtype=torch.float64).reshape(128,6).sin().requires_grad_();second=first.flip(0);y=torch.arange(128)%6;losses,counts=contributions(first,second,y)
 ce=.5*(torch.nn.functional.cross_entropy(first,y,reduction='none',label_smoothing=.01)+torch.nn.functional.cross_entropy(second,y,reduction='none',label_smoothing=.01))
 for c,l,n in zip((1,3),losses,counts):torch.testing.assert_close(l,ce[y==c].sum()/128,rtol=0,atol=0);assert n==int((y==c).sum())
 steps=[s for s in range(1,11101) if active(s)];assert len(steps)==2498 and steps[0]==1112 and steps[-1]==11100 and strength(1112)==.1 and strength(3330)==1.
 negative=0
 for yy in (y-8,y+6):
  try:contributions(first,second,yy)
  except ValueError:negative+=1
  else:raise AssertionError('Hidden/non-source TX accepted')
 try:block_update(gl,a,b,1.,'other')
 except ValueError:negative+=1
 else:raise AssertionError('Unregistered control accepted')
 # Inactive feature leaves must retain None, avoiding accidental AdamW decay.
 p1=torch.nn.Parameter(torch.tensor([2.,1.]));unused=torch.nn.Parameter(torch.ones(2));w=torch.nn.Parameter(torch.ones(2));pars=[('id_backbone.encoder.core.id_backbone.t1.weight',p1),('id_backbone.encoder.core.id_backbone.t2.weight',unused)]
 gs=[(gl.float(),None),(a.float(),None),(b.float(),None)];ds,info=corrected(pars,gs,1.,'BLOCK_PAIR');assert ds[1] is None
 proxy=surrogate(pars,ds);assert proxy==0;gw=torch.autograd.grad(proxy,w,allow_unused=True,retain_graph=True)[0];assert gw is None;proxy.backward();assert unused.grad is None;torch.testing.assert_close(p1.grad,dp.float());assert info['nonzero_blocks']==1
 return dict(status='PASS',protocol_negative_cases=negative,independent_projection_oracle=True,matched_L_identity_norm=True,physical_common_scale_equivariance=True,positive_zero_opposite_cases=True,paired_CE_contributions_exact=True,zero_value_linear_surrogate_gradient=True,calendar_calls=2498,first_step=1112,IQ_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':
 import json
 print(json.dumps(run()))
