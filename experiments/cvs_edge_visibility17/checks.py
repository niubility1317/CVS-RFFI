"""Focused source/config/input negatives and visibility invariants;no real IQ/model."""
import copy,json
import torch
from . import design as d
from .window import apply,qualify

def run():
 rejects=0;windows=0
 def reject(fn):
  nonlocal rejects
  try:fn()
  except (ValueError,TypeError,KeyError):rejects+=1;return
  raise AssertionError('Illegal case accepted')
 def op(x,s):
  nonlocal windows
  y,info=apply(x,s);windows+=len(x);return y,info
 c=d.config();d.validate(c)
 for key,value in [('edge_support',8),('zero_samples',0),('taper_samples',4),('reference',[64,176]),('fit_RX',[1,3,4,6]),('L_unique',6300),('target_access',True),('U_truth_read',True),('automatic_training',True),('optimizer_updates',1),('head_fit_updates',1),('total_NN_packets',15120)]:
  bad=copy.deepcopy(c);bad[key]=value;reject(lambda:d.validate(bad))
 reject(lambda:apply(torch.ones(2,256),'left'));reject(lambda:apply(torch.ones(2,2,255),'left'));reject(lambda:apply(torch.ones(0,2,256),'left'));reject(lambda:apply(torch.ones(2,2,256,dtype=torch.int64),'left'));reject(lambda:apply(torch.ones(2,2,256,dtype=torch.complex64),'left'));reject(lambda:apply(torch.full((2,2,256),float('nan')),'left'));reject(lambda:apply(torch.ones(2,2,256),'middle'))
 x=torch.randn(12,2,256,dtype=torch.float64,generator=torch.Generator().manual_seed(2026101717));saved=x.clone()
 for side in ('left','right'):
  rng=torch.get_rng_state().clone();y,info=op(x,side);assert torch.equal(rng,torch.get_rng_state()) and torch.equal(x,saved)
  assert torch.equal(x[:,:,80:160],y[:,:,80:160]) and all(info['reference_unchanged']) and all(info['outside_support_unchanged']) and max(info['power_increase'])<=0
  if side=='left':assert torch.count_nonzero(y[:,:,:8])==0 and torch.equal(y[:,:,16:],x[:,:,16:])
  else:assert torch.count_nonzero(y[:,:,-8:])==0 and torch.equal(y[:,:,:240],x[:,:,:240])
  scaled,_=op(7*x,side);assert torch.allclose(scaled,7*y,rtol=1e-13,atol=1e-13)
  rotated=torch.stack((-x[:,1],x[:,0]),dim=1);yr,_=op(rotated,side);assert torch.equal(yr,torch.stack((-y[:,1],y[:,0]),dim=1))
  v=x.clone().requires_grad_();z,_=op(v,side);gradient=torch.autograd.grad(z.sum(),v)[0];assert torch.equal(gradient[:,:,80:160],torch.ones_like(gradient[:,:,80:160]))
  if side=='left':assert torch.count_nonzero(gradient[:,:,:8])==0
  else:assert torch.count_nonzero(gradient[:,:,-8:])==0
 left,_=op(x,'left');right,_=op(x.flip(-1),'right');assert torch.equal(left.flip(-1),right)
 raw=dict(accuracy=.95,macro_F1=.95,by_RX={str(r):dict(accuracy=.95) for r in d.FIT},by_TX={str(t):dict(accuracy=.95) for t in range(6)},by_day={str(t):dict(accuracy=.95) for t in (1,2,3)})
 good=copy.deepcopy(raw);good['raw_correct_retention']=.999
 stats=dict(relative_IQ_change=[.1]*12,power_increase=[0.]*12,reference_unchanged=[True]*12,outside_support_unchanged=[True]*12)
 assert qualify(raw,good,stats)['decision']=='PASS'
 for key,value in [('accuracy',.94),('macro_F1',.94),('raw_correct_retention',.98)]:
  bad=copy.deepcopy(good);bad[key]=value;assert qualify(raw,bad,stats)['decision']=='FAIL'
 bad=copy.deepcopy(good);bad['by_RX']['6']['accuracy']=.93;assert qualify(raw,bad,stats)['decision']=='FAIL'
 bad=copy.deepcopy(stats);bad['relative_IQ_change']=[0.]*12;assert qualify(raw,good,bad)['decision']=='FAIL'
 bad=copy.deepcopy(stats);bad['relative_IQ_change'][0]=.36;assert qualify(raw,good,bad)['decision']=='FAIL'
 bad=copy.deepcopy(stats);bad['reference_unchanged'][0]=False;assert qualify(raw,good,bad)['decision']=='FAIL'
 return dict(status='PASS',negative_cases=rejects,scientific_counterexamples=7,synthetic_IQ_window_applications=windows,source_IQ_reads=0,NN=0,updates=0,head_fit_updates=0,target_access=False,reference_outside_exact=True,global_gain_phase_equivariant=True,input_RNG_unmodified=True)
if __name__=='__main__':print(json.dumps(run()))
