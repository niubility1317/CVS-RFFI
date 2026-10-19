"""Numeric group-risk and legal pair oracle; no IQ or optimizer."""
import torch

def run():
 torch.set_num_threads(2)
 try:
  from .loss import masks,anchor_loss,risk,penalty
 except ModuleNotFoundError as e:
  raise AssertionError('Missing source-only paired class/RX upper-half feature risk') from e
 y=torch.tensor([1,1,1,1,3,3,3,3]);rx=torch.tensor([1,1,3,3,1,1,3,3]);valid=torch.ones(8,dtype=torch.bool)
 vector=torch.arange(8,dtype=torch.float64,requires_grad=True)
 mean,info=risk(vector,y,rx,valid,'MEAN_WEAK_RX');tail,selected=risk(vector,y,rx,valid,'TAIL_WEAK_RX')
 assert torch.equal(mean,torch.tensor(3.5,dtype=mean.dtype)) and torch.equal(tail,torch.tensor(4.,dtype=tail.dtype))
 gm=torch.autograd.grad(mean,vector,retain_graph=True)[0];gt=torch.autograd.grad(tail,vector,retain_graph=True)[0]
 torch.testing.assert_close(gm,torch.full_like(vector,1/8),rtol=0,atol=0)
 torch.testing.assert_close(gt,torch.tensor([0,.25,0,.25,0,.25,0,.25],dtype=gt.dtype),rtol=0,atol=0)
 assert info['eligible'] and selected['eligible'] and info['selected_anchors']==8 and selected['selected_anchors']==4
 # Select physical averaged risks, rather than independently noisy views.
 a=torch.tensor([0.,6.,0.,6.,0.,6.,0.,6.]);b=6-a
 paired,_=risk((a+b)/2,y,rx,valid,'TAIL_WEAK_RX')
 if paired!=3:raise AssertionError('Mining individual stochastic views rather than physical paired risks')
 # Equal TX/RX groups, even when one cell has twice as many occurrences.
 ext_y=torch.cat((y,torch.ones(2,dtype=torch.long)));ext_rx=torch.cat((rx,torch.tensor([1,1])));ext=torch.cat((vector.detach(),torch.tensor([0.,0.],dtype=vector.dtype)))
 balanced,_=risk(ext,ext_y,ext_rx,torch.ones(10,dtype=torch.bool),'MEAN_WEAK_RX')
 torch.testing.assert_close(balanced,torch.tensor(3.4375,dtype=balanced.dtype),rtol=0,atol=0)
 yy=torch.tensor([1,1,1,3,3,3,0,0,0,1,3,0]);rr=torch.tensor([1,3,4,1,3,4,1,3,4,1,1,1]);z=torch.arange(60,dtype=torch.float64).reshape(12,5).sin().requires_grad_()
 pos,den,ok=masks(yy,rr)
 assert not bool((pos&(rr[:,None]==rr[None,:])).any())
 negatives=den&~pos
 assert not bool((negatives&((rr[:,None]!=rr[None,:])|(yy[:,None]==yy[None,:]))).any())
 each,actual=anchor_loss(z,yy,rr)
 unit=torch.nn.functional.normalize(z,dim=1,eps=1e-4);logits=unit@unit.T/.2
 oracle=torch.stack([torch.logsumexp(logits[i][den[i]],0)-logits[i][pos[i]].mean()-pos[i].sum().to(z.dtype).log() for i in range(12)])
 torch.testing.assert_close(each,oracle,atol=1e-12,rtol=1e-12)
 if not torch.equal(actual,ok):raise AssertionError('Anchor validity differs')
 grad=torch.autograd.grad(each.sum(),z)[0]
 if not torch.isfinite(grad).all() or grad.norm()<=0:raise AssertionError('Contrastive feature gradient missing')
 zero,skip=risk(vector,y*0+1,rx,valid,'TAIL_WEAK_RX')
 if skip['eligible'] or zero!=0 or torch.count_nonzero(torch.autograd.grad(zero,vector)[0]):raise AssertionError('Missing weak TX should skip with finite zero gradient')
 lone=torch.randn(4,3,dtype=torch.float64,requires_grad=True);no,_=anchor_loss(lone,torch.arange(4),torch.ones(4,dtype=torch.long))
 no.sum().backward()
 assert torch.isfinite(no).all() and torch.count_nonzero(no)==0 and torch.isfinite(lone.grad).all() and torch.count_nonzero(lone.grad)==0
 # Exercise the integration boundary: average two actual-view score vectors
 # BEFORE each class/RX tail selection, rather than averaging mined views.
 torch.manual_seed(712);ly=torch.arange(128)%6;lr=torch.tensor((1,3,4,6,8))[torch.arange(128)%5]
 first={k:torch.randn(128,c,4,dtype=torch.float64,requires_grad=True) for k,c in [('time_down',48),('t2.pool',96)]}
 second={k:torch.randn_like(v,requires_grad=True) for k,v in first.items()}
 actual_pair,_=penalty(first,second,ly,lr,'TAIL_WEAK_RX');wanted=[];wrong=[]
 for level in ('time_down','t2.pool'):
  scores=[]
  for maps in (first,second):
   h=maps[level];v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);desc=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4)
   score,valid_now=anchor_loss(desc,ly,lr);scores.append(score)
  wanted.append(risk((scores[0]+scores[1])/2,ly,lr,valid_now,'TAIL_WEAK_RX')[0])
  wrong.append((risk(scores[0],ly,lr,valid_now,'TAIL_WEAK_RX')[0]+risk(scores[1],ly,lr,valid_now,'TAIL_WEAK_RX')[0])/2)
 torch.testing.assert_close(actual_pair,torch.stack(wanted).mean(),rtol=1e-12,atol=1e-12)
 if abs(float((actual_pair-torch.stack(wrong).mean()).detach()))<1e-5:raise AssertionError('Fixture does not distinguish averaging then selecting')
 negative=0
 for bady,badrx in [(yy-7,rr),(yy,rr*0+2)]:
  try:masks(bady,badrx)
  except ValueError:negative+=1
  else:raise AssertionError('Hidden labels or foreign RX accepted')
 return dict(status='PASS',numeric_tail_value=float(tail.detach()),numeric_mean_value=float(mean.detach()),numeric_tail_gradient=gt.tolist(),paired_physical_risk_selection=True,source_group_balance=True,pair_formula_oracle=True,no_pair_zero_gradient=True,protocol_negative_cases=negative,IQ_reads=0,target_reads=0)

if __name__=='__main__':
 import json
 print(json.dumps(run()))
