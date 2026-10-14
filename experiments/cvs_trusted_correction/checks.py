import torch
from .policy import select,loss,weak_view,entropy
def run():
 n=128;t=torch.zeros(n,6);t[:,0]=12;second=t.clone();s=t.clone();s[:8,1]=3.5;s[:8,0]=3;base=torch.ones(n,dtype=torch.bool)
 # Most already-agree cases need nonzero CE mass to fund a bounded correction.
 s[8:,0]=3
 pseudo,m,f,detail=select(t,second,s,base,True)
 assert f['candidate_count']==8 and f['admitted_CE_sum']<=f['CE_budget']+1e-10 and 0<f['admitted_count']<=f['count_budget']
 assert torch.equal(m['selected'],m['original']|m['correction_admitted']) and not (m['correction_admitted']&m['original']).any()
 _,old,_,_=select(t,second,s,base,False);assert torch.equal(old['selected'],old['original'])
 assert torch.equal(loss(s,pseudo,m['selected']),torch.nn.functional.cross_entropy(s[m['selected']],pseudo[m['selected']]))
 # Correlated weak views do not override unstable/disagreeing teacher targets.
 second[:8,0]=0;second[:8,2]=12
 _,unstable,_,_=select(t,second,s,base,True);assert not unstable['correction_candidate'].any()
 # No original accepted mass means no correction bootstrap or implicit quota.
 _,empty,ff,_=select(t,t,s,torch.zeros(n,dtype=torch.bool),True);assert not empty['selected'].any() and ff['admitted_count']==0
 p=torch.tensor([4,1,5,0,3,2]);perm=torch.argsort(p);pp,mm,_,_=select(t[:,perm],t[:,perm],s[:,perm],base,True)
 assert torch.equal(pp,p[pseudo]) and torch.equal(mm['selected'],m['selected'])
 bad=0
 try:select(t[:,:5],t[:,:5],s[:,:5],base,True)
 except ValueError:bad+=1
 else:raise AssertionError('Wrong classes accepted')
 # Entropy can affect rejected samples; switching it off gives exactly no term.
 z=torch.randn(7,6,requires_grad=True);ent=entropy(z).mean();grad=torch.autograd.grad(ent,z)[0];assert torch.isfinite(grad).all() and grad.norm()>0
 x=torch.randn(4,2,256);state=torch.get_rng_state().clone();a=weak_view(x,783);b=weak_view(x,783);assert torch.equal(a,b) and torch.equal(state,torch.get_rng_state())
 return dict(status='PASS',same_accepted_mean=True,uniform_weights=True,bounded_correction=True,teacher_instability_rejected=True,no_bootstrap_quota=True,class_permutation=True,private_weak_RNG=True,protocol_negatives=bad,hidden_truth_arguments=False)
if __name__=='__main__':print(run())
