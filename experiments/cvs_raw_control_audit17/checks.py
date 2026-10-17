"""Bounded input/method safety and exact-state comparison counterexamples."""
import copy,json
import torch
from . import design as d
from .states import checked,compare,HEAD
from .audit import metric,prefix

def reject(f):
 try:f()
 except (ValueError,KeyError,TypeError):return
 raise AssertionError('Invalid input accepted')
def main():
 n=0
 for k,v in [('target_access',True),('new_network_forwards',1),('new_model_epochs',20),('new_optimizer_updates',1),('candidate_promotion',True),('sources',d.sources()[:-1]),('source_only',False)]:
  q=d.config();q[k]=v;reject(lambda:d.validate(q));n+=1
 x={f'parameter{i}':torch.zeros(1) for i in range(205)};x[HEAD]=torch.ones(6,160);assert checked(x) is x and compare(x,x)['tensors_equal']
 y=copy.deepcopy(x);y['parameter0'][0]=1.;z=compare(x,y);assert not z['tensors_equal'] and z['max_absolute_difference']==1.
 y=copy.deepcopy(x);y['parameter0']=y['parameter0'].double();assert not compare(x,y)['shapes_dtypes_equal']
 for kind in ('missing','head','nan'):
  y=copy.deepcopy(x)
  if kind=='missing':del y['parameter0']
  elif kind=='head':y[HEAD]=torch.ones(6,159)
  else:y['parameter0'][0]=float('nan')
  reject(lambda:checked(y));n+=1
 labels=[i for i in range(6) for j in range(210)];assert metric(labels,labels)['accuracy']==1.
 for y,p in [(labels[:-1],labels[:-1]),(labels,[6]*1260),([0]*1260,labels)]:reject(lambda:metric(y,p));n+=1
 assert prefix([1,2,3],[1,2,4])==3 and prefix([1.,2.],[1.+1e-8,2.])==1 and prefix([1.,2.],[1.+1e-8,2.],threshold=1e-5) is None
 assert len(d.sources())==5 and len({x['key'] for x in d.sources()})==5 and 'final_ssdg.pth' not in d.FILES and 'initial_smoke.pt' in d.FILES and all('source_readouts' not in x or x.endswith('.json') for x in d.FILES)
 print(json.dumps(dict(status='PASS',protocol_negatives=n,synthetic_state_tensors=206,state_difference_controls=3,realIQ=0,realCP=0,NN=0,updates=0,target=False)))
if __name__=='__main__':main()
