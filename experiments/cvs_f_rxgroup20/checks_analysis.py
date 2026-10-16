"""Offline prediction legality and metric counterexamples;zero IQ/NN."""
import copy,json,tempfile
from pathlib import Path
from .analyze import metrics,predictions,labeled_TX,VIEWS

def run():
 y=[i//210 for i in range(1260)];ids=[[1,1,d,s,t*15000+(d-1)*1000+s] for t in range(6) for d in (1,2,3) for s in range(70)]
 assert metrics(y,y)['accuracy']==1 and metrics(y,y)['macro_F1']==1
 m=metrics(y,[0]*1260);assert abs(m['accuracy']-1/6)<1e-14 and abs(m['macro_F1']-1/21)<1e-14
 assert all(labeled_TX(pid)==tx for tx,pid in zip(y,ids))
 n=0
 def reject(call):
  nonlocal n
  try:call()
  except ValueError:n+=1;return
  raise AssertionError('Bad prediction protocol accepted')
 reject(lambda:metrics([],[]));reject(lambda:metrics([0],[6]));reject(lambda:metrics([0,1],[0]));reject(lambda:labeled_TX([1,1,1,70,70]));reject(lambda:labeled_TX([1,3,1,0,0]))
 with tempfile.TemporaryDirectory() as root:
  base=Path(root);(base/'source_readouts').mkdir();r=dict(count=1260,accuracy=1.,macro_F1=1.,mean_CE=0.,physical_ids=ids,labels=y,predictions=y,margin_over30=[.5]*1260,head_fit_updates=0)
  def reset():
   for v in VIEWS:(base/'source_readouts'/(v+'.json')).write_text(json.dumps(r))
  reset();scores,data=predictions(base);assert len(scores)==7
  for key,change in [('physical_ids',[[1,3,*p[2:]] for p in ids]),('predictions',[6]+y[1:]),('head_fit_updates',1),('macro_F1',.5),('margin_over30',[.5]*1259)]:
   z=copy.deepcopy(r);z[key]=change;(base/'source_readouts/clean.json').write_text(json.dumps(z));reject(lambda:predictions(base));reset()
 return dict(status='PASS',protocol_counterexamples=n,prediction_records_checked=8820,NN=0,updates=0,real_IQ=0,checkpoint_loads=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
