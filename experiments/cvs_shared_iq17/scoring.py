"""Source default-head identity counts;no fitted model or feature analysis."""
from .design import FIT,VIEWS

def score(labels,pred):
 if len(labels)!=len(pred) or not labels or any(type(v) is not int or v not in range(6) for v in labels+pred):raise ValueError('Complete visible source sixclass decisions required')
 cm=[[0]*6 for _ in range(6)]
 for y,p in zip(labels,pred):cm[y][p]+=1
 f=[]
 for t in range(6):
  den=sum(cm[t])+sum(row[t] for row in cm);f.append(2*cm[t][t]/den if den else 0.)
 return dict(count=len(labels),accuracy=sum(cm[t][t] for t in range(6))/len(labels),macro_F1=sum(f)/6,confusion=cm)
def all_scores(labels,predictions,rx,day):
 if set(predictions)!=set(VIEWS) or len(labels)!=5040 or any(len(v)!=5040 for v in predictions.values()) or any(len(z)!=5040 for z in (rx,day)):raise ValueError('All source views and metadata required')
 raw=predictions['raw'];correct=[i for i,(p,y) in enumerate(zip(raw,labels)) if p==y];out={}
 for v,p in predictions.items():
  s=score(labels,p)
  def retention(take):
   base=[i for i in take if raw[i]==labels[i]]
   return sum(p[i]==labels[i] for i in base)/len(base) if base else 0.
  s.update(retention=retention(range(5040)),raw_correct_count=len(correct),RX_accuracy={},RX_retention={},TX_retention={})
  for r in FIT:
   take=[i for i,z in enumerate(rx) if z==r];s['RX_accuracy'][r]=sum(p[i]==labels[i] for i in take)/len(take);s['RX_retention'][r]=retention(take)
  for t in range(6):s['TX_retention'][t]=retention([i for i,y in enumerate(labels) if y==t])
  s['by_day']={d:score([labels[i] for i,z in enumerate(day) if z==d],[p[i] for i,z in enumerate(day) if z==d]) for d in (1,2,3)}
  out[v]=s
 return out
