"""Source-role and fixed-question counterexamples using metadata only."""
import copy,json
from .question import folds,qualify,EXPECTED,FIT,DAYS

def run():
 ids=[];labels=[]
 for r in FIT:
  for d in DAYS:
   for t in range(6):
    for j in range(70):ids.append([1,r,d,j,t*100000+d*1000+j]);labels.append(t)
 f=folds(ids,labels);assert len(f)==12 and all({ids[i][1] for i in q['reference']}==set(FIT)-{q['RX']} and {ids[i][2] for i in q['reference']}==set(DAYS)-{q['day']} for q in f)
 negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except ValueError:negative+=1
  else:raise AssertionError('Invalid source contract accepted')
 for kind in ('short','duplicate','RX1','target','EQ0','day4','hidden_y','unbalanced'):
  z=copy.deepcopy(ids);y=labels.copy()
  if kind=='short':z.pop();y.pop()
  elif kind=='duplicate':z[0]=z[1]
  elif kind=='RX1':z[0][1]=1
  elif kind=='target':z[0][1]=7
  elif kind=='EQ0':z[0][0]=0
  elif kind=='day4':z[0][2]=4
  elif kind=='hidden_y':y[0]=-1
  else:y[0]=1
  reject(lambda z=z,y=y:folds(z,y))
 s={k:dict(accuracy=a,retention=r,RX_accuracy={rx:a for rx in FIT},RX_retention={rx:r for rx in FIT},TX_retention={t:r for t in range(6)}) for k,a,r in [('raw',.999,1.),('direction',.97,.971),('permuted',.999,1.)]};e=dict(EXPECTED,identifiable_folds=12,target_access=False);assert qualify(s,e)['supported_for_separate_followup']
 for kind in ('raw','gap','transfer','dir_ret','dir_RX','dir_TX','perm_ret','perm_RX','identifiable','budget','target'):
  z=copy.deepcopy(s);a=copy.deepcopy(e)
  if kind=='raw':z['raw']['accuracy']=.97
  elif kind=='gap':z['direction']['accuracy']=.99
  elif kind=='transfer':z['direction']['RX_accuracy'][3]=z['direction']['RX_accuracy'][4]=.999
  elif kind=='dir_ret':z['direction']['retention']=.94
  elif kind=='dir_RX':z['direction']['RX_retention'][3]=.94
  elif kind=='dir_TX':z['direction']['TX_retention'][0]=.94
  elif kind=='perm_ret':z['permuted']['retention']=.98
  elif kind=='perm_RX':z['permuted']['RX_retention'][3]=.97
  elif kind=='identifiable':a['identifiable_folds']=11
  elif kind=='budget':a['total_forward_packets']-=1
  else:a['target_access']=True
  assert not qualify(z,a)['supported_for_separate_followup'];negative+=1
 return dict(status='PASS',counterexamples=negative,synthetic_metadata_records=5040,folds=12,allTX_held_RXday_not_class_specific=True,real_IQ=0,NN=0,updates=0,checkpoint_loads=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
