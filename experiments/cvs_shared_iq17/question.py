"""Fixed prospective source question;no training or target candidate selection."""
from collections import Counter
import math
FIT=(3,4,6,8)
DAYS=(1,2,3)
EXPECTED=dict(folds=12,reference_packets_per_fold=2520,held_packets_per_fold=420,reference_forward_packets=60480,input_backward_packets=30240,input_backward_calls=240,source_eval_packets=15120,noquery_smoke_packets=2,total_forward_packets=75602,optimizer_updates=0)

def folds(ids,labels):
 if len(ids)!=5040 or len(labels)!=5040 or any(not isinstance(z,(list,tuple)) or len(z)!=5 or any(type(v) is not int for v in z) or z[0]!=1 or z[1] not in FIT or z[2] not in DAYS for z in ids) or len({tuple(z) for z in ids})!=5040:raise ValueError('Only complete unique visible fit L5040 EQ1 allowed')
 if any(type(y) is not int or y not in range(6) for y in labels) or Counter((y,z[1],z[2]) for z,y in zip(ids,labels))!=Counter({(t,r,d):70 for t in range(6) for r in FIT for d in DAYS}):raise ValueError('All6TX/4fitRX/3day/70visibleL cells required')
 result=[]
 for rx in FIT:
  for day in DAYS:
   ref=[i for i,z in enumerate(ids) if z[1]!=rx and z[2]!=day];held=[i for i,z in enumerate(ids) if z[1]==rx and z[2]==day]
   assert len(ref)==2520 and len(held)==420 and not set(ref)&set(held)
   result.append(dict(RX=rx,day=day,reference=ref,held=held,seed=202610178001+100*rx+day,control_seed=202610178002+100*rx+day))
 assert Counter(i for v in result for i in v['held'])==Counter(range(5040))
 return result

def qualify(scores,execution):
 """Sensitivity only;never a DG/identity success declaration."""
 if set(scores)!=set(('raw','direction','permuted')):raise ValueError('Three complete fixed source views required')
 for v in scores.values():
  if set(v['RX_accuracy'])!=set(FIT) or set(v['RX_retention'])!=set(FIT) or set(v['TX_retention'])!=set(range(6)):raise ValueError('All fixed RX/TX strata required')
  values=[v['accuracy'],v['retention'],*v['RX_accuracy'].values(),*v['RX_retention'].values(),*v['TX_retention'].values()]
  if any(type(z) not in (int,float) or not math.isfinite(z) or not 0<=z<=1 for z in values):raise ValueError('Complete finite source metrics required')
 a,b,c=(scores[k] for k in ('raw','direction','permuted'));gap=100*(c['accuracy']-b['accuracy']);perrx={r:100*(c['RX_accuracy'][r]-b['RX_accuracy'][r]) for r in FIT};eps=1e-10
 conditions=dict(raw_accuracy=a['accuracy']>=.98-eps,directional_gap=gap>=2.-eps,transfer_RX_count=sum(v>=1.-eps for v in perrx.values())>=3,directional_retention=b['retention']>=.95-eps,directional_each_RX_retention=min(b['RX_retention'].values())>=.95-eps,directional_each_TX_retention=min(b['TX_retention'].values())>=.95-eps,permuted_retention=c['retention']>=.99-eps,permuted_each_RX_retention=min(c['RX_retention'].values())>=.98-eps,all12_directions_identifiable=execution['identifiable_folds']==12,complete_budget=all(execution[k]==v for k,v in EXPECTED.items()),no_target=execution['target_access'] is False)
 return dict(supported_for_separate_followup=all(conditions.values()),conditions=conditions,directional_accuracy_loss_vs_permuted_pp=gap,per_RX_gap_pp=perrx,identity_preservation_proved=False,Ag_efficacy_proved=False,joint_goal_complete=False,on_fail='Close fixed shared-direction question;no epsilon/xi/peak/iteration/model/RX/day/threshold/seed rescue')
