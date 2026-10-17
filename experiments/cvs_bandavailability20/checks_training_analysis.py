"""Focused source-only offline scoring and coverage counterexamples; zero NN/IQ."""
import copy,json,random,tempfile
from pathlib import Path
from .analyze_training import ARMS,VIEWS,availability,metrics,predictions,qualify

def run():
 count=0
 def reject(fn,text):
  nonlocal count
  try:fn()
  except ValueError as e:
   if text not in str(e):raise
   count+=1;return
  raise AssertionError('Invalid offline input was accepted')
 y=[i//210 for i in range(1260)];ids=[[1,1,d,s,t*15000+(d-1)*1000+s] for t in range(6) for d in (1,2,3) for s in range(70)];assert metrics(y,y)['accuracy']==metrics(y,y)['macro_F1']==1.
 reject(lambda:metrics([],[]),'decisions');reject(lambda:metrics([0],[6]),'decisions');reject(lambda:metrics([0,1],[0]),'decisions')
 with tempfile.TemporaryDirectory() as tmp:
  b=Path(tmp);(b/'source_readouts').mkdir();p=dict(count=1260,accuracy=1.,macro_F1=1.,mean_CE=0.,physical_ids=ids,labels=y,predictions=y,margin_over30=[.5]*1260,head_fit_updates=0)
  def reset():
   for view in VIEWS:(b/'source_readouts'/(view+'.json')).write_text(json.dumps(p),encoding='utf8')
  reset();s,d=predictions(b);assert len(s)==7
  for field,value,why in [('physical_ids',[[1,3,*q[2:]] for q in ids],'role'),('predictions',[6]+y[1:],'decisions'),('margin_over30',[.5]*1259,'margins'),('head_fit_updates',1,'role')]:
   bad=copy.deepcopy(p);bad[field]=value;(b/'source_readouts/clean.json').write_text(json.dumps(bad),encoding='utf8');reject(lambda:predictions(b),why);reset()
 scores={a:{v:dict(accuracy=.82 if a=='MASK' and v=='clean' else .2 if a=='MASK' else .8 if v=='clean' else .4,macro_F1=.82 if a=='MASK' else .8) for v in VIEWS} for a in ARMS};records={a:dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,availability_generated=568320,supcon_applied=3330,final_base_L_objective=.2,head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[],seen_V=dict(source_val_accuracy=.9)) for a in ARMS}
 assert all(qualify(scores,records)[0].values()) # Independent Ag clean-only; practical decline is not joint success.
 science=0
 for ref in ARMS[:-1]:
  for field in ('accuracy','macro_F1'):
   s=copy.deepcopy(scores);s[ref]['clean'][field]=.9;assert not all(qualify(s,records)[0].values());science+=1
  r=copy.deepcopy(records);r[ref]['seen_V']['source_val_accuracy']=.92;assert not all(qualify(scores,r)[0].values());science+=1
 for field,val in [('actual_steps',4439),('L_forward_packets',4546559),('availability_generated',0),('final_base_L_objective',.6),('target_access',True),('U_truth_read',True)]:
  r=copy.deepcopy(records);r['MASK'][field]=val;assert not all(qualify(scores,r)[0].values());science+=1
 rng=random.Random(202610171952);ks=[rng.randrange(1,7) for _ in range(128)]
 row=dict(availability_pairs=ks,availability_cap=[1.]*128,availability_relative_RMS=[.05]*128,availability_matched_gain=[.9]*128,availability_power_match_error=[0.]*128,availability_shadow_generated=128,extra_input='MASK',extra_As_packets=0,extra_BN_batch_updates=0);pids=[[1,3,1,i%70,3000+i%70] for i in range(128)]
 for field,bad in [('availability_cap',1.1),('availability_relative_RMS',.2),('availability_power_match_error',.01)]:
  z=copy.deepcopy(row);z[field][0]=bad;reject(lambda:availability(Path('unused'),'MASK',[dict(role='L',ids=pids)],[z],[]),'bounded')
 z=copy.deepcopy(row);z['availability_pairs'][0]=ks[0]%6+1;reject(lambda:availability(Path('unused'),'MASK',[dict(role='L',ids=pids)],[z],[]),'stream')
 z=copy.deepcopy(pids);z[0]=[1,1,1,0,0];reject(lambda:availability(Path('unused'),'MASK',[dict(role='L',ids=z)],[row],[]),'Held')
 return dict(status='PASS',protocol_negatives=count,scientific_negatives=science,positive_saved_prediction_fixture=8820,standalone_Ag_not_joint=True,IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(run()))
