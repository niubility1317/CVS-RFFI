"""Offline source prediction/coverage/qualification counterexamples;zeroIQ/NN."""
import copy,json,tempfile
from pathlib import Path
from .analyze import metrics,predictions,labeled_TX,VIEWS,ARMS,active,expected_packets,counter_cells,qualification

def run():
 y=[i//210 for i in range(1260)];ids=[[1,1,dd,s,t*15000+(dd-1)*1000+s] for t in range(6) for dd in (1,2,3) for s in range(70)]
 assert metrics(y,y)['accuracy']==metrics(y,y)['macro_F1']==1
 m=metrics(y,[0]*1260);assert abs(m['accuracy']-1/6)<1e-14 and abs(m['macro_F1']-1/21)<1e-14
 assert all(labeled_TX(pid)==tx for tx,pid in zip(y,ids));n=0
 def reject(fn):
  nonlocal n
  try:fn()
  except ValueError:n+=1;return
  raise AssertionError('Illegal source analysis input accepted')
 reject(lambda:metrics([],[]));reject(lambda:metrics([0],[6]));reject(lambda:metrics([0,1],[0]));reject(lambda:labeled_TX([1,1,1,70,70]));reject(lambda:labeled_TX([1,3,1,0,0]))
 with tempfile.TemporaryDirectory() as root:
  base=Path(root);(base/'source_readouts').mkdir();r=dict(count=1260,accuracy=1.,macro_F1=1.,mean_CE=0.,physical_ids=ids,labels=y,predictions=y,margin_over30=[.5]*1260,head_fit_updates=0)
  def reset():
   for v in VIEWS:(base/'source_readouts'/(v+'.json')).write_text(json.dumps(r),encoding='utf8')
  reset();scores,data=predictions(base);assert len(scores)==7
  for key,val in [('physical_ids',[[1,3,*p[2:]] for p in ids]),('predictions',[6]+y[1:]),('head_fit_updates',1),('macro_F1',.5),('margin_over30',[.5]*1259),('labels',[(v+1)%6 for v in y])]:
   z=copy.deepcopy(r);z[key]=val;(base/'source_readouts/clean.json').write_text(json.dumps(z),encoding='utf8');reject(lambda:predictions(base));reset()
 good=dict(TX=0,RX=3,day=1,scene=VIEWS[-1],state='BLOCKED',lock='unlocked',count=2);assert sum(counter_cells([good]).values())==2
 for key,val in [('TX',6),('RX',1),('day',0),('scene','weak'),('state','invalid'),('lock','invalid'),('count',0)]:
  z=copy.deepcopy(good);z[key]=val;reject(lambda:counter_cells([z]))
 reject(lambda:counter_cells([good,good]));assert sum(active('EARLY7',e) for e in range(1,21))==sum(active('LATE7',e) for e in range(1,21))==7
 records={a:dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,As_generated_packets=568320,As_training_observations=expected_packets(a),timing_steps=4440,head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[],seen_V=dict(source_val_accuracy=.9)) for a in ARMS};scores={a:{v:dict(accuracy=.82 if a=='LATE7' and v=='clean' else .5 if a=='LATE7' else .8 if v=='clean' else .4,macro_F1=.82 if a=='LATE7' else .8) for v in VIEWS} for a in ARMS};checks,comp=qualification(scores,records);assert all(checks.values());counter=0
 for ref,view,value in [('RAW','clean',.85),('ALL','clean',.85),('EARLY7','clean',.85),('RAW',VIEWS[-1],.52),('ALL',VIEWS[-1],.52),('EARLY7',VIEWS[-1],.52)]:
  z=copy.deepcopy(scores);z[ref][view]['accuracy']=value;assert not all(qualification(z,records)[0].values());counter+=1
 z=copy.deepcopy(records);z['LATE7']['As_training_observations']=568320;assert not all(qualification(scores,z)[0].values());counter+=1
 return dict(status='PASS',protocol_counterexamples=n,scientific_counterexamples=counter,prediction_records_checked=8820,NN=0,updates=0,real_IQ=0,checkpoint_loads=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
