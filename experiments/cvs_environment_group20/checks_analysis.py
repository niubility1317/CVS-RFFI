"""Focused offline source predictions/science/private-stream negatives;zero NN/IQ/CP."""
import copy,json,tempfile
from pathlib import Path
from .analyze import ARMS,VIEWS,grouping,metrics,predictions,qualify,stable_seed
def run():
 count=0
 def reject(fn,part):
  nonlocal count
  try:fn()
  except ValueError as e:
   if part not in str(e):raise
   count+=1;return
  raise AssertionError('Invalid offline input accepted')
 y=[i//210 for i in range(1260)];ids=[[1,6,d,s,t*15000+9000+(d-1)*1000+s] for t in range(6) for d in (1,2,3) for s in range(70)]
 assert metrics(y,y)['accuracy']==metrics(y,y)['macro_F1']==1.
 reject(lambda:metrics([],[]),'decisions');reject(lambda:metrics([0],[6]),'decisions');reject(lambda:metrics([0,1],[0]),'decisions')
 with tempfile.TemporaryDirectory() as tmp:
  b=Path(tmp);(b/'source_readouts').mkdir();p=dict(count=1260,accuracy=1.,macro_F1=1.,mean_CE=0.,physical_ids=ids,labels=y,predictions=y,margin_over30=[.5]*1260,head_fit_updates=0)
  def reset():
   for v in VIEWS:(b/'source_readouts'/(v+'.json')).write_text(json.dumps(p),encoding='utf8')
  reset();s,d=predictions(b);assert len(s)==7
  for field,value,why in [('physical_ids',[[1,1,*q[2:]] for q in ids],'role'),('predictions',[6]+y[1:],'decisions'),('margin_over30',[.5]*1259,'margins'),('head_fit_updates',1,'role')]:
   bad=copy.deepcopy(p);bad[field]=value;(b/'source_readouts/clean.json').write_text(json.dumps(bad),encoding='utf8');reject(lambda:predictions(b),why);reset()
 scores={a:{v:dict(accuracy=.82 if a=='PROP_GROUP' and v=='clean' else .48 if a=='PROP_GROUP' else .8 if v=='clean' else .4,macro_F1=.82 if a=='PROP_GROUP' else .8) for v in VIEWS} for a in ARMS}
 records={a:dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,group_generated=568320,supcon_applied=3330,As_training_observations=0 if a=='RAW' else 568320,final_base_L_objective=.2,held_RX=6,fit_RX=[1,3,4,8],head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[],seen_V=dict(source_val_accuracy=.9)) for a in ARMS}
 assert all(qualify(scores,records)[0].values());science=0
 for ref in ARMS[:-1]:
  for view,field in [('clean','accuracy'),('clean','macro_F1'),(VIEWS[-1],'accuracy'),(VIEWS[1],'accuracy')]:
   s=copy.deepcopy(scores);s[ref][view][field]=.95;assert not all(qualify(s,records)[0].values());science+=1
  r=copy.deepcopy(records);r[ref]['seen_V']['source_val_accuracy']=.92;assert not all(qualify(scores,r)[0].values());science+=1
 for field,val in [('actual_steps',4439),('L_forward_packets',4546559),('group_generated',0),('As_training_observations',0),('final_base_L_objective',.6),('held_RX',1),('fit_RX',[3,4,6,8]),('target_access',True),('U_truth_read',True),('checkpoint_sources',['foreign'])]:
  r=copy.deepcopy(records);r['PROP_GROUP'][field]=val;assert not all(qualify(scores,r)[0].values());science+=1
 pids=[[1,1,1,i%70,i%70] for i in range(128)];scene=VIEWS[1];ns='train/epoch1/step1';packets=[(1,1,1,i%70,i%70) for i in range(128)];seeds=[stable_seed(20261016101,ns,scene,p) for p in packets];env=stable_seed(20261016101,ns,scene,'environment',(1,1,1))
 row=dict(group_shadow_generated=128,group_policy='PROP_GROUP',extra_input='PROP_GROUP',extra_As_packets=128,extra_BN_batch_updates=0,extra_raw_MSE=1.,shadow_raw_MSE=1.,group_session=['rx1:day1:eq1']*128,group_scene=[scene]*128,group_state=['LOS']*128,group_lock=['locked']*128,group_environment_seed=[env]*128,group_packet_seed=seeds,group_noise_stream_seed=[stable_seed(s,'noise') for s in seeds],group_kernel_energy=[1.]*128,group_packet_residual_hz=[1.]*128,group_packet_coarse_error_hz=[1.]*128,group_packet_phase=[1.]*128,group_packet_tracking_z=[1.]*128,group_session_count=1,group_scene_counts={scene:128},group_state_counts={'LOS':128},group_lock_counts={'locked':128})
 for field,value,why in [('group_packet_seed',0,'namespace'),('group_environment_seed',0,'namespace'),('group_noise_stream_seed',0,'namespace'),('group_state','BLOCKED','shared'),('group_kernel_energy',2.,'shared'),('group_packet_phase',float('nan'),'Finite')]:
  z=copy.deepcopy(row);z[field][1]=value;reject(lambda:grouping(Path('unused'),'PROP_GROUP',[dict(role='L',ids=pids)],[z],[]),why)
 z=copy.deepcopy(row);z['extra_As_packets']=127;reject(lambda:grouping(Path('unused'),'PROP_GROUP',[dict(role='L',ids=pids)],[z],[]),'All128L')
 z=copy.deepcopy(pids);z[0]=[1,6,1,0,9000];reject(lambda:grouping(Path('unused'),'PROP_GROUP',[dict(role='L',ids=z)],[row],[]),'Legal')
 return dict(status='PASS',protocol_negatives=count,scientific_negatives=science,positive_sevenview_fixture=8820,IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(run()))
