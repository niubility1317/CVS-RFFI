"""Independent reader counterexamples;no IQ/checkpoint/NN or optimizer access."""
import copy,json,math,subprocess,sys
from pathlib import Path
from .analyze import prediction,qualification,decision_transitions,ssl_boundary,pair_formula,ARMS,VIEWS

def run():
 negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (AssertionError,ValueError,KeyError,TypeError):negative+=1
  else:raise AssertionError('Invalid source evidence accepted')
 labels=[];ids=[]
 for t in range(6):
  for day in (1,2,3):
   for s in range(70):labels.append(t);ids.append([1,1,day,s,t*15000+(day-1)*1000+s])
 p=dict(labels=labels,predictions=labels.copy(),physical_ids=ids);s=prediction(p);assert s['accuracy']==s['macro_F1']==1 and s['count']==1260
 for kind in ('short','unknown_y','unknown_p','hidden_y','targetRX','wrongEQ','wrongday','duplicate','unbalanced','bool'):
  z=copy.deepcopy(p)
  if kind=='short':z['predictions'].pop()
  elif kind=='unknown_y':z['labels'][0]=6
  elif kind=='unknown_p':z['predictions'][0]=6
  elif kind=='hidden_y':z['labels'][0]=-1
  elif kind=='targetRX':z['physical_ids'][0][1]=7
  elif kind=='wrongEQ':z['physical_ids'][0][0]=0
  elif kind=='wrongday':z['physical_ids'][0][2]=0
  elif kind=='duplicate':z['physical_ids'][0]=z['physical_ids'][1]
  elif kind=='unbalanced':z['labels'][0]=1
  else:z['predictions'][0]=False
  reject(lambda z=z:prediction(z))
 scores={a:{v:dict(accuracy=(.86 if a=='C_CE' else .85 if a=='Q_ord' else .80) if v=='clean' else .42 if a=='C_CE' else .40 if a=='Q_ord' else .30,macro_F1=.86 if a=='C_CE' else .85 if a=='Q_ord' else .80) for v in VIEWS} for a in ARMS}
 execution={a:dict(seen_V=dict(source_val_accuracy=.90),final_base_L_objective=.2,full_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,total_pair_weight=1.25,As_training_observations=0 if a=='Q0' else 568320) for a in ARMS};execution['C_CE'].update(As_pair_objective='TWO_CE',As_KL_loss_weight=0.,As_KL_gradient_terms=0,As_pair_calls=4440);assert qualification(scores,execution)['qualified']
 for kind in ('clean','cleanF1','low','lowF1','mid','urban','suburban','V','base','Q0low','Q0urban','updates','forwards','As','weight','KLloss','KLgrad','calls'):
  z=copy.deepcopy(scores);e=copy.deepcopy(execution);c=z['C_CE'];r=e['C_CE']
  if kind=='clean':c['clean']['accuracy']=.84
  elif kind=='cleanF1':c['clean']['macro_F1']=.84
  elif kind=='low':c[VIEWS[-1]]['accuracy']=.405
  elif kind=='lowF1':c[VIEWS[-1]]['macro_F1']=.84
  elif kind=='mid':c[VIEWS[-2]]['accuracy']=.39
  elif kind=='urban':
   for v in VIEWS[4:6]:c[v]['accuracy']=.40
  elif kind=='suburban':c[VIEWS[1]]['accuracy']=.39
  elif kind=='V':r['seen_V']['source_val_accuracy']=.89
  elif kind=='base':r['final_base_L_objective']=.6
  elif kind=='Q0low':z['Q0'][VIEWS[-1]]['accuracy']=.38
  elif kind=='Q0urban':
   for v in VIEWS[4:6]:z['Q0'][v]['accuracy']=.415
   z['Q0'][VIEWS[-1]]['accuracy']=.36
  elif kind=='updates':r['full_steps']=4439
  elif kind=='forwards':r['extra_L_forward_packets']-=1
  elif kind=='As':r['As_training_observations']-=1
  elif kind=='weight':r['total_pair_weight']=1.
  elif kind=='KLloss':r['As_KL_loss_weight']=.1
  elif kind=='KLgrad':r['As_KL_gradient_terms']=1
  elif kind=='calls':r['As_pair_calls']=4439
  assert not qualification(z,e)['qualified'],kind;negative+=1
 item=dict(raw_CE=1.,raw_KL=.2,ordinary_CE=.8,ordinary_KL=.1,s_slot_CE=1.2,s_slot_KL=.3,supcon_weight=.05,supcon=.4,extra_pair_CE=.7,extra_pair_KL=.4,As_pair_objective='TWO_CE',As_KL_loss_weight=0.,As_KL_gradient_terms=0,extra_pair_KL_diagnostic_only=True)
 for arm,expected in (('Q0',1.495),('Q_ord',1.495),('C_CE',1.395)):
  b,ex,total=pair_formula(item,arm);assert math.isclose(b,1.22,abs_tol=1e-12) and math.isclose(total,expected,abs_tol=1e-12)
 changed=dict(item,extra_pair_KL=17.);assert pair_formula(changed,'C_CE')==pair_formula(item,'C_CE')
 for key,value in (('As_pair_objective','RDROP'),('As_KL_loss_weight',.1),('As_KL_gradient_terms',1),('extra_pair_KL_diagnostic_only',False)):
  reject(lambda key=key,value=value:pair_formula(dict(item,**{key:value}),'C_CE'))
 reject(lambda:pair_formula(item,'other'))
 ref={'clean':copy.deepcopy(p),'urban':copy.deepcopy(p)};cand=copy.deepcopy(ref);ref['urban']['predictions'][0]=1;cand['urban']['predictions'][2]=1;cand['clean']['predictions'][2]=1
 z=decision_transitions(ref,cand,'urban');assert z['all']['repaired']==z['all']['broken']==1 and z['both_clean_correct']['repaired']==1 and z['both_clean_correct']['broken']==0 and z['clean_correctness_changed']['net_correct']==-1
 wrong=copy.deepcopy(cand);wrong['urban']['physical_ids'].reverse();reject(lambda:decision_transitions(ref,wrong,'urban'))
 ssl=[]
 for step in range(1,4441):
  epoch=(step-1)//222+1;total=(48 if (step-1)%222==0 else 256) if epoch>=14 else 0
  ssl.append(dict(step=step,epoch=epoch,metrics={'train/pseudo_total':total,'train/pseudo_selected':0,'train/pseudo_truth_available':0,'train/pseudo_correct':0,'train/a1_ema_successful_updates':step,'train/loss_unlabeled':0.}))
 assert ssl_boundary(ssl)['U_appearances']==396368
 for step,key,value in ((2886,'train/pseudo_total',256),(2887,'train/pseudo_total',0),(2887,'train/pseudo_truth_available',1),(2887,'train/pseudo_correct',1),(2887,'train/a1_ema_successful_updates',1)):
  before=ssl[step-1]['metrics'][key];ssl[step-1]['metrics'][key]=value;reject(lambda:ssl_boundary(ssl));ssl[step-1]['metrics'][key]=before
 harvest=Path(__file__).with_name('harvest.py');r=subprocess.run([sys.executable,'-S',str(harvest),'--row','Q_sat'],capture_output=True,text=True);assert r.returncode!=0 and 'invalid choice' in r.stderr;negative+=1
 return dict(status='PASS',counterexamples=negative,synthetic_source_predictions=1260,synthetic_SSL_steps=4440,old_and_new_objectives_recounted=True,real_IQ=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
