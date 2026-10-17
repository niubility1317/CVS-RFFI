"""Offline parser/scientific-boundary counterexamples;0 physicalIQ/NN/updates."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
from .analyze import prediction,routing,qualification,decision_transitions,ARMS,VIEWS

def run():
 negative=0
 def reject(f):
  nonlocal negative
  try:f()
  except (AssertionError,ValueError,KeyError,TypeError):negative+=1
  else:raise AssertionError('Invalid source evidence accepted')
 labels=[];ids=[]
 for t in range(6):
  for d in (1,2,3):
   for s in range(70):labels.append(t);ids.append([1,1,d,s,t*15000+(d-1)*1000+s])
 p=dict(labels=labels,predictions=labels.copy(),physical_ids=ids);s=prediction(p);assert s['accuracy']==s['macro_F1']==1 and s['count']==1260
 for kind in ('short','unknown_y','unknown_p','hidden_y','targetRX','wrongEQ','wrongday','duplicate','unbalanced','bool'):
  q=copy.deepcopy(p)
  if kind=='short':q['predictions'].pop()
  elif kind=='unknown_y':q['labels'][0]=6
  elif kind=='unknown_p':q['predictions'][0]=6
  elif kind=='hidden_y':q['labels'][0]=-1
  elif kind=='targetRX':q['physical_ids'][0][1]=7
  elif kind=='wrongEQ':q['physical_ids'][0][0]=0
  elif kind=='wrongday':q['physical_ids'][0][2]=0
  elif kind=='duplicate':q['physical_ids'][0]=q['physical_ids'][1]
  elif kind=='unbalanced':q['labels'][0]=1
  else:q['predictions'][0]=False
  reject(lambda q=q:prediction(q))
 scores={a:{v:dict(accuracy=.85 if a=='Q_mix' and v=='clean' else .84 if v=='clean' else .42 if a=='Q_mix' else .30 if a=='Q0' else .40,macro_F1=.85 if a=='Q_mix' else .84) for v in VIEWS} for a in ARMS};execution={a:dict(seen_V=dict(source_val_accuracy=.90),final_base_L_objective=.2) for a in ARMS};assert qualification(scores,execution)['qualified']
 for kind in ('clean','cleanF1','low','lowF1','mid','suburban','V','base','Qsatbetter','Q0low','Q0urban'):
  z=copy.deepcopy(scores);e=copy.deepcopy(execution)
  if kind=='clean':z['Q_mix']['clean']['accuracy']=.83
  elif kind=='cleanF1':z['Q_mix']['clean']['macro_F1']=.83
  elif kind=='low':z['Q_mix'][VIEWS[-1]]['accuracy']=.405
  elif kind=='lowF1':z['Q_mix'][VIEWS[-1]]['macro_F1']=.83
  elif kind=='mid':z['Q_mix'][VIEWS[-2]]['accuracy']=.39
  elif kind=='suburban':z['Q_mix'][VIEWS[1]]['accuracy']=.39
  elif kind=='V':e['Q_mix']['seen_V']['source_val_accuracy']=.89
  elif kind=='base':e['Q_mix']['final_base_L_objective']=.6
  elif kind=='Qsatbetter':z['Q_sat'][VIEWS[-1]]['accuracy']=.415
  elif kind=='Q0low':z['Q0'][VIEWS[-1]]['accuracy']=.38
  else:
   for v in VIEWS[4:6]:z['Q0'][v]['accuracy']=.415
   z['Q0'][VIEWS[-1]]['accuracy']=.36
  assert not qualification(z,e)['qualified'];negative+=1
 # Paired transitions distinguish the unchanged clean-correct set from changed errors.
 ref={'clean':copy.deepcopy(p),'urban':copy.deepcopy(p)};cand=copy.deepcopy(ref)
 ref['urban']['predictions'][0]=1;ref['urban']['predictions'][1]=1;cand['urban']['predictions'][1]=1;cand['urban']['predictions'][2]=1;cand['clean']['predictions'][2]=1
 z=decision_transitions(ref,cand,'urban');assert z['all']['repaired']==z['all']['broken']==1 and z['both_clean_correct']['repaired']==1 and z['both_clean_correct']['broken']==0 and z['clean_correctness_changed']['net_correct']==-1 and z['both_clean_wrong']['delta_pp'] is None
 wrong=copy.deepcopy(cand);wrong['urban']['physical_ids'].reverse();reject(lambda:decision_transitions(ref,wrong,'urban'))
 fixtures=0
 with tempfile.TemporaryDirectory() as temp:
  base=Path(temp)
  for arm,policy in [('Q_ord','ordinary'),('Q_sat','satellite'),('Q_mix','mixed')]:
   rows=[];selected=0;ordinary=0
   for step in range(1,4441):
    on=step>1110;sel=list(range(128)) if on and policy=='satellite' else sorted(np.random.default_rng(202610176001+step).permutation(128)[:64].tolist()) if on and policy=='mixed' else [];count=len(sel);selected+=count;ordinary+=128-count if on else 0;dd=[dict(phase=ph,level=le,shape=[128,4,80],total_anchors=128,valid_anchors=128,positive_min=1,positive_max=10,loss=.3) for ph in ('first','second') for le in ('time_down','t2.pool')] if on else []
    rows.append(dict(step=step,supcon_details=dd,supcon=.3 if on else 0.,route_policy=policy,route_active=on,route_selected_As_positions=sel,route_As_appearances=count,route_ordinary_appearances=128-count if on else 0,route_new_NN_forwards=0,route_new_loss_terms=0))
   audit=dict(calls=4440,applied=3330,As_appearances=selected,ordinary_appearances=ordinary,phase_terms=13320,teacher_calls=0,reader_calls=0,extra_NN_forwards=0,new_loss_terms=0,default_head_joint=True,target_access=False)
   (base/'supcon_views_audit.json').write_text(json.dumps(audit),encoding='utf8');out=routing(arm,base,rows);assert out['selected_As_appearances']==selected and out['existing_layer_phase_terms']==13320;fixtures+=4440
   if arm!='Q_ord':
    for key,value in [('route_selected_As_positions',[]),('route_active',False),('route_As_appearances',1),('route_new_NN_forwards',1),('supcon',.5)]:
     before=rows[1110][key];rows[1110][key]=value;reject(lambda:routing(arm,base,rows));rows[1110][key]=before
    audit['As_appearances']+=1;(base/'supcon_views_audit.json').write_text(json.dumps(audit),encoding='utf8');reject(lambda:routing(arm,base,rows))
 return dict(status='PASS',counterexamples=negative,complete_route_fixture_steps=fixtures,source_prediction_fixture_records=1260,real_IQ=0,NN=0,updates=0,checkpoint_loads=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
