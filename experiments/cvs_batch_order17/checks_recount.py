"""Metadata-only counterexamples for the offline reader;zero inference/updates."""
from copy import deepcopy
import json
from .recount_order import block,batch_metrics,same_stats,items
from .analyze_e20 import prediction,pair_formula,qualification,VIEWS

def run():
 count=0
 def reject(fn):
  nonlocal count
  try:fn()
  except (ValueError,AssertionError,TypeError,KeyError):count+=1
  else:raise AssertionError('Counterexample was accepted')
 flat=[(y,1,r,1,i,i) for i,(y,r) in enumerate(( (y,r) for y in range(6) for r in (3,4,6,8) for _ in range(32)))];before=[flat[i:i+128] for i in range(0,768,128)];after=[flat[i::6] for i in range(6)]
 def rows(actual):return [dict(original_ids=[list(p[1:]) for p in x],original_labels=[p[0] for p in x],actual_ids=[list(p[1:]) for p in z],actual_labels=[p[0] for p in z]) for x,z in zip(before,actual)]
 assert block(rows(before),'O0')[2]==0 and block(rows(after),'O1')[2]>0
 reject(lambda:block(rows(after),'O0'));reject(lambda:block(rows(before),'O1'))
 bad=rows(after);bad[0]['actual_ids'][0]=bad[0]['actual_ids'][1];reject(lambda:block(bad,'O1'))
 bad=rows(after);bad[0]['actual_labels'][0]=5;reject(lambda:block(bad,'O1'))
 reject(lambda:block(rows(after)[:5],'O1'));reject(lambda:block(rows(after),'unexpected'))
 reject(lambda:items([[1,1,1,i,i] for i in range(128)],[0]*128));reject(lambda:items([[1,3,1,i,i] for i in range(128)],[-1]*128))
 stats=batch_metrics(after[0]);assert stats['valid_anchors']==128;same_stats(stats,deepcopy(stats));bad=deepcopy(stats);bad['positive_min']+=1;reject(lambda:same_stats(stats,bad));bad=deepcopy(stats);bad.pop('positive_max');reject(lambda:same_stats(stats,bad))
 p=dict(labels=[],predictions=[],physical_ids=[])
 for y in range(6):
  for d in (1,2,3):
   for s in range(70):p['labels'].append(y);p['predictions'].append(y);p['physical_ids'].append([1,1,d,y*210+(d-1)*70+s,len(p['labels'])-1])
 assert prediction(p)['accuracy']==1
 bad=deepcopy(p);bad['physical_ids'][0]=bad['physical_ids'][1];reject(lambda:prediction(bad))
 bad=deepcopy(p);bad['predictions'][0]=1.0;reject(lambda:prediction(bad))
 bad=deepcopy(p);bad['labels'][0]=1;reject(lambda:prediction(bad))
 bad=deepcopy(p);bad['physical_ids'][0][1]=3;reject(lambda:prediction(bad))
 v=dict(raw_CE=1.,raw_KL=.1,ordinary_CE=2.,ordinary_KL=.2,s_slot_CE=3.,s_slot_KL=.3,supcon_weight=.1,supcon=.7,extra_pair_CE=4.,extra_pair_KL=.4)
 base,extra,total=pair_formula(v,'Q0');assert abs(base-1.995)<1e-12 and extra==4.4 and abs(total-3.095)<1e-12
 scores={a:{view:dict(accuracy=.8,macro_F1=.8) for view in VIEWS} for a in ('O0','O1')};scores['O1']['clean']['accuracy']=.81
 e={a:dict(seen_V=dict(source_val_accuracy=.9),actual_L_order=dict(all4440_actual_inputs_checked=True,all740_physical_block_multisets_identical=True,moved_positions=0 if a=='O0' else 20),full_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,total_pair_weight=1.25,As_training_observations=0,head_fit_updates=0,target_access=False,final_base_L_objective=.1) for a in ('O0','O1')}
 assert qualification(scores,e)['qualified']
 for view in VIEWS:
  bad=deepcopy(scores);bad['O1'][view]['accuracy']=.809 if view=='clean' else .794;assert not qualification(bad,e)['qualified'];count+=1
 bad=deepcopy(scores);bad['O1']['clean']['macro_F1']=.799;assert not qualification(bad,e)['qualified'];count+=1
 bad=deepcopy(e);bad['O1']['seen_V']['source_val_accuracy']=.894;assert not qualification(scores,bad)['qualified'];count+=1
 bad=deepcopy(e);bad['O1']['final_base_L_objective']=.5001;assert not qualification(scores,bad)['qualified'];count+=1
 return dict(status='PASS',rejected_metadata_counterexamples=count,new_NN=0,checkpoint_loads=0,real_IQ=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
