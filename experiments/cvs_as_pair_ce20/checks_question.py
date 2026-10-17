"""Fixed scientific criteria counterexamples;no model forward or data read."""
import copy,json
from . import design as d

def run():
 fake={a:dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,total_pair_weight=1.25,As_training_observations=0 if a=='Q0' else 568320,head_fit_updates=0,target_access=False,final_base_L_objective=.2,seen_V=dict(source_val_accuracy=.9),views={v:dict(accuracy=(.86 if a=='C_CE' else .85 if a=='Q_ord' else .80) if v=='clean' else .42 if a=='C_CE' else .40 if a=='Q_ord' else .30,macro_F1=.86 if a=='C_CE' else .85 if a=='Q_ord' else .80) for v in ('clean',*d.SCENES)}) for a in d.ARMS}
 fake['C_CE'].update(As_pair_objective='TWO_CE',As_KL_loss_weight=0.,As_KL_gradient_terms=0,As_pair_calls=4440)
 assert d.qualify(fake)['qualified'];negative=0
 for key in ('clean','clean_F1','low','low_F1','mid','urban','suburban','seenV','budget','As','KL','calls','loss','target'):
  q=copy.deepcopy(fake);r=q['C_CE']
  if key=='clean':r['views']['clean']['accuracy']=.84
  elif key=='clean_F1':r['views']['clean']['macro_F1']=.84
  elif key=='low':r['views'][d.SCENES[-1]]['accuracy']=.405
  elif key=='low_F1':r['views'][d.SCENES[-1]]['macro_F1']=.84
  elif key=='mid':r['views'][d.SCENES[-2]]['accuracy']=.39
  elif key=='urban':
   for v in d.SCENES[3:]:r['views'][v]['accuracy']=.405
  elif key=='suburban':r['views'][d.SCENES[0]]['accuracy']=.39
  elif key=='seenV':r['seen_V']['source_val_accuracy']=.89
  elif key=='budget':r['actual_steps']=4439
  elif key=='As':r['As_training_observations']=568319
  elif key=='KL':r['As_KL_gradient_terms']=1
  elif key=='calls':r['As_pair_calls']=4439
  elif key=='loss':r['final_base_L_objective']=.51
  elif key=='target':r['target_access']=True
  assert not d.qualify(q)['qualified'],key;negative+=1
 try:d.qualify({'C_CE':fake['C_CE']})
 except ValueError:negative+=1
 else:raise AssertionError('Missing controls accepted')
 return dict(status='PASS',scientific_counterexamples=negative,synthetic_metric_fixture=True,NN=0,real_IQ=0,checkpoint_reads=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
