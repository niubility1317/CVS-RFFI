"""Focused source config/scorer/CLI checks;no IQ/checkpoint/NN."""
import copy,json,subprocess,sys
from . import design as d
from .scoring import all_scores

def run():
 negative=0
 def reject(f):
  nonlocal negative
  try:f()
  except ValueError:negative+=1
  else:raise AssertionError('Invalid source run accepted')
 for key,value in [('model_seed',2026092702),('checkpoint_source','other'),('U_used',True),('V_used',True),('target_access',True),('optimizer_updates',1),('head_fit_updates',1),('new_training_loss_terms',1),('As_generations',1),('threads',4)]:
  c=copy.deepcopy(d.config());c[key]=value;reject(lambda c=c:d.validate(c))
 for key,value in [('epsilon_absolute_complex_RMS',.002),('xi_absolute_complex_RMS',.002),('peak_absolute',.01),('steps',2),('label_input',True)]:
  c=copy.deepcopy(d.config());c['operator'][key]=value;reject(lambda c=c:d.validate(c))
 for key in ('reference_forward_packets','input_backward_packets','total_forward_packets'):
  c=copy.deepcopy(d.config());c['expected'][key]-=1;reject(lambda c=c:d.validate(c))
 text=(d.ROOT/'experiments/cvs_shared_iq17/publish.py').read_text(encoding='utf8')
 def entrypoint(text):
  if 'experiments/cvs_commonphase_source/runner.py' in text or text.count('experiments/cvs_shared_iq17/runner.py')!=2:raise ValueError('Publisher must bootstrap and launch only the new runner')
 entrypoint(text);reject(lambda:entrypoint(text.replace('experiments/cvs_shared_iq17/runner.py','experiments/cvs_commonphase_source/runner.py',1)))
 labels=[];rx=[];day=[]
 for r in d.FIT:
  for dd in (1,2,3):
   for t in range(6):labels +=[t]*70;rx +=[r]*70;day +=[dd]*70
 p={k:labels.copy() for k in d.VIEWS}
 for i in range(10):p['direction'][i]=1
 s=all_scores(labels,p,rx,day);assert s['raw']['accuracy']==1 and s['direction']['accuracy']==s['direction']['retention']==5030/5040 and s['direction']['RX_accuracy'][3]==1250/1260 and s['direction']['TX_retention'][0]==830/840
 reject(lambda:all_scores(labels,p,rx[:-1],day));reject(lambda:all_scores(labels,p,rx,day[:-1]));q=copy.deepcopy(p);q['direction'].pop();reject(lambda:all_scores(labels,q,rx,day));q=copy.deepcopy(p);q['direction'][0]=-1;reject(lambda:all_scores(labels,q,rx,day))
 z=subprocess.run([sys.executable,'-S',str(d.ROOT/'experiments/cvs_shared_iq17/runner.py'),'--override'],capture_output=True,text=True)
 assert z.returncode and 'No model/source/operator/budget overrides' in z.stderr and 'ModuleNotFoundError' not in z.stderr;negative+=1
 return dict(status='PASS',counterexamples=negative,synthetic_source_scoring_records=15120,real_IQ=0,NN=0,input_backwards=0,updates=0,checkpoint_loads=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
