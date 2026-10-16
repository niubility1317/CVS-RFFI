"""Focused protocol negatives and equal-dose/order counterexamples;0realIQ/NN."""
import copy,json
from . import design as d
def run():
 n=0
 def reject(fn):
  nonlocal n
  try:fn()
  except (ValueError,TypeError,KeyError):n+=1;return
  raise AssertionError('Illegal fixed source timing accepted')
 for key,val in [('epochs',21),('model_seed',2026092702),('checkpoint_sources',['old']),('target_access',True),('separate_head_training',True),('source_receivers',[1,3,4,6,8]),('output_root','old')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c[key]=val;reject(lambda:d.validate(c))
 for key,val in [('scope','U'),('extra_weight',.5),('active_epochs',[1,20]),('same_dose_EARLY7_LATE7',0),('formal_test_unchanged',False)]:
  c=copy.deepcopy(d.config(d.rows()[0]));c['As_timing'][key]=val;reject(lambda:d.validate(c))
 for a,e in [('late',1),('LATE7',0),('EARLY7',21)]:reject(lambda:d.active(a,e))
 assert {a:d.packets(a) for a in d.ARMS}==dict(RAW=0,ALL=568320,EARLY7=198912,LATE7=198912)
 assert [e for e in range(1,21) if d.active('EARLY7',e)]==list(range(1,8)) and [e for e in range(1,21) if d.active('LATE7',e)]==list(range(14,21))
 result={}
 for a in d.ARMS:
  v=.82 if a=='LATE7' else .80;result[a]=dict(views={s:dict(accuracy=(v if s=='clean' else .50 if a=='LATE7' else .40),macro_F1=v) for s in ('clean',*d.SCENES)},seen_V=dict(source_val_accuracy=.90),actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,As_generated_packets=568320,As_training_observations=d.packets(a),timing_steps=4440,head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[])
 assert d.qualify(result)['qualified'];scientific=0
 for reference,view,change in [('RAW','clean',.03),('ALL','clean',.03),('EARLY7','clean',.03),('RAW',d.SCENES[-1],.12),('EARLY7',d.SCENES[-1],.12),('ALL',d.SCENES[-1],.12)]:
  bad=copy.deepcopy(result);bad[reference]['views'][view]['accuracy']+=change;assert not d.qualify(bad)['qualified'];scientific+=1
 bad=copy.deepcopy(result);bad['LATE7']['As_training_observations']=568320;assert not d.qualify(bad)['qualified'];scientific+=1
 return dict(status='PASS',protocol_negatives=n,scientific_counterexamples=scientific,equal_dose=198912,real_IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
