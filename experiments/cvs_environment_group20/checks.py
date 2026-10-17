"""Fixed source roles,qualifications and unsupported overrides;zero new NN."""
import copy,json
from . import design as d

def run():
 negative=0
 records={a:dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,group_generated=568320,supcon_applied=3330,As_training_observations=0 if a=='RAW' else 568320,final_base_L_objective=.2,held_RX=6,fit_RX=list(d.FIT),head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[],seen_V=dict(source_val_accuracy=.9),views={v:dict(accuracy=.84 if a=='PROP_GROUP' and v=='clean' else .4 if a=='PROP_GROUP' else .82 if v=='clean' else .3 if a=='RAW' else .38,macro_F1=.84 if a=='PROP_GROUP' else .82) for v in ('clean',*d.SCENES)}) for a in d.ARMS}
 assert d.qualify(records)['qualified']
 for ref in d.ARMS[:-1]:
  for kind in ('clean','F1','low','urban','practical','V'):
   r=copy.deepcopy(records)
   if kind=='clean':r['PROP_GROUP']['views']['clean']['accuracy']=.80
   elif kind=='F1':r['PROP_GROUP']['views']['clean']['macro_F1']=.80
   elif kind=='low':r['PROP_GROUP']['views'][d.SCENES[-1]]['accuracy']=r[ref]['views'][d.SCENES[-1]]['accuracy']
   elif kind=='urban':
    for v in d.SCENES[3:]:r['PROP_GROUP']['views'][v]['accuracy']=r[ref]['views'][v]['accuracy']
   elif kind=='practical':r['PROP_GROUP']['views'][d.SCENES[0]]['accuracy']=.1
   else:r['PROP_GROUP']['seen_V']['source_val_accuracy']=.8
   assert not d.qualify(r)['qualified'];negative+=1
 for name,value in (('actual_steps',4439),('group_generated',0),('As_training_observations',0),('final_base_L_objective',.6),('held_RX',1),('fit_RX',[3,4,6,8]),('target_access',True),('U_truth_read',True),('checkpoint_sources',['invalid'])):
  r=copy.deepcopy(records);r['PROP_GROUP'][name]=value;assert not d.qualify(r)['qualified'];negative+=1
 for field,value in (('TX_used_in_group',True),('hardware_session','TX'),('quality','fixedhighSNR'),('formal_test_unchanged',False)):
  c=copy.deepcopy(d.config(d.rows()[-1]));c['channel_group'][field]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Protected channel protocol override accepted')
 from . import source as _source
 from experiments.cvs_f_additive_rx6.checks import fold_checks
 fold=fold_checks()
 return dict(status='PASS',scientific_config_negatives=negative,source_role_checks=fold,source_IQ=0,CP_reads=0,NN=0,updates=0,no_goal_efficacy_claim=True)
if __name__=='__main__':print(json.dumps(run()))
