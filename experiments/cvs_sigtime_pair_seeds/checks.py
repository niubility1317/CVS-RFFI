import copy,json,torch
from . import design as d,qualification as q
from experiments.cvs_sigtime_pair50 import design as first

def checks():
 torch.set_num_threads(2);rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()}
 source=dict(status='PASS',source_only=True,target_metrics_read_for_decision=False,head_fit_updates=0,configuration_changes_for_expansion=False,predefined_remaining_seeds=list(d.SEEDS),fixed_rules=rules,final_L_CE={r:.2 for r in expected},actual_SIG_PAIR_minus_RANDOM_L720zL2=.1,paired_U_acceptance=.9,actual_observations={r:dict(status='VERIFIED',steps=11100,U_packets=963900,sampler_epochs=17,sampler_packets=963900,gradient_batches=666,gradient_epochs=[33,34,50],head_fit_updates=0,U_truth_read=False,loss_returned_unchanged=True) for r in expected})
 files={'source_qualification.json':source,'source_features.json':dict(status='VERIFIED',models=2,paired=[{}],source_only=True,query_read=False,head_fit=False),'completion.json':dict(status='EXPERIMENT_COMPLETE',models=3,new_trained_models=2,prediction_count=504000,truth_last=True),'experiment.json':dict(execution_commit=q.FIRST_EXECUTION,rows=[dict(config=first.config(r)) for r in first.rows()])};read=first.read;negative=0
 try:
  first.read=lambda path:files[path.name];assert q.check()['status']=='PASS'
  for key,value in [('status','FAIL'),('target_metrics_read_for_decision',True),('predefined_remaining_seeds',[2026092703]),('actual_observations',{}),('head_fit_updates',1),('configuration_changes_for_expansion',True),('actual_SIG_PAIR_minus_RANDOM_L720zL2',0.),('paired_U_acceptance',.1)]:
   saved=copy.deepcopy(files['source_qualification.json']);files['source_qualification.json'][key]=value
   try:q.check()
   except ValueError:negative+=1
   else:raise AssertionError('Failed source expansion accepted')
   files['source_qualification.json']=saved
 finally:first.read=read
 from . import model as api
 from experiments.cvs_sigtime_pair50 import observer as observation
 contexts=[]
 for seed in d.SEEDS:
  models=[]
  for row in d.training_rows():
   if row['model_seed']!=seed:continue
   c=d.config(row);d.validate(c);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']));assert c['source_neighbor_sampling']==template['source_neighbor_sampling'] and c['sampling_observation']==template['sampling_observation'];a=d.make_args(c,'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent)==(50,33,17,.16,.01) and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor'
   m=api.build(c,'cpu').eval();state={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
   assert out['id_feat_joint'].shape==(2,160) and m._source_neighbor_sampling['mode']==c['source_neighbor_sampling']['mode'] and all(torch.equal(v,m.state_dict()[k]) for k,v in state.items());models.append(m)
   previous=observation.d;observation.d=d
   try:assert observation.Observer(c).c==c
   finally:observation.d=previous
   contexts.append(dict(seed=seed,arm=row['arm'],output=c['output_root'],no_query_forward=True))
  ref=models[0].state_dict();assert len(models)==2 and all(all(torch.equal(v,m.state_dict()[k]) for k,v in ref.items()) for m in models)
 from .publish import configure
 from experiments.cvs_report_ssl import publish as t
 configure();assert all(p in t.PATHS for p in ['experiments/cvs_sigtime_pair50','experiments/cvs_neighbor_pair50','experiments/cvs_source_neighbor_pairs','experiments/cvs_trusted_weight50']);compile(t.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 return dict(status='PASS',source_qualification_negative_tests=negative,remaining_native_contexts=contexts,same_seed_initial_states_equal=True,original_SSL_recipe_equal=True,observer_design_binding=True,query_read=False,head_fit_updates=0,optimizer_updates=0,unchanged_core_review_reused=True)
if __name__=='__main__':print(json.dumps(checks()))
