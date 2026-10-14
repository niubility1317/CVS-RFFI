import copy,json,torch
from . import design as d,qualification as q
from experiments.cvs_local_order50 import design as first

def checks():
 torch.set_num_threads(2);rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()}
 source=dict(status='PASS',source_only=True,target_metrics_read_for_decision=False,head_fit_updates=0,configuration_changes_for_expansion=False,predefined_remaining_seeds=list(d.SEEDS),fixed_rules=rules,final_L_CE={r:.2 for r in expected},actual_ORDER_minus_SIG_L720zL2=.1,actual_ORDER_effective_gradient_t1_t2_nonzero_17_33_50=True,actual_observations={r:dict(status='VERIFIED',extra_batches=2750,gradient_batches=220,gradient_epochs=[1,17,33,50],head_fit_updates=0,U_truth_read=False) for r in expected})
 files={'source_qualification.json':source,'source_features.json':dict(status='VERIFIED',models=3,paired=[{}],source_only=True,query_read=False,head_fit=False),'completion.json':dict(status='EXPERIMENT_COMPLETE',models=4,new_trained_models=3,prediction_count=672000,truth_last=True),'experiment.json':dict(execution_commit=q.FIRST_EXECUTION,rows=[dict(config=first.config(r)) for r in first.rows()])};read=first.read;negatives=0
 try:
  first.read=lambda path:files[path.name];assert q.check()['status']=='PASS'
  for key,value in [('status','FAIL'),('target_metrics_read_for_decision',True),('actual_ORDER_minus_SIG_L720zL2',0.),('actual_ORDER_effective_gradient_t1_t2_nonzero_17_33_50',False),('predefined_remaining_seeds',[2026092703]),('actual_observations',{}),('head_fit_updates',1),('configuration_changes_for_expansion',True)]:
   saved=copy.deepcopy(files['source_qualification.json']);files['source_qualification.json'][key]=value
   try:q.check()
   except ValueError:negatives+=1
   else:raise AssertionError('Failed source expansion accepted')
   files['source_qualification.json']=saved
 finally:first.read=read
 from . import model as api
 contexts=[]
 for seed in d.SEEDS:
  models=[]
  for row in d.training_rows():
   if row['model_seed']!=seed:continue
   c=d.config(row);d.validate(c);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']));assert c['local_order']==template['local_order'];a=d.make_args(c,'cpu');assert a.epochs==50 and a.label_epochs==33 and a.pseudo_epochs==17 and a.pseudo_temporal_mode=='batch_neighbor' and a.use_ema_teacher
   m=api.build(c,'cpu').eval();state={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
   assert out['id_feat_joint'].shape==(2,160) and all(torch.equal(v,m.state_dict()[k]) for k,v in state.items());models.append(m);contexts.append(dict(seed=seed,arm=row['arm'],output=c['output_root'],no_query_forward=True))
  firststate=models[0].state_dict();assert len(models)==3 and all(all(torch.equal(v,m.state_dict()[k]) for k,v in firststate.items()) for m in models)
 from .publish import configure
 from experiments.cvs_report_ssl import publish as t
 configure();assert 'experiments/cvs_local_order50' in t.PATHS and 'experiments/cvs_source_layer_order' in t.PATHS and 'experiments/cvs_trusted_weight50' in t.PATHS;compile(t.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 return dict(status='PASS',source_qualification_negative_tests=negatives,remaining_native_contexts=contexts,same_seed_actual_initial_states_equal=True,original_local_order_recipe_equal=True,query_read=False,head_fit_updates=0,optimizer_updates=0,unchanged_core_review_reused=True)
if __name__=='__main__':print(json.dumps(checks()))
