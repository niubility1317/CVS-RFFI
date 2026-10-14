import copy,json,math,torch
from . import design as d,qualification as q
from experiments.cvs_backbone_fishr50 import design as first
def checks():
 torch.set_num_threads(2);rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()}
 source=dict(status='PASS',source_only=True,target_metrics_read_for_decision=False,head_fit_updates=0,configuration_changes_for_expansion=False,predefined_remaining_seeds=list(d.SEEDS),fixed_rules=rules,final_L_CE={r:.2 for r in expected},expand_both_candidates_only_after_both_source_PASS=True,candidates=[dict(arm=a,actual_L720zL2=.1,actual_t1_t2_gradient_nonzero_17_33_50=True,passed=True) for a in ('BONE_RAW_FISHR','BONE_COND_FISHR')],actual_observations={r:dict(status='VERIFIED',steps=11100,extra_batches=650,extra_source_packet_forwards=117000,gradient_batches=52,gradient_epochs=[1,17,33,50],head_fit_updates=0,U_truth_read=False) for r in expected})
 files={'source_qualification.json':source,'source_features.json':dict(status='VERIFIED',models=3,paired=[{},{}],source_only=True,query_read=False,head_fit=False),'completion.json':dict(status='EXPERIMENT_COMPLETE',models=4,new_trained_models=3,prediction_count=672000,truth_last=True),'experiment.json':dict(execution_commit=q.FIRST_EXECUTION,rows=[dict(config=first.config(r)) for r in first.rows()])};read=first.read;negative=0
 try:
  first.read=lambda path:files[path.name];assert q.check()['status']=='PASS'
  cases=[('status','FAIL'),('target_metrics_read_for_decision',True),('predefined_remaining_seeds',[2026092703]),('actual_observations',{}),('head_fit_updates',1),('configuration_changes_for_expansion',True),('expand_both_candidates_only_after_both_source_PASS',False),('candidates',source['candidates'][:1])]
  bad=copy.deepcopy(source['candidates']);bad[0]['actual_L720zL2']=0.;cases.append(('candidates',bad));bad=copy.deepcopy(source['candidates']);bad[1]['actual_t1_t2_gradient_nonzero_17_33_50']=False;cases.append(('candidates',bad))
  for key,value in cases:
   saved=copy.deepcopy(files['source_qualification.json']);files['source_qualification.json'][key]=value
   try:q.check()
   except ValueError:negative+=1
   else:raise AssertionError('Failed source expansion accepted')
   files['source_qualification.json']=saved
 finally:first.read=read
 from . import model as api
 from experiments.cvs_backbone_fishr50 import objective
 contexts=[];differences=[]
 for seed in d.SEEDS:
  models=[]
  for row in d.training_rows():
   if row['model_seed']!=seed:continue
   c=d.config(row);d.validate(c);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']));assert c['backbone_fishr']==template['backbone_fishr'] and c['feature_combination']==template['feature_combination']
   a=d.make_args(c,'cpu');original=first.make_args(template,'cpu')
   equal=lambda x,y:x==y or isinstance(x,float) and isinstance(y,float) and math.isnan(x) and math.isnan(y)
   changed={k for k,v in vars(a).items() if not equal(v,vars(original).get(k))}
   assert changed<={'seed','model_seed','run_id','candidate_id','output_dir'},changed
   assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent,a.label_smoothing)==(50,33,17,.16,.01,.01) and a.pseudo_temporal_mode=='batch_neighbor' and a.use_ema_teacher
   m=api.build(c,'cpu').eval();state={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
   assert out['id_feat_joint'].shape==(2,160) and all(torch.equal(v,m.state_dict()[k]) for k,v in state.items());models.append(m);contexts.append(dict(seed=seed,arm=row['arm'],no_query_forward=True));differences.append(sorted(changed))
  initial=models[0].state_dict();assert len(models)==3 and all(all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) for m in models)
 assert d.FISHR_PARAMETERS==first.FISHR_PARAMETERS and len(d.FISHR_PARAMETERS)==2 and len(d.PARAMETERS)==3 and d.EXTRA_BATCHES==650 and d.COHORT_PACKETS==180 and d.COEFFICIENT==.01
 from .publish import configure
 from experiments.cvs_report_ssl import publish as transport
 configure();assert all(p in transport.PATHS for p in ('experiments/cvs_backbone_fishr50','experiments/cvs_backbone_fishr_seeds','experiments/cvs_signature_order50','experiments/cvs_source_layer_order'));compile(transport.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 return dict(status='PASS',source_qualification_negative_tests=negative,remaining_native_contexts=contexts,native_argument_differences=differences,original_mechanism_exact=True,same_seed_actual_initial_states_equal=True,only_actual_t1_t2_required_SIG_diagnostic_only=True,query_read=False,head_fit_updates=0,optimizer_updates=0,unchanged_core_review_reused=True,remote_compile='PASS')
if __name__=='__main__':print(json.dumps(checks()))
