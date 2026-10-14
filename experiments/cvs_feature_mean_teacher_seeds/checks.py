import copy,json,math,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,qualification as q,model as api
from experiments.cvs_feature_mean_teacher50 import design as first,observer as observations
def checks():
 torch.set_num_threads(2);rules=first.config(first.training_rows()[0])['source_falsification'];arms={'SIG_TIME','FEATURE_MT'}
 source=dict(status='PASS',source_only=True,target_metrics_read_for_decision=False,head_fit_updates=0,configuration_changes_for_expansion=False,predefined_remaining_seeds=list(d.SEEDS),fixed_rules=rules,final_L_CE={r:.2 for r in arms},feature_discrepancy_ratio=.9,actual_candidate_minus_control_L720zL2=.1,actual_t1_t2_SIG_gradient_signal=dict(FEATURE_MT=True),actual_observations={r:dict(status='VERIFIED',steps=11100,U_feature_packets=963900,U_feature_steps=3774,gradient_batches=666,gradient_epochs=[33,34,50],extra_model_forwards=0,head_fit_updates=0,U_truth_read=False,hard_pseudo_loss_returned_unchanged=True,auxiliary_classW_dependency=False) for r in arms})
 files={'source_qualification.json':source,'source_features.json':dict(status='VERIFIED',models=2,paired=[{}],source_only=True,query_read=False,head_fit=False),'completion.json':dict(status='EXPERIMENT_COMPLETE',models=3,new_trained_models=2,prediction_count=504000,truth_last=True),'experiment.json':dict(execution_commit=q.FIRST_EXECUTION,rows=[dict(config=first.config(r)) for r in first.rows()])};read=first.read;negative=0
 try:
  first.read=lambda path:files[path.name];assert q.check()['status']=='PASS'
  for key,value in [('status','FAIL'),('target_metrics_read_for_decision',True),('predefined_remaining_seeds',[2026092703]),('actual_observations',{}),('head_fit_updates',1),('configuration_changes_for_expansion',True),('feature_discrepancy_ratio',None),('feature_discrepancy_ratio',.96),('feature_discrepancy_ratio',float('nan')),('actual_candidate_minus_control_L720zL2',0.),('actual_t1_t2_SIG_gradient_signal',dict(FEATURE_MT=False))]:
   saved=copy.deepcopy(files['source_qualification.json']);files['source_qualification.json'][key]=value
   try:q.check()
   except ValueError:negative+=1
   else:raise AssertionError('Failed source consistency qualification accepted')
   files['source_qualification.json']=saved
 finally:first.read=read
 contexts=[];argument_changes=[]
 for seed in d.SEEDS:
  models=[]
  for row in d.training_rows():
   if row['model_seed']!=seed:continue
   c=d.config(row);d.validate(c);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']));assert all(c[k]==template[k] for k in ('feature_mean_teacher','source_neighbor_sampling','sampling_observation','feature_combination'))
   a=d.make_args(c,'cpu');old=first.make_args(template,'cpu');equal=lambda x,y:x==y or isinstance(x,float) and isinstance(y,float) and math.isnan(x) and math.isnan(y);changed={k for k,v in vars(a).items() if not equal(v,vars(old).get(k))};assert changed<={'seed','model_seed','run_id','candidate_id','output_dir'}
   m=api.build(c,'cpu').eval()
   with torch.no_grad():assert m.forward_identity_only(torch.zeros(2,2,256))['id_feat_joint'].shape==(2,160)
   models.append(m);contexts.append(dict(seed=seed,arm=row['arm'],no_query_forward=True));argument_changes.append(sorted(changed))
  initial=models[0].state_dict();assert len(models)==2 and all(torch.equal(v,models[1].state_dict()[k]) for k,v in initial.items())
 c=d.config(next(r for r in d.training_rows() if r['arm']=='FEATURE_MT'));m=api.build(c,'cpu').train();teacher=copy.deepcopy(m).eval().requires_grad_(False);previous=observations.d;observations.d=d
 try:
  with tempfile.TemporaryDirectory() as directory:
   obs=observations.Observer(c);obs.initialize(m,c,Path(directory));fake=SimpleNamespace(_trusted_pseudo_loss_hook=None,_source_feature_auxiliary_hook=None,_unlabeled_feature_observer_hook=None,_build_ssdg_wisig_data=lambda:None)
   with obs.native_scope(fake):
    x=torch.randn(4,2,256);y=torch.arange(4)%6;ce=F.cross_entropy(m.forward_identity_only(x)['tx_logits'],y)
    with torch.no_grad():weak=teacher.forward_identity_only(x)
    strong=m.forward_identity_only(x+.015);pseudo=weak['tx_logits'].argmax(1);uce=F.cross_entropy(strong['tx_logits'],pseudo);ent=-(strong['tx_logits'].softmax(1)*strong['tx_logits'].log_softmax(1)).sum(1).mean();obs.steps=33*222;a=d.make_args(c,'cpu')
    obs.capture(model=m,args=a,epoch=34,batch_idx=1,teacher_features=weak['id_feat_joint'],student_features=strong['id_feat_joint']);returned=obs.observe(model=m,args=a,epoch=34,batch_idx=1,loss_l=ce,loss_u=uce,loss_ent=ent,active=True,logits=strong['tx_logits'],pseudo=pseudo,mask=torch.ones(4,dtype=torch.bool));extra,_=obs.aux(m,33*222+1);grad=torch.autograd.grad(extra,obs.params+[obs.head],allow_unused=True)
    assert returned is uce and grad[-1] is None and all(v is not None and v.norm()>0 for v in grad[:-1]) and obs.pending is None
 finally:observations.d=previous
 from .publish import configure
 from experiments.cvs_report_ssl import publish as transport
 configure();assert all(p in transport.PATHS for p in ('experiments/cvs_feature_mean_teacher_seeds','experiments/cvs_feature_mean_teacher50','experiments/cvs_sigtime_pair50'));compile(transport.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 return dict(status='PASS',source_qualification_negative_tests=negative,remaining_native_contexts=contexts,native_argument_differences=argument_changes,original_mechanism_exact=True,same_seed_initial_states_equal=True,remaining_observer_context_actual_feature_gradients=True,auxiliary_W_gradient_absent=True,query_read=False,head_fit_updates=0,optimizer_updates=0,unchanged_core_review_reused=True,remote_compile='PASS')
if __name__=='__main__':print(json.dumps(checks()))
