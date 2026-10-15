import copy,json,os,torch
from torch.nn import functional as F
from . import design as d,model as api,provenance as proof
negative=[]
def reject(label,call):
 try:call()
 except (ValueError,RuntimeError):negative.append(label);return
 raise AssertionError('Illegal path accepted: '+label)
def fixture(c,report):
 e=c['epochs'];steps=e*222;p=c['output_root']
 done=dict(status='SOURCE_TRAINED',config=c,epoch=e,logged_steps=steps,optimizer_steps=steps,L_context_steps=steps,target_access=False,target_evaluated=False,checkpoint_sources=[],ancestors=[],initialization='scratch',label_epochs=c['label_epochs'],pseudo_epochs=c['pseudo_epochs'],baseline_recipe=c['training_recipe'],head_fit_updates=0,native_fitted_probes_disabled=True,native_disabled_probe_calls=1,identity_input_policy=dict(explicit_CFO_identity_input=False),feature_extractor=report,mask_counts={},final_source_metrics=dict(source_val_count=27000,source_val_rx_accuracy={str(r):1. for r in (1,3,4,6,8)}))
 initial=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed'])
 freeze=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(__import__('pathlib').Path(p)/'final_ssdg.pth'),epoch=e,selection='final_only',target_access=False)
 resolved=dict(config=c,native_fitted_probes_disabled=True,identity_input_policy=dict(explicit_CFO_identity_input=False),actual_feature_extractor=report)
 mechanism=dict(epoch=e,optimizer_steps=steps,L_context_steps=steps,paired_calls=steps,extra_L_identity_forwards=steps*128,alpha=c['training_mechanism']['alpha'],full_five_RX=True,original_U_sampler_unchanged=True,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=c)
 expected=dict(role_ids=dict(L_s=['synthetic-L'],U_s=['synthetic-U'],V=['synthetic-V']));contract=dict(expected,native_role_comparison='EXACT_MATCH')
 return [done,initial,freeze,resolved,vars(d.make_args(c,'cpu')),mechanism,contract,expected]
def main():
 torch.set_num_threads(2);totals=d.totals()
 assert totals==dict(models=12,epochs=420,optimizer_steps=93240,extra_L_identity_forwards=11934720,U_observations=8164800,clean_decisions=2016000,metric_records=216)
 for row in d.rows():
  c=d.config(row);a=d.make_args(c,'cpu');assert a.epochs==row['epochs'] and (a.label_epochs,a.pseudo_epochs)==d.stages(row['epochs']) and a.use_unlabeled and a.use_ema_teacher and not a.use_aug and not a.use_concat_sat_channel_aug and a.lr==.0002 and a.weight_decay==.0001 and a.lambda_u==.16 and a.lambda_ent==.01
 first=d.config(d.rows()[0]);bad=dict(first,epochs=50);reject('budget/config mismatch',lambda:d.validate(bad))
 for key,value in [('head_fit_updates',1),('test_views',['practical_high']),('label_epochs',12),('model_seed',7)]:
  bad=dict(first);bad[key]=value;reject(key,lambda:d.validate(bad))
 names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
 initial=None;checked=0
 for row in d.rows()[:4]:
  c=d.config(row);m=api.build(c,'cpu').train();state={k:v.detach().clone() for k,v in m.state_dict().items()}
  if initial is None:initial=state
  else:assert set(state)==set(initial) and all(torch.equal(v,initial[k]) for k,v in state.items())
  W=api.classifier(m).weight;teacher=copy.deepcopy(m).eval();torch.manual_seed(12);x=torch.randn(128,2,256);y=torch.arange(128)%6
  with m._phase1_labeled_feature_context():out=m.forward_identity_only(x)
  ce=F.cross_entropy(out['tx_logits'],y);correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  total=ce+correction;assert torch.allclose(total.detach(),metrics['rdrop/paired_CE']+metrics['rdrop/symmetric_KL']*m._rdrop_alpha,atol=1e-6)
  params=dict(m.named_parameters());grads=torch.autograd.grad(total,[params[n] for n in names]+[W]);assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
  assert api.classifier(m).weight is W and m._rdrop_calls==1 and m._rdrop_extra_packets==128 and teacher._rdrop_calls==0 and all(p.grad is None for p in m.parameters())
  values=fixture(c,m._feature_learning_report);proof.validate_evidence(c,*values)
  for field,value in [('epoch',e:=c['epochs']-1),('optimizer_steps',1),('checkpoint_sources',['foreign']),('head_fit_updates',1)]:
   bad=copy.deepcopy(values);bad[0][field]=value;reject(c['arm']+' '+field,lambda:proof.validate_evidence(c,*bad))
  bad=copy.deepcopy(values);bad[-2]['role_ids']=dict(bad[-2]['role_ids'],U_s=['other']);reject(c['arm']+' physical role mismatch',lambda:proof.validate_evidence(c,*bad))
  bad=copy.deepcopy(values);bad[4]['source_fold_held_rx']=1;reject(c['arm']+' RX-fold ancestry',lambda:proof.validate_evidence(c,*bad))
  ck=dict(run_id=d.RUN,candidate_id=c['row_id'],epoch=c['epochs'],checkpoint_selection='final_only',a1_ema_successful_updates=c['epochs']*222,ema_model={'synthetic':True},args=values[4]);proof.validate_checkpoint(ck,c)
  bad=dict(ck,epoch=20 if c['epochs']==50 else 50);reject(c['arm']+' checkpoint budget',lambda:proof.validate_checkpoint(bad,c));checked+=1
  del m,teacher,out,total,correction,grads
 from . import evaluate
 old=evaluate.preflight;oldenv=os.environ.get('CUDA_VISIBLE_DEVICES');called=[]
 def closed(*args):called.append('prediction preflight');raise ValueError('Incomplete predictions')
 try:
  evaluate.preflight=closed;os.environ['CUDA_VISIBLE_DEVICES']='';reject('truth-last incomplete prediction',lambda:evaluate.score(first['row_id']));assert called==['prediction preflight']
 finally:
  evaluate.preflight=old
  if oldenv is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=oldenv
 from experiments.cvs_rdrop_fold20.resources import choose
 caps={g:dict(pids=set(),free_MiB=24000) for g in range(8)};caps[0]['pids']={1};assert choose(caps)==1 and choose({7:dict(pids=set(),free_MiB=12000)})==7 and choose({0:dict(pids={1,2,3},free_MiB=24000)}) is None
 print(json.dumps(dict(status='PASS',totals=totals,negative_cases=len(negative),negative_labels=negative,actual_budget_mode_models=checked,matched_initial_state_tensors=len(initial),synthetic_L_forward_packets=1024,body_and_defaultW_gradients_nonzero=True,checkpoint_loads=0,real_IQ_reads=0,query_reads=0,optimizer_steps=0,head_fit_updates=0)))
if __name__=='__main__':main()
