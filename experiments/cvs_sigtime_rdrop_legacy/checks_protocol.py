import copy,json
from pathlib import Path
from . import design as d
from .provenance import validate_evidence,validate_checkpoint
from .gate import qualify
from .checks import reject
def main():
 from . import model
 c=d.config(d.rows()[0]);m=model.build(c,'cpu');report=m._feature_learning_report;steps=c['epochs']*222;p=Path(c['output_root'])
 metric=dict(source_val_count=27000,source_val_accuracy=.98,source_val_worst_rx=.96,source_val_rx_accuracy={str(rx):.98 for rx in (1,3,4,6,8)})
 done=dict(status='SOURCE_TRAINED',config=c,epoch=c['epochs'],logged_steps=steps,optimizer_steps=steps,L_context_steps=steps,checkpoint_sources=[],ancestors=[],initialization='scratch',target_access=False,target_evaluated=False,baseline_recipe=c['training_recipe'],label_epochs=c['label_epochs'],pseudo_epochs=c['pseudo_epochs'],head_fit_updates=0,native_fitted_probes_disabled=True,native_disabled_probe_calls=1,identity_input_policy={'explicit_CFO_identity_input':False},feature_extractor=report,mask_counts=dict(time=0,frequency=0,PA=0,reference=0),final_source_metrics=metric)
 initial=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed'])
 freeze=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=c['epochs'],selection='final_only',target_access=False)
 resolved=dict(config=c,native_fitted_probes_disabled=True,identity_input_policy={'explicit_CFO_identity_input':False},actual_feature_extractor=report)
 args=vars(d.make_args(c,'cpu'));mechanism=dict(epoch=c['epochs'],optimizer_steps=steps,L_context_steps=steps,objective_calls=steps,extra_L_identity_forwards=0,positive_KL_calls=0,actual_augmented_input_reused=False,augmentation_redraws=0,label_smoothing=.01,use_aug=True,sinc_coordinates='RAW_HZ',original_U_sampler_unchanged=True,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=c)
 expected=dict(capsule_id='synthetic-contract-unit-fixture',split_id='fixture',role_ids={'L_s':['L'],'U_s':['U'],'V':['V']});contract=dict(copy.deepcopy(expected),native_role_comparison='EXACT_MATCH')
 fixture=[c,done,initial,freeze,resolved,args,mechanism,contract,expected]
 assert validate_evidence(*fixture)==p
 changes=[(1,'ancestors',['unverified']),(1,'checkpoint_sources',['oldbest']),(1,'target_access',True),(1,'head_fit_updates',1),(1,'optimizer_steps',steps-1),(2,'target_contact',True),(3,'selection','V_best'),(5,'label_smoothing',0.),(5,'use_aug',False),(5,'seed',1),(5,'baseline_ckpt','bad.pth'),(6,'extra_L_identity_forwards',128),(6,'augmentation_redraws',1),(6,'label_smoothing',0.),(7,'native_role_comparison','SUBSET')]
 for i,key,value in changes:
  bad=copy.deepcopy(fixture);bad[i][key]=value;reject(lambda:validate_evidence(*bad))
 bad=copy.deepcopy(fixture);bad[7]['role_ids']['L_s']=['different'];reject(lambda:validate_evidence(*bad))
 ck=dict(run_id=d.RUN,candidate_id=c['row_id'],epoch=c['epochs'],checkpoint_selection='final_only',a1_ema_successful_updates=steps,ema_model={'fixture':1},args=args);validate_checkpoint(ck,c)
 for key,value in [('run_id','other'),('epoch',50),('checkpoint_selection','target_best'),('ema_model',None)]:
  bad=copy.deepcopy(ck);bad[key]=value;reject(lambda:validate_checkpoint(bad,c))
 for e in d.BUDGETS:
  source={mode:dict(done=dict(optimizer_steps=e*222,target_access=False,target_evaluated=False,final_training_L_CE=.1,final_source_metrics=copy.deepcopy(metric)),mechanism=dict(positive_KL_calls=e*222 if mode!='ORIGINAL' else 0)) for mode in d.MODES}
  assert qualify(e,source)['qualified']
  for key in ('CE','V_mean','V_worst','KL'):
   bad=copy.deepcopy(source)
   if key=='CE':bad['RDROP']['done']['final_training_L_CE']=.501
   elif key=='V_mean':bad['RDROP']['done']['final_source_metrics']['source_val_accuracy']-=.011
   elif key=='V_worst':bad['RDROP']['done']['final_source_metrics']['source_val_worst_rx']-=.021
   else:bad['RDROP']['mechanism']['positive_KL_calls']=0
   assert not qualify(e,bad)['qualified']
  bad=copy.deepcopy(source);bad['RDROP']['done']['target_access']=True;reject(lambda:qualify(e,bad))
 print(json.dumps(dict(status='PASS',protocol_negative_cases=len(changes)+1+4+2,source_gate_boolean_failures=8,synthetic_metadata_fixture_only=True,source_candidate_or_target_data_reads=0,checkpoint_deserializations=0,head_fit_updates=0,optimizer_steps=0)))
if __name__=='__main__':main()
