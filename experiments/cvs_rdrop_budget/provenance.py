from pathlib import Path
from . import design as d
from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS,report_validator
def validate_args(actual,c):
 d.validate(c);actual=actual.get('args',actual);expected=d.make_args(c,'cpu')
 if any(actual.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or actual.get('source_fold_held_rx') is not None:raise ValueError('Actual full-source budget/recipe or RX exclusion differs')
def validate_evidence(c,done,initial,freeze,resolved,args,mechanism,contract,expected):
 d.validate(c);p=Path(c['output_root']);e=c['epochs'];steps=e*222
 if (done.get('status')!='SOURCE_TRAINED' or done.get('config')!=c or done.get('epoch')!=e
  or any(done.get(k)!=steps for k in ('logged_steps','optimizer_steps','L_context_steps'))
  or done.get('target_access') is not False or done.get('target_evaluated') is not False
  or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[] or done.get('initialization')!='scratch'
  or done.get('label_epochs')!=c['label_epochs'] or done.get('pseudo_epochs')!=c['pseudo_epochs'] or done.get('baseline_recipe')!=c['training_recipe']):raise ValueError('Own source training identity/budget/ancestry differs')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):raise ValueError('Own scratch student/EMA required')
 if freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=e,selection='final_only',target_access=False) or resolved.get('config')!=c:raise ValueError('Own fixed final checkpoint required')
 if done.get('head_fit_updates')!=0 or done.get('native_fitted_probes_disabled') is not True or done.get('native_disabled_probe_calls')!=1 or resolved.get('native_fitted_probes_disabled') is not True:raise ValueError('Separate fitted heads forbidden')
 if any(x.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False for x in (done,resolved)):raise ValueError('NoExplicitCFO input policy differs')
 report=done.get('feature_extractor',{});report_validator(report,c)
 if report.get('paired_dropout_objective')!=c['training_mechanism'] or resolved.get('actual_feature_extractor')!=report or any(done.get('mask_counts',{'missing':1}).values()):raise ValueError('Actual default feature mechanism differs')
 if mechanism!=dict(epoch=e,optimizer_steps=steps,L_context_steps=steps,paired_calls=steps,extra_L_identity_forwards=steps*128,alpha=c['training_mechanism']['alpha'],full_five_RX=True,original_U_sampler_unchanged=True,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=c):raise ValueError('Actual double-forward mechanism/budget incomplete')
 metric=done.get('final_source_metrics',{})
 if metric.get('source_val_count')!=27000 or set(metric.get('source_val_rx_accuracy',{}))!=set(map(str,(1,3,4,6,8))):raise ValueError('Full original source V required')
 validate_args(args,c)
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(contract,expected);return p
def source_proof(c):
 p=Path(c['output_root']);return validate_evidence(c,*[d.read(p/name) for name in ('completion.json','initialization.json','source_frozen.json','resolved_config.json','actual_native_args.json','source_training_mechanism.json','source_contract.json')],d.read(d.SOURCE))
def validate_checkpoint(ck,c):
 if ck.get('run_id')!=d.RUN or ck.get('candidate_id')!=c['row_id'] or ck.get('epoch')!=c['epochs'] or ck.get('checkpoint_selection')!='final_only' or ck.get('a1_ema_successful_updates')!=c['epochs']*222 or not ck.get('ema_model'):raise ValueError('Own fixed-final budget checkpoint/EMA required')
 validate_args(ck.get('args',{}),c)
def own_model(c,device):
 import torch
 from . import model
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
 result=model.build(c,device);result.load_state_dict(ck['model'],strict=True);return result.eval().requires_grad_(False)
