from pathlib import Path
from . import design as d
from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS
def validate_args(actual,c):
 d.validate(c);actual=actual.get('args',actual);expected=d.make_args(c,'cpu')
 if any(actual.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or actual.get('source_fold_held_rx') is not None or actual.get('label_smoothing')!=.01 or actual.get('use_aug') is not True or actual.get('seed')!=c['model_seed']:raise ValueError('Actual original SIG_TIME recipe or full-source roles differ')
def validate_evidence(c,done,initial,freeze,resolved,args,mechanism,contract,expected):
 d.validate(c);p=Path(c['output_root']);e=c['epochs'];steps=e*222;paired=c['mode']!='ORIGINAL'
 if done.get('status')!='SOURCE_TRAINED' or done.get('config')!=c or done.get('epoch')!=e or any(done.get(k)!=steps for k in ('logged_steps','optimizer_steps','L_context_steps')) or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[] or done.get('initialization')!='scratch' or done.get('target_access') is not False or done.get('target_evaluated') is not False:raise ValueError('Own complete source-only scratch trajectory required')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):raise ValueError('Original student/ownEMA ancestry must be empty')
 if freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=e,selection='final_only',target_access=False) or resolved.get('config')!=c or done.get('baseline_recipe')!=c['training_recipe'] or done.get('label_epochs')!=c['label_epochs'] or done.get('pseudo_epochs')!=c['pseudo_epochs']:raise ValueError('Fixed original schedule and final checkpoint required')
 if done.get('head_fit_updates')!=0 or done.get('native_fitted_probes_disabled') is not True or done.get('native_disabled_probe_calls')!=1 or resolved.get('native_fitted_probes_disabled') is not True:raise ValueError('Only original jointly trained default head allowed')
 if any(v.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False for v in (done,resolved)):raise ValueError('NoExplicitCFO input must remain')
 report=done.get('feature_extractor',{});reference=dict(c['feature_extractor'],arm=c['combination']['SIG_TIME'])
 fields=('arm','layer','local_dimension','windows','descriptor_width','output_embedding','classifier_changed','gain_initial','gain_bound','time_coordinate','per_packet')
 if any(report.get(k)!=reference[k] for k in fields) or report.get('sinc_coordinates')!=c['combination']['Sinc_coordinates'] or report.get('original_SIGTIME_RDrop')!=c['training_mechanism'] or resolved.get('actual_feature_extractor')!=report or any(done.get('mask_counts',{'missing':1}).values()):raise ValueError('Actual old SIG_TIME feature graph differs')
 expected_mechanism=dict(epoch=e,optimizer_steps=steps,L_context_steps=steps,objective_calls=steps,extra_L_identity_forwards=steps*128 if paired else 0,actual_augmented_input_reused=paired,augmentation_redraws=0,label_smoothing=.01,use_aug=True,sinc_coordinates=c['combination']['Sinc_coordinates'],original_U_sampler_unchanged=True,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=c)
 if any(mechanism.get(k)!=v for k,v in expected_mechanism.items()) or not 0<=mechanism.get('positive_KL_calls',-1)<=steps or (not paired and mechanism.get('positive_KL_calls')!=0):raise ValueError('Actual old input reuse/two-forward mechanism or budget incomplete')
 metric=done.get('final_source_metrics',{})
 if metric.get('source_val_count')!=27000 or set(metric.get('source_val_rx_accuracy',{}))!=set(map(str,(1,3,4,6,8))):raise ValueError('Complete original source V required')
 validate_args(args,c)
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(contract,expected);return p
def source_proof(c):
 if c['baseline_reuse']:return reference_proof(c)
 p=Path(c['output_root']);unit=d.read(p/'sinc_optimizer_observation.json')
 if unit['status']!='VERIFIED' or unit['steps']!=11100 or unit['mode']!=c['sinc_optimizer_units']['mode'] or unit['head_fit_updates'] or unit['U_truth_read'] or unit['extra_optimizer_updates']:raise ValueError('Actual fixed Sinc optimizer witness incomplete')
 return validate_evidence(c,*[d.read(p/n) for n in ('completion.json','initialization.json','source_frozen.json','resolved_config.json','actual_native_args.json','source_training_mechanism.json','source_contract.json')],d.read(d.SOURCE))
def validate_checkpoint(ck,c):
 if ck.get('run_id')!=d.RUN or ck.get('candidate_id')!=c['row_id'] or ck.get('epoch')!=c['epochs'] or ck.get('checkpoint_selection')!='final_only' or ck.get('a1_ema_successful_updates')!=c['epochs']*222 or not ck.get('ema_model'):raise ValueError('Own fixed-final checkpoint/EMA required')
 validate_args(ck.get('args',{}),c)
def own_model(c,device):
 import torch
 from . import model
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
 m=model.build(c,device);m.load_state_dict(ck['model'],strict=True);return m.eval().requires_grad_(False)


def reference_proof(c):
 """Reuse the verified original fixed seed, including its technical recovery root."""
 d.validate(c)
 if not c['baseline_reuse']:raise ValueError('Only completed fixed RDrop references may be reused')
 p=d.reference_source(c['model_seed']);done=d.read(p/'completion.json');actual=done['config'];ref=d.reference_config(c['model_seed']);metadata={'run_id','output_root','launch_owner'}
 if {k:v for k,v in actual.items() if k not in metadata}!={k:v for k,v in ref.items() if k not in metadata}:raise ValueError('Reference original RDrop scientific config differs')
 if actual['output_root']!=p.as_posix() or actual['run_id']!=p.parent.parent.name:raise ValueError('Reference physical recovery identity differs')
 if any(done.get(k)!=11100 for k in ('logged_steps','optimizer_steps','L_context_steps')) or done['epoch']!=50 or done['checkpoint_sources'] or done['ancestors'] or done['target_access'] or done['target_evaluated'] or done['head_fit_updates'] or not done['native_fitted_probes_disabled']:raise ValueError('Reference source trajectory is incomplete or contaminated')
 initial=d.read(p/'initialization.json');freeze=d.read(p/'source_frozen.json');resolved=d.read(p/'resolved_config.json');args=d.read(p/'actual_native_args.json');mechanism=d.read(p/'source_training_mechanism.json')
 if not initial['scratch_only'] or initial['checkpoint_sources'] or initial['ancestors'] or initial['target_contact'] or initial['seed']!=c['model_seed'] or initial['teacher_origin']!='own scratch student only':raise ValueError('Reference source ancestry differs')
 if freeze['config']!=actual or freeze['selection']!='final_only' or freeze['epoch']!=50 or freeze['target_access'] or resolved['config']!=actual or resolved['commit']!=d.MODEL_CODE_COMMIT:raise ValueError('Reference code/selection/source identity differs')
 expected=d.parent.make_args(ref,'cpu')
 if any(args.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or args.get('seed')!=c['model_seed'] or args.get('source_fold_held_rx') is not None:raise ValueError('Reference actual original native recipe differs')
 if mechanism['objective_calls']!=11100 or mechanism['extra_L_identity_forwards']!=1420800 or mechanism['config']!=actual or not mechanism['actual_augmented_input_reused'] or mechanism['head_fit_updates'] or mechanism['U_truth_read']:raise ValueError('Reference actual original RDrop objective differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p
