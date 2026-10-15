from pathlib import Path
from . import design as d
from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS
def validate_args(actual,c):
 d.validate(c);actual=actual.get('args',actual);expected=d.make_args(c,'cpu')
 if any(actual.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or actual.get('source_fold_held_rx')!=1 or actual.get('source_fold_training_receivers')!='3,4,6,8':raise ValueError('Actual source-fold SSL recipe differs')
def source_proof(c):
 d.validate(c);p=Path(c['output_root']);done,initial,freeze,resolved,args,contract,progress,fold=[d.read(p/n) for n in ('completion.json','initialization.json','source_frozen.json','resolved_config.json','actual_native_args.json','source_contract.json','exchange_progress.json','source_fold.json')]
 if done.get('status')!='SOURCE_TRAINED' or done.get('config')!=c or done.get('epoch')!=20 or any(done.get(k)!=4440 for k in ('logged_steps','optimizer_steps','L_context_steps')) or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[] or done.get('initialization')!='scratch':raise ValueError('Complete own scratch trajectory required')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=d.SEED):raise ValueError('No ancestor/teacher inherited RX1 exposure permitted')
 if freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=20,selection='final_only',target_access=False) or resolved.get('config')!=c or done.get('baseline_recipe')!=c['training_recipe']:raise ValueError('Own fixed final20 freeze required')
 if done.get('target_access') is not False or done.get('target_evaluated') is not False or done.get('head_fit_updates')!=0 or done.get('native_fitted_probes_disabled') is not True or done.get('native_disabled_probe_calls')!=1:raise ValueError('Source only default joint head required')
 report=done.get('feature_extractor',{});reference=c['feature_extractor']
 fields=('arm','layer','local_dimension','windows','descriptor_width','output_embedding','classifier_changed','gain_initial','gain_bound','time_coordinate','per_packet')
 if any(report.get(k)!=reference[k] for k in fields) or report.get('sinc_coordinates')!='RAW_HZ' or report.get('original_SIGTIME_RDrop')!=d.native_config()['training_mechanism']:raise ValueError('Actual original SIG_TIME structure differs')
 if report.get('conditional_residual_exchange')!=c['training_mechanism'] or resolved.get('actual_feature_extractor')!=report or any(done.get('mask_counts',{'missing':1}).values()):raise ValueError('Actual feature mechanism differs')
 if any(v.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False for v in (done,resolved)):raise ValueError('NoExplicitCFO required')
 roles=contract.get('role_ids',{});expected=d.read(d.SOURCE)['role_ids']
 if set(roles)!=set(expected) or {k:len(v) for k,v in roles.items()}!={'L_s':5040,'U_s':45360,'V':21600} or any(len(set(v))!=len(v) or not set(v)<=set(expected[k]) for k,v in roles.items()):raise ValueError('Effective four source RX roles differ')
 if contract.get('held_RX')!=1 or contract.get('source_receivers')!=list(d.FIT) or contract.get('native_role_comparison')!='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD' or not fold.get('all_training_calibration_probe_loaders_RX1_excluded'):raise ValueError('True all-path RX1 exclusion required')
 if progress.get('config')!=c or progress.get('epoch')!=20 or progress.get('actual_steps')!=4440 or progress.get('extra_L_identity_packets')!=568320 or not 0<=progress.get('swapped_packets',-1)<=progress.get('cohort_packets',-1)<=568320 or progress.get('max_energy_relative_error',1)>2e-6:raise ValueError('Actual exchange signal/budget invalid')
 if done.get('label_epochs')!=13 or done.get('pseudo_epochs')!=7 or done.get('final_source_metrics',{}).get('source_val_count')!=21600:raise ValueError('Original fixed source schedule/V required')
 validate_args(args,c);return p
def validate_checkpoint(ck,c):
 if ck.get('run_id')!=d.RUN or ck.get('candidate_id')!=c['row_id'] or ck.get('epoch')!=20 or ck.get('checkpoint_selection')!='final_only' or ck.get('a1_ema_successful_updates')!=4440 or not ck.get('ema_model'):raise ValueError('Own fixed final20 checkpoint and ownEMA required')
 validate_args(ck.get('args',{}),c)
