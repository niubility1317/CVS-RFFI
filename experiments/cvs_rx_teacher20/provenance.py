from pathlib import Path
from collections import Counter
from . import design as d
from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS,report_validator
def validate_args(actual,c):
 d.validate(c);actual=actual.get('args',actual);expected=d.make_args(c,'cpu')
 if any(actual.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or any(v!=0 for k,v in actual.items() if k.startswith('lambda_')):raise ValueError('Actual pure-CE teacher recipe or inheritance differs')
def validate_evidence(c,done,initial,freeze,resolved,args,contract,expected,subset,progress):
 d.validate(c);p=Path(c['output_root']);exposure=c['teacher_exposure'];roles=contract.get('role_ids',{})
 if done.get('status')!='SOURCE_TRAINED' or done.get('config')!=c or done.get('epoch')!=20 or any(done.get(k)!=4440 for k in ('optimizer_steps','logged_steps','L_context_steps')) or done.get('target_access') is not False or done.get('target_evaluated') is not False or done.get('initialization')!='scratch' or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[]:raise ValueError('Complete own scratch source teacher required')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='disabled',seed=d.SEED) or freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=20,selection='final_only',target_access=False):raise ValueError('Own empty ancestry and fixed final20 freeze required')
 if resolved.get('config')!=c or done.get('baseline_recipe')!=c['training_recipe'] or done.get('label_epochs')!=20 or done.get('pseudo_epochs')!=0:raise ValueError('Actual pure L20 schedule differs')
 if done.get('head_fit_updates')!=0 or done.get('native_fitted_probes_disabled') is not True or done.get('native_disabled_probe_calls')!=1 or resolved.get('native_fitted_probes_disabled') is not True:raise ValueError('Default head must be the only classifier training')
 if any(x.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False for x in (done,resolved)):raise ValueError('NoExplicitCFO input contract differs')
 report=done.get('feature_extractor',{});report_validator(report,c)
 if report.get('teacher_exposure')!=exposure or report.get('paired_dropout_objective') is not None or resolved.get('actual_feature_extractor')!=report or any(done.get('mask_counts',{'missing':1}).values()):raise ValueError('Teacher feature path/exposure differs')
 if set(roles)!=set(expected['role_ids']) or {k:len(v) for k,v in roles.items()}!={'L_s':5040,'U_s':0,'V':21600} or any(len(set(v))!=len(v) or not set(v)<=set(expected['role_ids'][k]) for k,v in roles.items()):raise ValueError('Teacher effective roles escape original carrier')
 if contract.get('teacher_exposure')!=exposure or contract.get('source_receivers')!=exposure['train_RX'] or contract.get('carrier_contract_ref')!=d.SOURCE or contract.get('native_role_comparison')!='EXACT_MATCH_CARRIER_THEN_TEACHER_SUBSET':raise ValueError('Teacher subset contract identity differs')
 records=subset.get('train_records',[])
 if subset.get('config')!=c or subset.get('checkpoint_sources')!=[] or subset.get('ancestors')!=[] or subset.get('U_iterator_calls')!=0 or subset.get('held_V_IQ_read_before_freeze') is not False or subset.get('fit_V_RX_counts')!={str(rx):5400 for rx in d.FIT}:raise ValueError('Actual source exposure proof incomplete')
 if len(records)!=5040 or len({r['base_index'] for r in records})!=5040 or set(r['physical_id'] for r in records)!=set(roles['L_s']) or set(r['RX'] for r in records)!=set(exposure['train_RX']):raise ValueError('Actual selected L records differ')
 cells=Counter((r['TX'],r['day']) for r in records);full=Counter((r['TX'],r['RX'],r['day']) for r in records)
 if set(cells)!={(t,day) for t in range(6) for day in (1,2,3)} or set(cells.values())!={280} or set(full.values())!={exposure['per_RX_TX_day']}:raise ValueError('Matched teacher TX/day/RX exposure differs')
 if progress.get('epoch')!=20 or progress.get('verified_L_steps')!=4440 or not 0<progress.get('unique_L_packets',0)<=5040 or progress.get('U_iterator_calls')!=0 or progress.get('EMA') is not False or progress.get('target_access') is not False or progress.get('head_fit_updates')!=0:raise ValueError('Actual teacher exposure counters incomplete')
 metric=done.get('final_source_metrics',{})
 if metric.get('source_val_count')!=21600 or set(metric.get('source_val_rx_accuracy',{}))!=set(map(str,d.FIT)):raise ValueError('Fit-only source V required')
 validate_args(args,c);return p
def source_proof(c):
 p=Path(c['output_root']);return validate_evidence(c,*[d.read(p/name) for name in ('completion.json','initialization.json','source_frozen.json','resolved_config.json','actual_native_args.json','source_contract.json')],d.read(d.SOURCE),d.read(p/'teacher_subset.json'),d.read(p/'teacher_progress.json'))
def validate_checkpoint(ck,c):
 if ck.get('epoch')!=20 or ck.get('run_id')!=d.RUN or ck.get('candidate_id')!=c['row_id'] or ck.get('checkpoint_selection')!='final_only' or ck.get('a1_ema_successful_updates')!=4440 or ck.get('ema_model') is not None:raise ValueError('Own fixed pure-CE/noEMA final teacher checkpoint required')
 validate_args(ck.get('args',{}),c)
