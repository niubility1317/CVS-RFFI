"""Reuse frozen artifacts, never inherit a historical trained model."""
from pathlib import Path
from . import design as d
from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS
def prove(row):
 if row not in d.historical_rows():raise ValueError('Only fixed6 historical controls')
 root=Path(row['root']);source=root/'source';done=d.read(source/'completion.json');c=done['config'];expected=d.native.config(next(r for r in d.native.rows() if r['row_id']==row['row_id']))
 ignore={'run_id','output_root','launch_owner'}
 if {k:v for k,v in c.items() if k not in ignore}!={k:v for k,v in expected.items() if k not in ignore} or c['run_id']!=row['run_id'] or c['output_root']!=str(source):raise ValueError('Historical native scientific recipe differs')
 steps=row['epochs']*222;args=d.read(source/'actual_native_args.json');wanted=vars(d.native.make_args(expected,'cpu'));wanted.update(run_id=c['run_id'],candidate_id=c['row_id'],output_dir=c['output_root'])
 if any(args.get(k)!=wanted.get(k) for k in RECIPE_FIELDS) or args.get('source_fold_held_rx') is not None or args.get('seed')!=row['model_seed']:raise ValueError('Historical actual native arguments differ')
 if done['status']!='SOURCE_TRAINED' or done['epoch']!=row['epochs'] or done['optimizer_steps']!=steps or done['logged_steps']!=steps or done['L_context_steps']!=steps or done['checkpoint_sources'] or done['ancestors'] or done['initialization']!='scratch' or done['head_fit_updates'] or done['target_access'] or done['target_evaluated']:raise ValueError('Historical fixed source trajectory incomplete')
 init=d.read(source/'initialization.json');freeze=d.read(source/'source_frozen.json');resolved=d.read(source/'resolved_config.json')
 if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['seed']!=row['model_seed'] or freeze['selection']!='final_only' or freeze['config']!=c or resolved['config']!=c:raise ValueError('Historical source ancestry/freeze differs')
 report=done['feature_extractor']
 if report['original_SIGTIME_RDrop']!=expected['training_mechanism'] or 'RDrop_paired_masks' in report or report['sinc_coordinates']!='RAW_HZ' or not done['native_fitted_probes_disabled'] or done['native_disabled_probe_calls']!=1:raise ValueError('Historical native feature/default-head execution differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
 pred=d.read(root/'prediction/complete.json');provenance=d.read(root/'prediction/provenance.json');complete=d.read(root/'completion.json')
 if pred['status']!='PREDICTIONS_COMPLETE' or pred['config']!=c or pred['count']!=168000 or pred['views']!=['clean'] or pred['readouts']!=['NATIVE_HEAD'] or pred['truth_read'] or pred['query_fit'] or pred['head_fit_updates']:raise ValueError('Historical native predictions incomplete')
 if provenance['config']!=c or not provenance['default_classifier_only'] or provenance['separate_head_training'] or provenance['truth_read'] or provenance['query_fit'] or complete['status']!='MODEL_PREDICTED_SCORED' or complete['config']!=c or not complete['truth_last'] or complete['head_fit_updates'] or complete['new_training_steps']!=steps:raise ValueError('Historical native scoring provenance differs')
 if any(not (root/n).is_file() for n in ('prediction/predictions.npz','scores/correctness.npz','scores/metrics.json')):raise ValueError('Historical native decision artifacts missing')
 return root
def preflight_all():return dict(status='VERIFIED',rows=[dict(row_id=r['row_id'],physical_run_id=r['run_id'],root=str(prove(r))) for r in d.historical_rows()],new_training_steps=0,checkpoint_loads=0,IQ_reads=0,truth_reads=0,target_scores_read=False)
