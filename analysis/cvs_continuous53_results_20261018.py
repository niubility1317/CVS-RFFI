"""Recovery binding for the unchanged P52 four-row metadata analysis.

Only the recovered As row is new. Source readouts run before any formal
aggregation; the independent final stage reuses saved predictions, never IQ,
checkpoints, fitted heads or new network forwards.
"""
import argparse,csv,importlib.util
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('continuous53_bound52',HERE/'cvs_continuous52_results_20261018.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
PROJECT=core.PROJECT
RUN='20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01'
ORIGINAL_RUN='20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01'
REF=core.REF;ROOT=PROJECT/'runs'/RUN;ORIGINAL_ROOT=PROJECT/'runs'/ORIGINAL_RUN
SEED=core.SEED;VIEWS=core.VIEWS
CACHED_SOURCE=core.CACHED_SOURCE
read,write=core.read,core.write

def rows():
 result=[]
 for style in ('STAGED','CONT'):
  for arm in ('B00','B01'):
   run=REF if style=='STAGED' else ORIGINAL_RUN if arm=='B00' else RUN
   result.append(dict(key=style+'_'+arm,style=style,arm=arm,seed=SEED,
    # The old parser uses reused to distinguish staged20+10 from continuous30.
    reused=style=='STAGED',execution_reused=not(style=='CONT' and arm=='B01'),
    run=run,root=PROJECT/'runs'/run/(arm+'-s'+str(SEED))))
 return result

def completed_training(root):
 root=Path(root);row=next((r for r in rows() if r['root']==root and r['style']=='CONT'),None)
 if row is None:raise ValueError('Only exact old-control/new-recovery continuous roots')
 p=root/'source';done=read(p/'completion.json');c=done['config'];init=read(p/'initialization.json');frozen=read(p/'source_frozen.json');r=read(p/'result.json');contract=read(PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
 if (done['epoch'],done['optimizer_steps'],c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates'])!=(30,6660,30,13,17,6660) or c['run_id']!=row['run'] or c['row_id']!=root.name or c['arm']!=row['arm'] or c['model_seed']!=SEED or c['training_mode']!='single_continuous_E30' or c['epoch_offset'] or c['step_offset'] or c['parent_source'] or c['checkpoint_sources'] or c['new_Ag'] or c['Ag_mode']!='NONE':raise ValueError('Exact continuous30 row identity/budget differs')
 if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH' or read(p/'source_contract.json')['role_ids']!=contract['role_ids']:raise ValueError('Own fullsource scratch provenance differs')
 if done['target_access'] or done['head_fit_updates'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or r['config']!=c or r['head_fit_updates'] or r['U_truth_read'] or r['target_access']:raise ValueError('Own fixed defaulthead/source freeze required')
 args=read(p/'actual_native_args.json');binding=read(p/'seed_binding.json')
 if (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['seed'])!=(30,13,17,SEED) or args['baseline_ckpt'] or args['teacher_ckpt'] or binding['actual_F_factory_seed']!=SEED:raise ValueError('Actual native calendar/factory seed differs')
 return done

core.RUN=RUN;core.ROOT=ROOT;core.rows=rows;core.completed_training=completed_training
source_complete,formal_complete=core.source_complete,core.formal_complete
qualify,interaction,clean_cohort_effect=core.qualify,core.interaction,core.clean_cohort_effect

def failed_attempt():
 """Recount every completed failed-attempt epoch/step and opaque loader record."""
 root=ORIGINAL_ROOT/('B01-s'+str(SEED));p=root/'source'
 if not (ORIGINAL_ROOT/'failure.json').is_file() or (p/'final_ssdg.pth').exists():raise ValueError('Original unrecoverable failed As attempt must remain preserved')
 epochs=list(core.lines(p/'metrics_epoch.jsonl'))
 if [e['epoch'] for e in epochs]!=list(range(1,21)):raise ValueError('All original failed-attempt20 completed epochs required')
 for e in epochs:
  for k in ('lr','train_loss','train_grad_total','val_tx_acc','epoch_time_s'):core.numeric(e[k],k)
  if e['train_pseudo_truth_available']:raise ValueError('Original failed attempt U TX truth exposed')
 steps=0
 for q in core.lines(p/'step_metrics.jsonl.gz'):
  steps+=1
  if q['step']!=steps or q['epoch']!=(steps-1)//222+1:raise ValueError('Original complete successful-step log differs')
  if any(q['metrics'].get(k) or 0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad','train/pseudo_truth_available')):raise ValueError('Original failed attempt invalid completed update')
 if steps!=4440:raise ValueError('Original failed attempt did not complete exactly20 epochs')
 records=Counter();packets=Counter();unique={'L':set(),'U':set()}
 for q in core.lines(p/'training_sequence.jsonl.gz'):
  role=q['role']
  if role not in unique:raise ValueError('Only opaque original L/U loader roles')
  records[role]+=1;packets[role]+=len(q['ids']);unique[role].update(tuple(x) for x in q['ids'])
 if records['L']!=4441 or packets['L']!=568448 or packets['U']!=396900 or len(unique['L'])!=6300 or len(unique['U'])!=56700:raise ValueError('Original failed step or complete opaque loader coverage differs')
 return dict(run=ORIGINAL_RUN,row_id=root.name,status='FAILED_E21_OLD_E20_CHANNEL_BOUNDARY',completed_epochs=20,successful_updates=steps,L_NN_on_completed_steps=steps*768,As_packets_on_completed_steps=steps*64,known_NN_lower_bound=steps*768,total_NN=None,unrecorded_failed_step_and_U_eval_NN='UNKNOWN;not imputed',completed_epoch_wall_seconds=sum(e['epoch_time_s'] for e in epochs),sequence_records=dict(records),sequence_samples=dict(packets),unique_opaque_training_ids={k:len(v) for k,v in unique.items()},failed_step_L_input_count=128,final_checkpoint=False,recovery_initializer=False,U_TX_decoded=False,curve=epochs)

def source_analysis(out):
 failed=failed_attempt() # Complete metadata recount before any target stage.
 core.source_analysis(out)
 curves=[]
 for row in rows():
  path=out/(row['key']+'_training.json');training=read(path)
  training['execution_reused']=row['execution_reused']
  for q in training['curve']:q.update(reused=row['execution_reused'],execution_reused=row['execution_reused'])
  curves.extend(training['curve']);write(path,training)
 with (out/'training_curves.csv').open('w',newline='',encoding='utf8') as f:
  fields=list(dict.fromkeys(k for q in curves for k in q));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(curves)
 write(out/'failed_original_attempt.json',failed)
 done=read(out/'complete.json');done.update(new_rows=1,reused_rows=3,technical_failed_epochs=20,technical_failed_updates=4440);write(out/'complete.json',done)
 summary=read(out/'source_summary.json');summary.update(row_binding=[dict(r,root=str(r['root'])) for r in rows()],technical_failure_accounted_separately=True);write(out/'source_summary.json',summary)

def stage_assessment(result):
 r=result['rows'];changes={v:100*(r['CONT_B01']['views'][v]['accuracy']-r['CONT_B00']['views'][v]['accuracy']) for v in VIEWS}
 clean=changes['clean'];urban=float(np.mean([changes[v] for v in VIEWS[-3:]]))
 return dict(clean_delta_pp=clean,clean_max_drop_pp=.1,clean_cost_within_stage_limit=clean>=-.1-1e-10,urban_mean_delta_pp=urban,paired_accuracy_changes_pp=changes,observed_dual_gain=clean>0 and urban>0,model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,report_only=True,changes_source_criteria=False)

def matched_cohort_retention(data):
 if set(data)!=set(r['key'] for r in rows()):raise ValueError('All four truth-last saved prediction rows required')
 reference=data['CONT_B00'];y=np.asarray(reference['y']);ids=reference['ids'];cohort=np.asarray(reference['pred']['clean'])==y;den=int(cohort.sum());records=[]
 for row in rows():
  key=row['key'];x=data[key]
  if not np.array_equal(x['ids'],ids) or not np.array_equal(x['y'],y):raise ValueError('Matched continuous C0 cohort physical IDs/labels differ')
  for view in VIEWS:
   p=np.asarray(x['pred'][view])
   if p.shape!=y.shape:raise ValueError('Matched C0 cohort decisions must align')
   count=int(((p==y)&cohort).sum());records.append(dict(row_id=key,view=view,cohort='fixed_CONT_B00_final30_clean_correct',denominator=den,retained=count,retention=count/den if den else None,training_parent_cohort=False))
 return records

def final_analysis(out,source,truth):
 # Capture the already-read truth-last arrays; no second truth/prediction read.
 data={};collect=core.base.collect_row
 def captured(key,truth_side):
  result,arrays=collect(key,truth_side);data[key]=arrays;return result,arrays
 core.base.collect_row=captured
 try:q=core.final_analysis(out,source,truth)
 finally:core.base.collect_row=collect
 retention=matched_cohort_retention(data);core.csv_write(out/'fixed_CONT_B00_retention.csv',retention);q['fixed_CONT_B00_retention']=retention
 for row in rows():
  key=row['key'];q['rows'][key]['reused']=row['execution_reused'];q['cost'][key]['reused']=row['execution_reused']
 failed=read(source/'failed_original_attempt.json');old_smoke=read(ORIGINAL_ROOT/'synthetic_smoke.json')['NN']+read(ORIGINAL_ROOT/'real_checkpoint_smoke.json')['NN']
 if old_smoke!=1538:raise ValueError('Original failed-run shared launch smoke budget differs')
 q.update(new_models=1,reused_models=3,new_model_epochs=30,reused_model_epochs=90,new_updates=6660,reused_updates=19980,failed_original_attempt=failed,failed_original_model_epochs=20,failed_original_updates=4440,total_executed_model_epochs_including_failed=140,total_executed_updates_including_failed=31080,new_total_NN=sum(c['total'] for c in q['cost'].values() if not c['reused'])+q['shared_new_smoke_NN'],reused_total_NN=sum(c['total'] for c in q['cost'].values() if c['reused']),historical_original52_shared_smoke_NN=old_smoke,stage_assessment=stage_assessment(q),row_binding=[dict(r,root=str(r['root'])) for r in rows()])
 q['known_total_NN_lower_bound_including_failed']=q['new_total_NN']+q['reused_total_NN']+old_smoke+failed['known_NN_lower_bound']
 q['total_NN_including_failed']=None
 q['comparison_scope']='Four successful E30 models;one new recovered As,three reused;failed20 separately;no failed weights or formal results used as initialization'
 write(out/'analysis_summary.json',q)
 done=read(out/'complete.json');done.update(new_models=1,reused_models=3,failed_original_model_epochs=20,stage_assessment=q['stage_assessment']);write(out/'complete.json',done)
 return q

def coverage_analysis(out,source):
 import cvs_continuous52_coverage_20261018 as coverage
 import sys
 coverage.analysis=sys.modules[__name__]
 coverage.run(out,source)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','coverage','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path);p.add_argument('--truth',type=Path);a=p.parse_args()
 if a.mode in ('source','coverage') and a.truth is not None:raise ValueError('Source/coverage cannot receive truth')
 if a.mode=='source':source_analysis(a.output)
 elif a.mode=='coverage':
  if a.source_analysis is None:raise ValueError('Coverage requires complete source analysis')
  coverage_analysis(a.output,a.source_analysis)
 else:
  if a.truth is None or a.source_analysis is None:raise ValueError('Final requires independent source completion and truth-side path')
  final_analysis(a.output,a.source_analysis,a.truth)
if __name__=='__main__':main()
