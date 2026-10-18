"""Three fixed model seeds: source-only saved-logit E30/E50 state recount."""
import argparse,contextlib,json,math,time
from pathlib import Path
import cvs_budget50_source_states56_20261018 as core
np=core.np

SEEDS=(2026092701,2026092702,2026092703)
RUN30_01='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
RUN50_01='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
RUN30_23='20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02'
RUN50_23='20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01'
MAIN_PAIRS=(('AS50','NOAS50'),('AS50','AS30'))
CONTEXT=('scene','draw','dimension','stratum')
class SourceNotReady(RuntimeError):pass


def seed_runs(seed):
 if type(seed) is not int or seed not in SEEDS:raise ValueError('Only all three fixed model seeds are allowed')
 return (RUN30_01,RUN50_01) if seed==SEEDS[0] else (RUN30_23,RUN50_23)

@contextlib.contextmanager
def bind_seed(seed):
 run30,run50=seed_runs(seed);previous=(core.SEED,core.RUN30,core.RUN50)
 try:
  core.SEED,core.RUN30,core.RUN50=seed,run30,run50
  yield core
 finally:core.SEED,core.RUN30,core.RUN50=previous

def summary_path(project,seed):
 return Path(project)/'runs'/seed_runs(seed)[1]/'analysis_auto_r01/source_summary.json'

def map_source_summary(summary,seed):
 """Map owner-produced source entries in memory; never synthesize missing F1."""
 expected_run=seed_runs(seed)[1]
 if summary.get('status')!='SOURCE_ANALYZED' or summary.get('target_scores_read') is not False or summary.get('run_id')!=expected_run:raise ValueError('Completed source-only owner summary/run required')
 source=summary.get('summary',{})
 if seed==SEEDS[0]:
  if summary.get('model_seed_count')!=1 or set(source)!=set(core.ARMS):raise ValueError('Fixed seed01 source summary keys differ')
  mapped={arm:source[arm] for arm in core.ARMS}
 else:
  expected={arm+'-s'+str(s) for s in SEEDS for arm in core.ARMS}
  if summary.get('model_seeds')!=list(SEEDS) or set(source)!=expected:raise ValueError('All fixed source seed-labelled keys required')
  mapped={arm:source[arm+'-s'+str(seed)] for arm in core.ARMS}
 for arm,value in mapped.items():
  if not isinstance(value,dict) or not isinstance(value.get('panel'),dict):raise ValueError('Saved owner source panel required: '+arm)
  for scene in ('clean',*core.SCENES):
   metrics=value['panel'].get(scene,{})
   if any(type(metrics.get(k)) not in (int,float) or not math.isfinite(metrics[k]) for k in ('accuracy','macro_F1')):raise ValueError('Saved source accuracy/F1 absent; no synthetic fallback')
 return dict(status='SOURCE_ANALYZED',target_scores_read=False,summary=mapped)

def required_inputs(project):
 result=[]
 for seed in SEEDS:
  with bind_seed(seed):
   for path in core.rows(project).values():result.extend([path/'source_qualification/complete.json',path/'source_qualification/predictions.npz'])
  result.append(summary_path(project,seed))
 panel=Path(project)/'runs'/core.PANEL_RUN/'diagnostic'
 result.extend([panel/'data_roles.json',panel/'observations/completion.json',panel/'observations/physics.npz',panel/'observations/physical_records.jsonl.gz'])
 return list(dict.fromkeys(result))

def check_ready(project):
 missing=[str(path) for path in required_inputs(project) if not path.is_file()]
 if missing:raise SourceNotReady('Source artifacts not ready: '+json.dumps(missing))
 summaries={}
 for seed in SEEDS:
  path=summary_path(project,seed)
  try:value=core.read(path)
  except json.JSONDecodeError as exc:raise SourceNotReady('Source summary is still being written: '+str(path)) from exc
  if value.get('status')!='SOURCE_ANALYZED':raise SourceNotReady('Owner source summary incomplete: '+str(path))
  summaries[seed]=map_source_summary(value,seed)
 return summaries

def stats3(values):
 if set(values)!=set(SEEDS):raise ValueError('All three fixed seeds required; no skipped or substituted seed')
 seq=[values[seed] for seed in SEEDS]
 if all(value is None for value in seq):return dict(values=seq,mean=None,sample_SD=None,valid_model_seed_count=0,model_seed_count=3,empty_cell=True)
 if any(value is None or not math.isfinite(float(value)) for value in seq):raise ValueError('Incomplete seed values cannot produce a paired mean')
 x=np.asarray(seq,float)
 return dict(values=x.tolist(),mean=float(x.mean()),sample_SD=float(x.std(ddof=1)),valid_model_seed_count=3,model_seed_count=3,empty_cell=False)

def index_records(records):
 result={}
 for row in records:
  key=tuple(row[k] for k in CONTEXT)
  if key in result:raise ValueError('Duplicate source stratum')
  result[key]=row
 return result

def aggregate(per_seed):
 if set(per_seed)!=set(SEEDS):raise ValueError('Complete original three-seed state matrix required')
 tables=[]
 for new,control in MAIN_PAIRS:
  byseed={seed:index_records([r for r in per_seed[seed]['effects'] if (r['new'],r['control'])==(new,control)]) for seed in SEEDS}
  tables.append((new+'_minus_'+control,byseed,'accuracy_delta_pp','macro_f1_delta_pp'))
 tables.append(('budget_x_As',{seed:index_records(per_seed[seed]['interaction']) for seed in SEEDS},'budget_x_As_accuracy_pp',None))
 result=[]
 for name,byseed,accuracy_field,f1_field in tables:
  keys=list(byseed[SEEDS[0]])
  if not keys:raise ValueError('Required paired comparison has no source strata')
  if any(set(byseed[seed])!=set(keys) for seed in SEEDS):raise ValueError('All seeds must retain identical state/bin/draw cells, including empty cells')
  for key in keys:
   rows={seed:byseed[seed][key] for seed in SEEDS};counts={int(row['count']) for row in rows.values()}
   if len(counts)!=1:raise ValueError('Fixed source physical observation counts differ across model seeds')
   count=counts.pop();a=stats3({seed:rows[seed][accuracy_field] for seed in SEEDS});f=stats3({seed:rows[seed][f1_field] for seed in SEEDS}) if f1_field else None
   if (count==0)!=a['empty_cell']:raise ValueError('Empty source cell must not become a fabricated numeric seed result')
   record=dict(comparison=name,**dict(zip(CONTEXT,key)),observations_per_model_seed=count,model_seed_count=3,valid_model_seed_count=a['valid_model_seed_count'],empty_cell=a['empty_cell'],small_cell=count<30,paired_mean_accuracy_delta_pp=a['mean'],paired_sample_SD_accuracy_delta_pp=a['sample_SD'],paired_mean_macro_f1_delta_pp=f['mean'] if f else None,paired_sample_SD_macro_f1_delta_pp=f['sample_SD'] if f else None)
   for seed,value in zip(SEEDS,a['values']):record['s'+str(seed)+'_accuracy_delta_pp']=value
   if f:
    for seed,value in zip(SEEDS,f['values']):record['s'+str(seed)+'_macro_f1_delta_pp']=value
   result.append(record)
 return result

def save_seed(project,out,seed,mapped_summary,code_commit):
 start=time.perf_counter();out.mkdir()
 with bind_seed(seed):
  data,physics,audit=core.load_inputs(project);strata,effects,interaction,overall=core.analyze(data,physics);reconciled=core.verify_source_summary(mapped_summary,overall)
  for records in (effects,interaction):
   for row in records:row['model_seed']=seed
  core.csv_write(out/'source_strata.csv',[{k:v for k,v in row.items() if k!='confusion'} for row in strata]);core.write(out/'source_confusions.json',strata);core.csv_write(out/'paired_effects.csv',effects);core.csv_write(out/'budget_interaction.csv',interaction)
  cohort_count=int((data['NOAS30']['clean'].argmax(-1)==data['NOAS30']['TX']).sum())
  summary=dict(status='SOURCE_STATE_ANALYZED',model_seed=seed,model_seed_count=1,condition='full',source_unique_V=1800,channel_draws=2,fixed_clean_cohort='This seed own NOAS30 clean correct; unchanged across its four rows and every scene',fixed_clean_cohort_count=cohort_count,rows=overall,alignment=audit,source_summary_path=str(summary_path(project,seed)),source_summary_reconciliation=reconciled,bins=core.bins_description(),logit_scale=core.LOGIT_SCALE,logit_scale_confirmed_from_code=True,source_high_urban_available=False,fullV_27000_not_recounted=True,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,training=0,scoring=0,formal_files_read=False,no_promotion=True)
  core.write(out/'source_summary.json',summary);core.plot(out,strata)
  report=dict(status='SOURCE_STATE_ANALYZED',model_seed=seed,code_commit=code_commit,run30=core.RUN30,run50=core.RUN50,model_seed_count=1,source_strata=len(strata),paired_strata=len(effects),interaction_strata=len(interaction),source_summary_reconciliation=reconciled,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,training=0,scoring=0,formal_files_read=False,no_promotion=True,wall_seconds=time.perf_counter()-start)
  core.write(out/'report.json',report)
 return dict(strata=strata,effects=effects,interaction=interaction,overall=overall,cohort_count=cohort_count,report=report)

def plot_three_seed(out,records):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(18,10));comparisons=['AS50_minus_AS30','AS50_minus_NOAS50'];colors=['#d97706','#2563eb','#059669'];labels=[s+'/'+l for s in core.STATES for l in core.LOCKS]
 for ri,scene in enumerate(core.SCENES[1:]):
  for ci,comparison in enumerate(comparisons):
   ax=axes[ri,ci];lookup={r['stratum']:r for r in records if (r['scene'],r['draw'],r['dimension'],r['comparison'])==(scene,-1,'state_lock',comparison)};seq=[lookup[label] for label in labels];xx=np.arange(len(labels))
   means=[r['paired_mean_accuracy_delta_pp'] if r['paired_mean_accuracy_delta_pp'] is not None else np.nan for r in seq];sd=[r['paired_sample_SD_accuracy_delta_pp'] if r['paired_sample_SD_accuracy_delta_pp'] is not None else np.nan for r in seq]
   ax.errorbar(xx,means,yerr=sd,c='black',marker='D',capsize=3,label='paired mean +/- sample SD',linestyle='none',zorder=4)
   for i,seed in enumerate(SEEDS):
    values=[r['s'+str(seed)+'_accuracy_delta_pp'] if r['s'+str(seed)+'_accuracy_delta_pp'] is not None else np.nan for r in seq];ax.scatter(xx+(i-1)*.12,values,s=24,c=colors[i],label=str(seed),zorder=5)
   ax.axhline(0,c='gray',lw=1);ax.set(xticks=xx,xticklabels=[label+'\nn='+str(r['observations_per_model_seed']) for label,r in zip(labels,seq)],ylabel='Paired accuracy change (percentage points)',title=scene+' / '+comparison);ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Source V / all three fixed model seeds / full residual only\nPooled two fixed channel draws; n is observations per model seed, not independent model seeds');fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(out/('three_seed_state_budget_deltas.'+ext),dpi=170)
 plt.close(fig)

def run(project,out,code_commit):
 out=Path(out)
 if out.exists():raise FileExistsError('Immutable new output root required; reconcile existing results')
 started=time.perf_counter();cpu_started=time.process_time();summaries=check_ready(project);out.mkdir();per_seed={}
 for seed in SEEDS:per_seed[seed]=save_seed(project,out/('s'+str(seed)),seed,summaries[seed],code_commit)
 aggregate_records=aggregate(per_seed);core.csv_write(out/'three_seed_paired_summary.csv',aggregate_records);core.write(out/'three_seed_paired_summary.json',aggregate_records)
 for name,key in [('all_seed_paired_effects.csv','effects'),('all_seed_budget_interaction.csv','interaction')]:core.csv_write(out/name,[row for seed in SEEDS for row in per_seed[seed][key]])
 overall=[dict(model_seed=seed,row_id=arm,scene=scene,draw=draw,**{k:v for k,v in metric.items() if k!='confusion'}) for seed in SEEDS for arm,views in per_seed[seed]['overall'].items() for name,metric in views.items() for scene,draw_text in [name.rsplit('__',1)] for draw in [int(draw_text)]]
 core.csv_write(out/'all_seed_overall.csv',overall);plot_three_seed(out,aggregate_records)
 report=dict(status='THREE_SEED_SOURCE_STATE_ANALYZED',code_commit=code_commit,model_seeds=list(SEEDS),model_seed_count=3,source_role='V',unique_source_physical_samples=1800,channel_draws=2,draws_are_not_model_seeds=True,condition='full',logical_rows=12,all_three_model_seeds_required=True,per_seed_fixed_NOAS30_clean_correct_count={str(seed):per_seed[seed]['cohort_count'] for seed in SEEDS},source_summary_reconciliations=96,aggregate_strata=len(aggregate_records),empty_cells_retained=True,source_high_urban_available=False,fullV_27000_not_recounted=True,bins=core.bins_description(),NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,training=0,scoring=0,formal_files_read=False,no_promotion=True,target_feedback=False,physical_causality_claimed=False,counterfactual_classification=False,cpu_threads=2,cuda_visible_devices='',cpu_seconds=time.process_time()-cpu_started,wall_seconds=time.perf_counter()-started,artifacts=['all_seed_overall.csv','all_seed_paired_effects.csv','all_seed_budget_interaction.csv','three_seed_paired_summary.csv','three_seed_paired_summary.json','three_seed_state_budget_deltas.png','three_seed_state_budget_deltas.pdf']+['s'+str(seed)+'/' for seed in SEEDS],limitations=['Three fixed model seeds, each with its own NOAS30 clean-correct cohort.','Channel draws repeat the same source physical packets; n is not the model-seed count.','Sample SD uses the three paired model-seed deltas, never packet/draw pseudo-replicates.','State associations do not identify a unique physical cause or counterfactual effect.','No formal target files, IQ, checkpoint, model inference, optimizer or deployment changes.','All original bins, empty/small cells, phase/AGC/boundary observation limitations inherit source_states56.'])
 core.write(out/'report.json',report);return report

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--code-commit',required=True);args=parser.parse_args()
 try:result=run(args.project,args.out,args.code_commit)
 except SourceNotReady as exc:
  print(json.dumps(dict(status='AWAITING_SOURCE_ARTIFACTS',reason=str(exc),NN=0,IQ_reads=0,checkpoint_reads=0,formal_files_read=False,automatic_retry=False)),flush=True);raise SystemExit(75)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
