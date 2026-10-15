from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-rdrop-budget-e20e50-clean-manysig-m12-r01'
RELEASE='cvs_SIGTIME_rdrop_budget20_50_20261015_r01'
OWNER='codex/root/SIGTIME-rdrop-budget20-50-20261015'
DEPENDENCY_RELEASE=PROJECT+'/releases/'+parent.RELEASE
DEPENDENCY_COMMIT='3a5cff49091deef46cad5d787e5daea7b4c0dfb9'
FOLD_RELEASE=PROJECT+'/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
SOURCE_RUN='20261015-phase1-SIGTIME-rdrop-sourcefold-s2026092701-m2-r01'
BASE=Path(PROJECT)/'runs'/RUN
BUDGETS=(20,50);MODES=('TWO_CE','RDROP');ARMS=tuple('E'+str(e)+'_'+m for e in BUDGETS for m in MODES)
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);STEPS=222
def rows():return [dict(row_id=f'E{e}_{m}-s{seed}',arm=f'E{e}_{m}',mode=m,epochs=e,model_seed=seed) for seed in SEEDS for e in BUDGETS for m in MODES]
def parent_config(seed):return parent.config(next(r for r in parent.rows() if r['arm']=='BASE_SSL' and r['model_seed']==seed))
def stages(e):
 if e not in BUDGETS:raise ValueError('Only fixed E20/E50 budgets permitted')
 return (13,7) if e==20 else (33,17)
def config(row):
 if row not in rows():raise ValueError('Unregistered budget/arm/model-seed row')
 e=row['epochs'];label,pseudo=stages(e);updates=e*STEPS
 return dict(parent_config(row['model_seed']),schema='phase1_rdrop_budget20_50_v1',run_id=RUN,**row,
  launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,
  label_epochs=label,pseudo_epochs=pseudo,readouts=list(READOUTS),test_views=list(VIEWS),head_fit_updates=0,
  baseline_reference='Fresh matched two-clean-L CE within same budget/seed; no score or weight substitution',
  training_mechanism=dict(two_same_clean_L_stochastic_forwards=True,paired_CE='mean CE1 CE2, LS0',
   KL='0.5 mean sum (p-q)*(logp-logq), both differentiable',alpha=1. if row['mode']=='RDROP' else 0.,
   active_steps=[1,updates],original_native_dropout=True,extra_parameters=0,extra_optimizer_steps=0,
   extra_L_identity_forwards=updates*128,original_L_forward_packets=updates*128,
   U_recipe='Original oldgate+ownEMA,lambda_u0.16,lambda_ent0.01,unchanged hidden-TX Gaussian weak/strong',
   main_default_head_joint=True,teacher='Own scratch only, EMA0.999, no new loss on teacher',
   native_CE_log='CE1; pairedCE/KL logged separately; total adds0.5*(CE2-CE1)+alphaKL once'),
  training_recipe=f'Original full-five-RX SIG_TIME+CENTER25K/defaultcos; scratch{e}/LS0;{label}L+{pseudo}U oldgate+ownEMA;mean two clean L CE;all native dropout unchanged',
  source_selection=dict(run_id=SOURCE_RUN,criterion='Completed fixed RX1 source training: fewer errors and no higher CE vs matched TWO_CE',weight_inheritance=False,
   user_budget_override='User requests complete E50 three seeds and E20 budget comparison; no additional E20 source-seed qualification before this fixed matrix'),
  source_falsification=dict(health_report_only=True,final_training_L_CE_health_max=.5,no_target_candidate_selection=True),
  budget_authorization='Current user explicitly requests full E50 three seeds plus E20 comparison; both mechanisms at both budgets, 12 independent scratch models',
  cost_claim=f'{e}epochs/{updates}updates;{updates*128} extra L identity packet forwards;{pseudo*56700} U observations;168000 new clean decisions;no inherited training',
  goal=dict(E50_RDROP_clean_three_seed_mean=.84,E20_RDROP_budget_comparison=True,primary='E50_RDROP',goal_completed=False,target_feedback=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed budget/recipe/data/default-head matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(parent_config(c['model_seed']),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=c['epochs'];a.label_epochs=c['label_epochs'];a.pseudo_epochs=c['pseudo_epochs'];a.label_smoothing=0.
 if a.use_aug or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt or not a.use_ema_teacher or a.batch_size!=128:
  raise ValueError('Scratch clean-L, original own-EMA and L128 required')
 return a
def require_source(value):
 if value.get('run_id')!=SOURCE_RUN or value.get('status')!='SOURCE_MATRIX_COMPLETE' or value.get('actual_optimizer_steps')!=8880 or value.get('target_access') is not False:
  raise ValueError('Completed legal source training prerequisite required')
 rs=value['results'];a,b=(rs[k] for k in ('RDROP','TWO_CE'))
 if any(r['epoch']!=20 or r['actual_steps']!=4440 or r['head_fit_updates']!=0 or r['target_access'] or not r['source_health'] for r in rs.values()):raise ValueError('Original fixed source training boundary differs')
 a,b=(r['exports']['FINAL'] for r in (a,b))
 if a['count']!=1260 or b['count']!=1260 or a['errors']>=b['errors'] or a['mean_CE']>b['mean_CE'] or not value['qualification']['qualified']:
  raise ValueError('Fixed source error/CE prerequisite failed')
 return dict(source_run=SOURCE_RUN,source_qualified=True,checkpoint_inherited=False,user_E50_and_E20_budget_override=True,target_feedback=False)
def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],views=list(VIEWS),readouts=list(READOUTS),default_head_only=True)
def totals():return dict(models=12,epochs=sum(r['epochs'] for r in rows()),optimizer_steps=sum(r['epochs']*222 for r in rows()),extra_L_identity_forwards=sum(r['epochs']*222*128 for r in rows()),U_observations=sum(stages(r['epochs'])[1]*56700 for r in rows()),clean_decisions=12*168000,metric_records=12*18)
