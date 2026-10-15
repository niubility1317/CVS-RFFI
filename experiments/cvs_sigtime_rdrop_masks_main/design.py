from pathlib import Path
from experiments.cvs_sigtime_rdrop_legacy import design as native
from experiments.cvs_sigtime_rdrop_masks20 import design as screen
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(native,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01';RELEASE='cvs_SIGTIME_rdrop_paired_masks_main20_50_20261015_r01';OWNER='codex/root/SIGTIME-RDrop-paired-masks-main-20261015'
BASE=Path(PROJECT)/'runs'/RUN;BUDGETS=(20,50);MODES=('ANTI_CE','ANTI_RDROP');ARMS=tuple(f'E{e}_{m}' for e in BUDGETS for m in MODES);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
MODEL_RELEASE,MODEL_COMMIT=screen.MODEL_RELEASE,screen.MODEL_COMMIT
MASK_RELEASE=PROJECT+'/releases/'+screen.RELEASE;MASK_COMMIT='c803f2b13e74bd6d721c3795e64cb1c12ec377cc';SCREEN_RUN=screen.RUN
DEPENDENCY_RELEASE,DEPENDENCY_COMMIT,FOLD_RELEASE=(getattr(screen,k) for k in ('DEPENDENCY_RELEASE','DEPENDENCY_COMMIT','FOLD_RELEASE'))
HISTORY_ORIGINAL=native.RUN;HISTORY_RECOVERY='20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02'
stages=native.stages
def rows():return [dict(row_id=f'E{e}_{m}-s{s}',arm=f'E{e}_{m}',mode=m,epochs=e,model_seed=s,coupling='ANTITHETIC') for s in SEEDS for e in BUDGETS for m in MODES]
def native_config(c):return native.config(next(r for r in native.rows() if r['epochs']==c['epochs'] and r['model_seed']==c['model_seed'] and r['mode']==('RDROP' if c['mode']=='ANTI_RDROP' else 'TWO_CE')))
def config(row):
 if row not in rows():raise ValueError('Fixed12 paired-mask main rows required')
 c=native_config(row);return dict(c,schema='SIGTIME_RDrop_paired_masks_main_v1',run_id=RUN,**row,updates=row['epochs']*222,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,baseline_reference='Reuse all12 matched historical TWO_CE/RDROP scored artifacts as independent-mask controls; no old weights loaded',source_falsification=None,
  training_recipe=f'Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint; full-source scratch{row["epochs"]} '+str(stages(row['epochs']))+' L/U; fixed antithetic masks',
  training_mechanism=dict(screen.config(screen.rows()[0])['training_mechanism'],arm=row['mode'],coupling='ANTITHETIC',alpha=1. if row['mode']=='ANTI_RDROP' else 0.,extra_L_identity_forwards=row['epochs']*222*128),
  source_selection=dict(status='PASS',run_id=SCREEN_RUN,model_code_commit=MASK_COMMIT,held_RX1_errors_original=192,held_RX1_errors_candidate=187,held_CE_difference=-.02963761151515998,error_interaction=-40,weights_inherited=False),
  budget_authorization='User full E50 three seeds and E20 budget comparison; only SIG_TIME+RDrop. Fixed12 new scratch models following preregistered source qualification, all12 historical independent-mask controls reused.',
  goal=dict(primary='E50_ANTI_RDROP',clean_three_seed_mean=.84,mean_gain_over_same_budget_RDROP_strictly_positive=True,positive_seeds_min=2,mean_error_interaction_strictly_negative=True,E20_budget_comparison=True,global_novelty_claim=False,target_feedback=False,goal_completed=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed12 full-source paired-mask matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=native.make_args(native_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if getattr(a,'source_fold_held_rx',None) is not None or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Full original source and own scratch required')
 return a
def historical_rows():
 result=[]
 for s in SEEDS:
  for e in BUDGETS:
   for m in ('TWO_CE','RDROP'):
    rid=f'E{e}_{m}-s{s}';recovered=e==50 and ((m=='RDROP' and s in SEEDS[1:]) or (m=='TWO_CE' and s==SEEDS[2]));run=HISTORY_RECOVERY if recovered else HISTORY_ORIGINAL
    result.append(dict(row_id=rid,arm=f'E{e}_{m}',mode=m,epochs=e,model_seed=s,run_id=run,root=(Path(PROJECT)/'runs'/run/rid).as_posix()))
 return result
def totals():
 return dict(new_models=12,new_epochs=420,new_optimizer_steps=93240,extra_L_identity_forwards=11934720,U_observations=2*3*(7+17)*56700,new_clean_decisions=2016000,new_metric_records=216,reused_models=12,reused_epochs=420,reused_clean_decisions=2016000,compared_rows=24)
def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_commit=MODEL_COMMIT,mask_release=MASK_RELEASE,mask_commit=MASK_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],historical_rows=historical_rows(),views=list(VIEWS),readouts=list(READOUTS),default_head_only=True)
