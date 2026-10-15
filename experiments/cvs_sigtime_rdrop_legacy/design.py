from pathlib import Path
from experiments.cvs_feature_sequence50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01'
RELEASE='cvs_originalSIGTIME_rdrop20_50_20261015_r01'
OWNER='codex/root/original-SIGTIME-RDrop-20261015'
DEPENDENCY_RELEASE=PROJECT+'/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02'
DEPENDENCY_COMMIT='3a5cff49091deef46cad5d787e5daea7b4c0dfb9'
FOLD_RELEASE=PROJECT+'/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
BASE=Path(PROJECT)/'runs'/RUN;BUDGETS=(20,50);MODES=('ORIGINAL','TWO_CE','RDROP');ARMS=tuple(f'E{e}_{m}' for e in BUDGETS for m in MODES)
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);STEPS=222
def legacy_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='SIG_TIME' and r['model_seed']==seed))
def stages(e):
 if e not in BUDGETS:raise ValueError('Only user requested E20 and E50 budgets permitted')
 return (13,7) if e==20 else (33,17)
def rows():return [dict(row_id=f'E{e}_{m}-s{s}',arm=f'E{e}_{m}',mode=m,epochs=e,model_seed=s) for s in SEEDS for e in BUDGETS for m in MODES]
def config(row):
 if row not in rows():raise ValueError('Unregistered original SIG_TIME RDrop row')
 e=row['epochs'];label,pseudo=stages(e);paired=row['mode']!='ORIGINAL';alpha=1. if row['mode']=='RDROP' else 0.
 return dict(legacy_config(row['model_seed']),schema='original_SIGTIME_RDrop_matched_budget_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,baseline_reference='Fresh same-recipe ORIGINAL; historical83.0393% is unpaired context only',
  label_epochs=label,pseudo_epochs=pseudo,readouts=list(READOUTS),test_views=list(VIEWS),head_fit_updates=0,
  training_recipe=f'Original historical SIG_TIME/RAW_HZ/original normal augmentation/LS.01/defaultcos joint; scratch{e}/{label}L+{pseudo}U oldgate+ownEMA; mode='+row['mode'],
  training_mechanism=dict(mode=row['mode'],alpha=alpha,label_smoothing=.01,use_aug=True,sinc_coordinates='RAW_HZ',
   input='Capture actual augmented native L forward input inside L-only context; second forward reuses identical augmented IQ, never pre-augmentation raw batch',
   objective='Original smoothed CE1' if not paired else '.5*(smoothedCE1+smoothedCE2)+alpha*symKL(original logits1,logits2)',
   hook_correction='0' if not paired else '.5*(smoothedCE2-native_identity_loss)+alpha*symKL',
   default_W_joint=True,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=e*222*128 if paired else 0,
   original_L_identity_forwards=e*222*128,original_hidden_U_recipe_unchanged=True,no_new_U_consistency=True,augmentation_redraws_in_second_forward=0),
  source_falsification=dict(first_seed=SEEDS[0],per_budget_independent=True,final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02,minimum_positive_KL_call_fraction=.01,
   selection='Per budget, first seed all3 full source; RDrop healthy and V no lower than ORIGINAL/TWO_CE within1pp mean/2pp worstRX; finite complete signal, no target metrics. PASS releases6 remaining rows of that budget; FAIL preserves first3 and other budget continues'),
  checkpoint_selection='final_only',separate_head_training=False,
  budget_authorization='User explicitly asks original SIG_TIME plus organically combined RDrop published for verification; prior full E50 three-seed and E20 comparison retained. Fixed18 scratch models, no old weights inherited.',
  goal=dict(primary='E50_RDROP',clean_three_seed_mean=.84,E20_budget_comparison=True,legacy_SIGTIME_recipe=True,goal_completed=False,target_feedback=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed original SIG_TIME recipe/budget/matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(legacy_config(c['model_seed']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=c['epochs'];a.label_epochs=c['label_epochs'];a.pseudo_epochs=c['pseudo_epochs']
 if a.label_smoothing!=.01 or not a.use_aug or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt or not a.use_ema_teacher or a.batch_size!=128:raise ValueError('Original augmentation/LS.01/scratch/ownEMA/defaultL128 recipe required')
 return a
def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],views=list(VIEWS),readouts=list(READOUTS),default_head_only=True)
def totals():return dict(models=18,epochs=sum(r['epochs'] for r in rows()),optimizer_steps=sum(r['epochs']*222 for r in rows()),extra_L_identity_forwards=sum(r['epochs']*222*128 for r in rows() if r['mode']!='ORIGINAL'),U_observations=sum(stages(r['epochs'])[1]*56700 for r in rows()),clean_decisions=18*168000,metric_records=18*18)
