from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CLASSES,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-RX-teacher-exposure-source-e20-s2026092701-m2-r01'
RELEASE='cvs_RX_teacher_exposure_source20_20261015_r01'
OWNER='codex/root/RX-teacher-exposure-source20-20261015'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;SUBSET_SEED=2026101501;HELD=1;FIT=(3,4,6,8);RX=(1,3,4,6,8);ARMS=('SAMPLE_80','EXCLUDE_RX1');STEPS=222;UPDATES=4440
DEPENDENCY_RELEASE=PROJECT+'/releases/'+parent.RELEASE
DEPENDENCY_COMMIT='3a5cff49091deef46cad5d787e5daea7b4c0dfb9'
FOLD_RELEASE=PROJECT+'/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
def base_config():return parent.config(next(r for r in parent.rows() if r['arm']=='BASE_CE' and r['model_seed']==SEED))
def rows():return [dict(row_id=arm+'-s'+str(SEED),arm=arm,model_seed=SEED) for arm in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only fixed RX1 two-teacher source falsification permitted')
 return dict(base_config(),schema='phase1_RX_teacher_exposure_source20_v1',run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,
  baseline_reuse=False,readouts=[],test_views=[],head_fit_updates=0,
  training_recipe='scratch20 pure clean L CE/LS0/original AdamW+cos/native dropout;SIG_TIME+CENTER25K/defaulthead joint;no U iteration/EMA/augmentation',
  teacher_exposure=dict(held_RX=HELD,train_RX=list(RX if row['arm']=='SAMPLE_80' else FIT),train_L=5040,
   per_TX_day=280,per_RX_TX_day=56 if row['arm']=='SAMPLE_80' else 70,subset_seed=SUBSET_SEED,
   all_ancestors_exclude_held_RX=row['arm']=='EXCLUDE_RX1',initialization='scratch',U_training_packets=0,EMA=False,
   native_training_steps=4440,L_training_forward_packets=568320,fit_validation_RX=list(FIT),fit_V=21600,
   held_RX1_V=5400,held_V_first_forward='after final20 freeze',held_V_train_updates=0),
  source_selection=dict(primary='EXCLUDE_RX1',control='SAMPLE_80',fixed_final=True,
   qualify='Both source healthy; EXCLUDE_RX1 fewer held-V errors and CE no higher than SAMPLE_80, and net correction positive in frozen original A0 gate-rejected held-V region',
   no_threshold_scan=True,no_teacher_bank_or_student=True),
  goal=dict(source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False),
  budget_authorization='Active user cross-RX teacher report requests two equal-budget teachers first; fixed single seed source-only E20, no full bank/student or target; current RDrop12 budgets unchanged')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Teacher exposure/budget/defaulthead/permissions differ')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(base_config(),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=20;a.label_epochs=20;a.pseudo_epochs=0;a.label_smoothing=0.
 for key in vars(a):
  if key.startswith('lambda_'):setattr(a,key,0.)
 if a.use_unlabeled or a.use_ema_teacher or a.use_aug or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Pure CE scratch/no U/no EMA/no training augmentation required')
 return a
def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True,target_access=False)
