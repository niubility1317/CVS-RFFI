"""Fixed-source teacher provenance for additional practical U teaching only."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02'
RELEASE='cvs_FcrossRX_clean_teacher_U_E30_20261018_r02'
OWNER='codex/root/clean-teacher65'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=old.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('ZERO_KL','EVOLVING_AS_EMA','FROZEN_AS20_EMA','FROZEN_NOAS20_EMA')
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
TRAIN_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
U_SEED,DROPOUT_SEED,HARDWARE_SEED=20261018441,20261018442,20261016423
LOCAL_NN=6656 # One local real-F four-arm synthetic; checked against actual counters.
read,write=old.read,old.write

def reference_config(arm='B01'):
 if arm not in ('B00','B01'):raise ValueError('Only corresponding legal P43 NoAs/As own20 references')
 return old.config(next(r for r in old.rows() if r['arm']==arm),30)

def old_config(arm='FROZEN_NOAS20_EMA'):
 if arm not in ARMS:raise ValueError('Only four registered U teacher controls')
 return reference_config('B01')

def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]

def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only own As20 plus fixed10; student total30')
 inherited=old_config(row['arm']);c=deepcopy(inherited)
 teachers={a:reference_config(r)['parent_source']+'/final_ssdg.pth' for a,r in [('FROZEN_AS20_EMA','B01'),('FROZEN_NOAS20_EMA','B00')]}
 c.update(schema='FcrossRX_clean_teacher65_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),parent_source=inherited['parent_source'],
  checkpoint_sources=list(dict.fromkeys([inherited['parent_source']+'/final_ssdg.pth',*teachers.values()])),
  teacher_sources=teachers,
  model_initialization='Strict own P43 B01 As20 student/ownEMA; reset AdamW/cos10/cache; two fixed teacher EMA states from legal same-seed all-source20',
  method='full_practical_U_fixed_same_age_clean_teacher_origin',stage='FcrossRX_fullsource_clean_teacher65',
  training_recipe='Original sixL768/1.25L/SupCon.1/RDrop/nativeU unchanged; one additional fullAsU student and two frozen rawU teachers, all4rows same subset',
  U_teaching=dict(tau=.95,max_weight=.16,ramp_epochs=5,temperature=1.,
     mode=row['arm'],normalization='subset batchmean q*KL(teacher||student); not positive-q count',
     U_subset='first min(128,native_U_batch); native256/tail124 unchanged; no label/confidence selection or padding',
     extra_student_U_NN_expected=284120,extra_frozen_teacher_U_NN_expected=568240,
     all_rows_identical_extra_forwards=True,teacher_targets_stopgrad=True,
     quality_rule='clip((teacher pmax-.95)/.05,0,1); teacher-specific confidence is part of method',
     source_global_disable=False,student_urban_confidence_gate=False,old_U_masks_unchanged=True,
     original_L_losses_unchanged=True,all_labeled_As_CE_retained=True,
     frozen_teacher_epochs=20,new_teacher_training_updates=0,
     frozen_teacher_storage='external training-only holder, not student Module/state_dict/optimizer; no deployment dependency',
     nominal_extra_training_reference_parameters=2*900722,new_student_parameters=0,
     original_U_native_NN_expected=567000),
  scene_prior=dict(scenes=list(TRAIN_SCENES),probabilities=[1/3]*3,adaptive=False,state_filter=False),
  baseline_reference='Reuse frozen P43 B00/B01 FINAL30; new ZERO_KL matches all additional U/frozen-teacher forwards',
  budget_authorization='Four independent ownAs20+10 student rows, each cumulative30; reusedNoAs20 teacher is an additional historical20 prerequisite, not new training; report system50 prerequisite+new and actual extraNN',
  benchmark_context='New teacher-origin test with same-age frozenAs control, not a rescue or exact rerun of historical P39',
  goal=dict(source_FC_fullV_clean_vs_NoAs30_min_pp=-.1,source_FC_midlow_vs_As30_min_pp=.5,
   source_FC_low_vs_As30_min_pp=0.,source_FC_midlow_vs_FA_min_pp=.5,
   source_FC_midlow_vs_ZERO_min_pp=.5,source_FC_low_vs_FA_min_pp=0.,
   controls_not_promotion_candidates=True,source_FAIL_still_all7=True,no_automatic_expansion=True))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed all-source/sixL/default-head/oldU/fourteacher/sourcehidden E30 contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(old_config(c['arm']),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 if a.teacher_ckpt:raise ValueError('Native U teacher must remain own EMA, never external override')
 return a
