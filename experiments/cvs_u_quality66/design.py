"""Same-mass allocation of extra full-practical hidden-U teaching; fixed model."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_U_quality_E30_20261018_r01'
OWNER='codex/root/U-quality66'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=old.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('MASS_UNIFORM','OBSERVATION_WEIGHTED')
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
TRAIN_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
U_SEED,DROPOUT_SEED,HARDWARE_SEED=20261018661,20261018662,20261016423
LOCAL_NN=3072 # Expected once-only bounded actual-F local two-arm check; actual result recorded separately.
read,write=old.read,old.write

def reference_config(arm='B01'):
 if arm not in ('B00','B01'):raise ValueError('Only legal P43 NoAs/As references')
 return old.config(next(r for r in old.rows() if r['arm']==arm),30)
def old_config(arm=None):return reference_config('B01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only own As20 plus fixed10; student total30')
 inherited=old_config();c=deepcopy(inherited);teacher=inherited['parent_source']+'/final_ssdg.pth'
 c.update(schema='FcrossRX_U_quality66_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),parent_source=inherited['parent_source'],
  checkpoint_sources=[teacher],teacher_sources={'B01':teacher},
  model_initialization='Strict own P43 B01 As20 student/ownEMA; reset AdamW/cos10/cache; same own20 frozenEMA raw teacher',
  method='same_mass_observation_allocation_extra_practical_U_KL',stage='FcrossRX_fullsource_U_quality66',
  training_recipe='Original sixL768/1.25L/SupCon.1/RDrop/nativeU unchanged; one extra fullAs U student and one frozen As20 rawU teacher',
  U_teaching=dict(tau=.95,max_weight=.16,ramp_epochs=5,temperature=1.,mode=row['arm'],
   normalization='whole subset mean(q*KL(teacher||student)); no selected-count normalization',
   U_subset='first min(128,native_U_batch); native256/tail124 unchanged; no confidence/label selection or padding',
   observation_weight='sigmoid((signal_to_added_noise_db-5)/5); explicit None=>0; all3train scenes; not total SNR or recoverability truth',
   teacher_weight='clip((teacher pmax-.95)/.05,0,1)',
   allocation='OBSERVATION_WEIGHTED:qT*qN; MASS_UNIFORM:qT*sum(qT*qN)/sum(qT); sumqT0=>0',
   extra_student_U_NN_expected=284120,extra_frozen_teacher_U_NN_expected=284120,
   all_rows_identical_extra_forwards=True,teacher_targets_stopgrad=True,
   source_global_disable=False,student_urban_confidence_gate=False,old_U_masks_unchanged=True,
   original_L_losses_unchanged=True,all_labeled_As_CE_retained=True,
   frozen_teacher_epochs=20,new_teacher_training_updates=0,
   frozen_teacher_storage='external training-only holder; never student Module/state/optimizer/deploy',
   nominal_extra_training_reference_parameters=900722,new_student_parameters=0,original_U_native_NN_expected=567000),
  scene_prior=dict(scenes=list(TRAIN_SCENES),probabilities=[1/3]*3,adaptive=False,state_filter=False),
  baseline_reference='P43 B00/B01 FINAL30 reused; two new rows exactly matched extra forward/cost/stream/teacher/q mass',
  budget_authorization='Two independent own As20+10 student rows total30; no extra teacher training or additional teacher parent; only source-qualified candidate can later total50 with matched controls',
  benchmark_context='Physical q historically failed for L transfer/margin; new tested location is extra U defaulthead KL, not new quality concept',
  goal=dict(fullV_clean_vs_NoAs30_min_pp=-.1,midlow_vs_As30_min_pp=.5,low_vs_As30_min_pp=0.,
   midlow_vs_mass_uniform_min_pp=.25,low_vs_mass_uniform_min_pp=0.,
   controls_not_promotion_candidates=True,source_FAIL_still_all7=True,target_feedback=False,
   E50_only_after_source_PASS=True,E50_total_not_additional=True))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed fullsource/defaulthead/sixL/oldU/oneAs20teacher/two allocation E30 contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(old_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 if a.teacher_ckpt:raise ValueError('Native U remains own EMA, no external override')
 return a
