"""One interaction: shared propagation and within-environment As discrimination."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_resample75 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261019-phase1-FcrossRX-propContrast-e30-s2026092701-m4-r01'
RELEASE='cvs_FcrossRX_propContrast_E30_20261019_r01'
OWNER='codex/root/prop-contrast76'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=old.REFERENCE_RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('C00','C10','C01','C11')
LOCAL_NN,SMOKE_NN=0,3076
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
TRAIN_SCENES=old.TRAIN_SCENES
read,write=old.read,old.write

def rows():
 return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED,share_propagation=a[1]=='1',As_contrast=a[2]=='1') for a in ARMS]

def old_config():return old.config(old.rows()[0])
def reference_config(arm='AS_CE'):return old.reference_config(arm)

def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Four fixed same-parent20+10 interaction rows only')
 c=deepcopy(old_config());c.pop('noise_pair',None)
 c.update(schema='FcrossRX_prop_contrast76_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),method=row['arm'],stage='FcrossRX_prop_contrast76',
  training_recipe='Original sixL768 CE1.25/ORD128 SupCon.1/ordinary-raw KL/nativeU;repeat same As IQ for both Dropout passes',
  model_initialization='Strict shared SAT18 AS_CE20 student/ownEMA;reset AdamW/cos10/CENTER_UNITS',
  baseline_reference='New C00/C10/C01 controls; P43 NoAs30/As30 reused whole-recipe references, not same-parent NoAs',
  prop_contrast=dict(share_propagation=row['share_propagation'],As_contrast=row['As_contrast'],
   group='RX/day/eq only;never TX;refresh every cumulative epoch/step',scene_grouped_in_all_arms=True,
   train_scenes=list(TRAIN_SCENES),channel_seed=20261016421,receiver_seed=20261016423,
   propagation_fields=['geometry','probs','state','z','ray_hz','ray_coef','f_doppler'],
   packet_private=['noise','phase','sync','tracking','equalizer'],hardware='Stable exact RX/day/eq session object',
   original_namespace='FcrossRX/train/e{epoch}/step{step}/L_S',
   source='Same raw L IQ;native normalize_input then complete residual process;not As(As(x))',
   receiver='Each packet native quality/compensation/AGC;white-input proxy may naturally give same lock in common H',
   scene_prior_changed=False,strength_changed=False,quality_rejection=False,
   L_NN_per_step=768,As_NN_per_step=128,native_waveforms_per_step=64,second_IQ='copy exact first received IQ',
   ordinary_SupCon_coefficient=.1,As_SupCon_coefficient=.05 if row['As_contrast'] else 0.,
   As_SupCon_levels=['time_down','t2.pool'],As_SupCon_temperature=.2,
   As_positive='same TX,different physical RX;any day',As_negative='different TX,same RX/day/eq only',
   As_same_TX_same_RX='excluded',As_invalid_anchor='zero auxiliary term;all labeled CE still retained',
   As_descriptor='per-As64 mean/std temporal maps;detached batch standardization then L2,original formula',
   As_pairs='each existing mixed forward independently;mean two depths and two passes;no cross-view KL',
   As_KL=False,total_As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,total_L_CE_scale=1.25,
   extra_NN=0,extra_parameters=0,extra_heads=0,U_changed=False,formal_test_changed=False),
  history_boundary='P20 shared propagation with ordinary-only SupCon;P41 SAT192 and mixedContrast used independent H;this exact environment-conditioned interaction not previously run',
  budget_authorization='Four fixed commonAS_CE20+10 tails,totalE30/6660 each;one modelseed;not E50',
  goal=dict(candidate='C11',fullV_clean_vs_NoAs_min_pp=-.1,
   pooled_midlow_vs_max_all_three_controls_and_originalAs_min_pp=.5,
   pooled_midlow_interaction_min_pp=.5,source_FAIL_still_all7=True,
   E50_only_after_source_PASS=True,no_weight_seed_or_condition_rescue=True,target_feedback=False))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed four-cell/common-scene/sharedH/As loss contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(old_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 return a
