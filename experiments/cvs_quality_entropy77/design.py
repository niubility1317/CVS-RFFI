"""One association: physical added-noise fraction and As target entropy."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_prop_contrast76 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261019-phase1-FcrossRX-qualityEntropy-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_qualityEntropy_E30_20261019_r01'
OWNER='codex/root/quality-entropy77'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=old.REFERENCE_RUN
CONTROL_RUN=old.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('R','Q')
LOCAL_NN,SMOKE_NN=0,1540
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
TRAIN_SCENES=old.TRAIN_SCENES
read,write=old.read,old.write

def rows():
 return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED,share_propagation=False,As_contrast=False,epsilon_assignment='permuted' if a=='R' else 'physical') for a in ARMS]

def control_config():return old.config(old.rows()[0])
def old_config():return old.old_config()
def reference_config(arm='AS_CE'):return old.reference_config(arm)

def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only registered Q/R same-parent20+10 rows')
 c=deepcopy(control_config());c.pop('prop_contrast',None)
 c.update(schema='FcrossRX_quality_entropy77_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),method=row['arm'],stage='FcrossRX_quality_entropy77',
  training_recipe='Same P76 C00 sixL768/CE1.25/ORD128 contrast.1/nativeU;only As64 CE label smoothing changes',
  baseline_reference='P76 C00 same-parent/common-scene independentH CE01 reference reused;P43NoAs/As whole recipe refs',
  quality_entropy=dict(epsilon_min=.01,epsilon_max=.20,formula='.01+.19/(1+10**(native_signal_to_added_noise_db/10))',
   proxy='Native actual waveform signal_to_added_noise_db;not original recording total SNR or white-input lock proxy',
   native_None_boundary='If native ratio=None and measured added noise>0:zero signal epsilon.20;added noise0:epsilon.01;invalid or absent evidence fails;all CE retained',
   assignment=row['epsilon_assignment'],permutation='Dedicated label-independent stable stream per cumulative epoch/step;exact same batch64 epsilon multiset',
   same_epsilon_both_dropout_passes=True,epsilon_not_target_or_TX_based=True,
   train_scenes=list(TRAIN_SCENES),channel_seed=20261016421,receiver_seed=20261016423,
   same_IQ_and_channel_stream_as_P76_C00=True,share_propagation=False,As_contrast=False,
   source='raw L IQ to original normalize_input then complete practical residual;second IQ exact copy',
   scene_group='RX/day/eq as frozen P76C00;all scenes/state support;not TX',
   physical_channel_changed=False,scene_prior_changed=False,quality_rejection=False,
   L_NN_per_step=768,As_NN_per_step=128,native_waveforms_per_step=64,
   ordinary_SupCon_coefficient=.1,extra_SupCon=0,As_KL=False,total_As_CE_coefficient=.25,
   per_pass_As_CE_coefficient=.125,total_L_CE_scale=1.25,all_labeled_CE_retained=True,
   ordinary_raw_objectives_unchanged=True,U_changed=False,extra_NN=0,extra_parameters=0,extra_heads=0,formal_test_changed=False),
  history_boundary='Fixed AsLS=.05 failed;quality-risk reweights losses. Here Q-R isolates physical epsilon association at exact same epsilon multiset;not presumed beneficial.',
  budget_authorization='Two commonAS_CE20+10 tails,totalE30/6660 each;one modelseed;not E50',
  goal=dict(candidate='Q',fullV_clean_vs_NoAs_min_pp=-.1,pooled_midlow_vs_max_R_C00_originalAs_min_pp=.5,
   source_FAIL_still_all7=True,E50_only_after_source_PASS=True,no_epsilon_seed_or_condition_rescue=True,target_feedback=False))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed two-row physical epsilon association contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(control_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 return a
