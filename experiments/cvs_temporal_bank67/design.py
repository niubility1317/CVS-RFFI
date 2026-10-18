"""Replace batch-neighbor confirmation with the existing same-physical epoch bank."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_temporal_bank_E30_20261018_r01'
OWNER='codex/root/temporal-bank67'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=old.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('B00','B01')
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
LOCAL_NN=0
SMOKE_NN=8
read,write=old.read,old.write

def reference_config(arm='B01'):
 if arm not in ARMS:raise ValueError('Only corresponding P43 NoAs/As reference')
 return old.config(next(r for r in old.rows() if r['arm']==arm),30)
def old_config(arm='B01'):return reference_config(arm)
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only corresponding own20+10 total30')
 inherited=old_config(row['arm']);c=deepcopy(inherited)
 c.update(schema='FcrossRX_temporal_bank67_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),parent_source=inherited['parent_source'],
  checkpoint_sources=[inherited['parent_source']+'/final_ssdg.pth'],
  model_initialization='Strict corresponding P43 own20 student/ownEMA; reset AdamW/cos10/temporal cache',
  method='native_epoch_bank_instead_of_batch_neighbor',stage='FcrossRX_fullsource_temporal_bank67',
  training_recipe='Original sixL768/1.25/SupCon/RDrop/fullAs64/raw native U teacher unchanged; replace temporal confirmation only',
  temporal_mode='epoch_bank',temporal_min_streak=2,
  temporal_confirmation=dict(mode='epoch_bank',min_streak=2,threshold=.8,
   physical_key=['rx_i','day_i','eq_i','sig_i','base_index'],
   native_function='_temporal_bank_mask_tensor',new_bank_at_tail_start=True,
   same_epoch_visit_does_not_increment=True,U_TX_truth_read=False,
   confidence_domain_strong_rules_unchanged=True,loss_scale_unchanged=True,
   teacher_input='original raw U',student_input='original strong_augment(U,std=.015)',
   no_added_gate=True,extra_forward_NN=0,extra_teacher_training=0),
  baseline_reference='P43 B00/B01 FINAL30 reused; paired NoAs/As with same replacement, no teacher or loss change',
  cost_claim='Each tail2220 updates,L1704960/nativeU567000/nativeNN3162962; no additional NN',
  budget_authorization='Two corresponding own20+10 totals30; only source-qualified recipe may total50, no automatic seed expansion',
  benchmark_context='Existing epoch_bank algorithm previously tested in old reference E200/no-As; current fullsource F+fullAs matching use is untested',
  goal=dict(fullV_clean_vs_matchedNoAs_bank_min_pp=-.1,fullV_clean_vs_originalNoAs30_min_pp=-.1,
   midlow_vs_originalAs30_min_pp=.5,low_vs_originalAs30_min_pp=0.,
   E50_only_after_source_PASS=True,E50_total_not_additional=True,source_FAIL_still_all7=True,target_feedback=False))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed fullsource/defaulthead/sixL/one temporal replacement total30 differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(old_config(c['arm']),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 a.pseudo_temporal_mode='epoch_bank'
 if a.pseudo_temporal_bank_min_streak!=2 or a.pseudo_temporal_min_conf!=.8 or a.teacher_ckpt:raise ValueError('Only native temporal mode may change')
 return a
