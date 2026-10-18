"""Only E21..30 scene probabilities change; reuse the exact original As20 parent."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old
from .sampler import SCENES as TRAIN_SCENES,URBAN,NOMINAL
ROOT=Path(__file__).resolve().parents[2];PROJECT=old.PROJECT;SOURCE=old.SOURCE
RUN='20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_urban_prior_E30_20261018_r01';OWNER='codex/root/urban-prior59';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
FULL_FP32_POLICY=old.FULL_FP32_POLICY;CLASSES=old.CLASSES;SCENES=old.SCENES;VIEWS=old.VIEWS;FORMAL_VIEWS=old.FORMAL_VIEWS;TRUTH=old.TRUTH;PANEL=old.PANEL;SOURCE_CACHE=old.SOURCE_CACHE
read,write=old.read,old.write
REFERENCE_RUN=old.RUN

def old_config():return old.config(next(r for r in old.rows() if r['arm']=='B01'),30)
def rows():return [dict(arm='URBAN',row_id='URBAN-s'+str(SEED),model_seed=SEED)]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('One seed, one new10 tail only; no scratch20 or E50')
 c=deepcopy(old_config());c.update(schema='FcrossRX_urban_prior59_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),parent_source=old_config()['parent_source'],checkpoint_sources=[old_config()['parent_source']+'/final_ssdg.pth'],model_initialization='exact original P43 B01 scratchAs20 student and ownEMA;reset optimizer/cos10',training_recipe='Unchanged original6L/fullsource/cleanretain/SupCon/RDrop/oldU;only newE21..30 scene prior .2/.4/.4',scene_prior=dict(scenes=list(TRAIN_SCENES),probabilities=list(URBAN),nominal_probabilities=list(NOMINAL),constant=True,adaptive=False,sampler='legacy categorical bucket plus private within-bucket uniform;explicit inverseCDF',q=None,filter=False,rejection=False,hardware_session='RX/day/eq;notTX',conditional_physical_stream_unchanged=True),baseline_reference='Reused exact P43 B01 and B00 FINAL30;no control retraining or formal rerender',benchmark_context='Source P58 fixed full-state budget associations;no target-based parameter selection')
 c.update(method='constant_scene_prior_tail',stage='FcrossRX_fullsource_urban_prior59',budget_authorization='One new own10 tail/2220 updates;reuse original B01 scratchAs20/4440 updates;no new scratch20 or E50',cost_claim='New10 L_NN1704960 and actual As142080;native teacher/validation/source-readout/formal/smoke counted separately',goal=dict(source_fullV_clean_vs_NoAs30_min_pp=-.1,source_panel_mid_low_mean_vs_As30_min_pp=.5,source_panel_low_vs_As30_min_pp=0.,source_FAIL_still_all7=True,no_automatic_expansion=True))
 return c

def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed all-source/defaulthead/6L/oldU/E30/one-scene-prior contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(old_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';return a
