"""One scratch E20 full-source raw-practical arm, fixed before target-score access."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_fullsource_concat22 import design as reference
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,FORMAL_VIEWS,TRUTH,CLASSES,SCENES,VIEWS,TRAIN_SCENES,RX,SEED,STEPS,UPDATES,SMOKE_SOURCE=(getattr(reference,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','FORMAL_VIEWS','TRUTH','CLASSES','SCENES','VIEWS','TRAIN_SCENES','RX','SEED','STEPS','UPDATES','SMOKE_SOURCE'))
RUN='20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_fullsource_rawpractical_E20_20261017_r01'
OWNER='codex/root/FcrossRX-fullsource-rawpractical-20261017'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=reference.RUN;REFERENCE_ROOT=reference.BASE/'B01-s2026092701'
def rows():return [dict(arm='RAW_PRAC',row_id='RAW_PRAC-s2026092701',model_seed=SEED)]
def parent_config():return reference.parent_config()
def config(row):
 if row not in rows():raise ValueError('One fixed full-source row only')
 c=deepcopy(reference.config(reference.rows()[1]));c.update(schema='FcrossRX_fullsource_rawpractical_E20_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference='Immutable full-source native B01 of '+REFERENCE_RUN+'; comparison only, never weight initialization',training_recipe='Identical full F+CROSS_RX/oldgate/ownEMA/E20/4L/halfweights; only satellite branch input bypasses common ordinary augmentation',input_route='clean=ordinary(original IQ);satellite=full practical(original IQ)',Ag=False)
 c['channel']['input']='original normalized source IQ before ordinary; receiver state recomputed normally'
 c['scientific_targets']=dict(urban_gain_pp=3.,low_gain_pp=5.,clean_drop_max_pp=.5,low_absolute_goal=.60,suburban_gap_exploratory_pp=2.,reference='fixed B01, no target feedback or selection',on_negative='Retain negative result; no probability/weight/scene/seed/budget rescue of this fixed full-source route')
 return c
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed full-source raw-input route/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=reference.make_args(reference.config(reference.rows()[1]),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
