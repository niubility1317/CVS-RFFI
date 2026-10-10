"""Three fixed scratch seeds: legacy pseudo gating and cosine, no satellite augmentation."""
from pathlib import Path
from experiments.cvs_phase1_repair import design as parent
from experiments.cvs_phase1_stack.source import require_budget
from experiments.cvs_phase1_repair.design import PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write,cosine_lr
ROOT=Path(__file__).resolve().parents[2]
RUN='20261010-phase1-legacy-cosine-no-sat-manysig-m3-r01'
RELEASE='cvs_legacy_no_sat_20261010_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/legacy-no-sat-20261010'
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('legacy_cosine_no_sat',)
def rows():return [dict(stage='repair',arm=ARMS[0],model_seed=s,row_id=ARMS[0]+'-s'+str(s)) for s in SEEDS]
def config(row):
 c=parent.config(dict(stage='repair',arm='pseudo_batch_cosine',model_seed=row['model_seed'],row_id='pseudo_batch_cosine-s'+str(row['model_seed'])))
 c.update(schema='legacy_no_sat_v1',run_id=RUN,**row,features=['ema','pseudo','twostage'],method='cvs_legacy_no_sat',output_root=(BASE/row['row_id']/'source').as_posix())
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or config(row)!=c:raise ValueError('Changed preregistered configuration')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.args_from_validated_config(c,device)
 a.pseudo_domain_gate=False;a.pseudo_temporal_mode='batch_neighbor';a.pseudo_temporal_bank_min_streak=2
 a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
 return a
