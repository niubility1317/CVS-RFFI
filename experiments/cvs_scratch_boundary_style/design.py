"""Early representation experiment: random student, pretrained source-U peers."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_e20_boundary_style import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-scratch-boundary-style-manysig-m27-r01'
RELEASE='cvs_scratch_boundary_style_20261012_r01';OWNER='codex/root/scratch-boundary-style-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE','SSL','M_SHALLOW','M_MIDDLE','M_BOTH','L_REPLAY','RANDOM_STYLE','HARD_STYLE','JOINT');PRIMARY='JOINT'
PAIRS=[(a,b) for a,b in old.PAIRS if a in (*ARMS,'P0') and b in (*ARMS,'P0')]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
 from .provenance import teacher_path
 if row not in rows():raise ValueError('Unregistered scratch boundary/style row')
 c=deepcopy(old.config(row))
 c.update(schema='scratch_boundary_style_v1',run_id=RUN,**row,method='cvs_scratch_boundary_style',stage='phase1_scratch_boundary_style',
  output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,model_initialization='scratch',parent_checkpoint=None,
  checkpoint_sources=[teacher_path(seed) for seed in c['teacher']['seeds']],initialization_checkpoint_sources=[],scratch_scope='student_identity_only')
 c['teacher']['cache_root']=(BASE/'peer_cache').as_posix()
 c['evaluation']['primary_arm']=PRIMARY
 c['multilevel']['head_warmup_steps']=0
 c['multilevel']['gradient_budget_scope']='fixed 12-pair L-only own-margin calibration of initial random student; no current-batch adaptation'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Scratch boundary/style configuration differs')
 return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
