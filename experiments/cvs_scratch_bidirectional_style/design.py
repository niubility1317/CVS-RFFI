"""Matched bounded-positive and bounded-bidirectional shallow/middle styles."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch_identity_style import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-scratch-bidirectional-style-manysig-m18-r01'
RELEASE='cvs_scratch_bidirectional_style_20261012_r01'
OWNER='codex/root/scratch-bidirectional-style-20261012'
BASE=Path(PROJECT)/'runs'/RUN
POLICIES=('DELAY','CAP','DELAYCAP')
ARMS=tuple(p+'_'+pool for p in POLICIES for pool in ('POS','BIDIR'))
PRIMARY='DELAYCAP_BIDIR'
PAIRS=[(p+'_BIDIR',p+'_POS') for p in POLICIES]+[(arm,'P0') for arm in ARMS]
REFERENCE_RUN=old.RUN
REFERENCE_PAIRS=[(arm,arm.rsplit('_',1)[0]+'_JOINT') for arm in ARMS]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Unregistered bidirectional-style row')
 policy,pool=row['arm'].rsplit('_',1)
 parent=next(r for r in old.rows() if r['arm']==policy+'_JOINT' and r['model_seed']==row['model_seed'])
 c=deepcopy(old.config(parent))
 c.update(schema='scratch_bidirectional_style_v1',run_id=RUN,**row,method='cvs_scratch_bidirectional_style',stage='phase1_scratch_bidirectional_style',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['teacher']['cache_root']=(BASE/'peer_cache').as_posix()
 c['multilevel'].update(style_transform='bounded_logstats_v1',style_pool=pool,
  style_bounds=dict(mean_in_own_sigma=1.0,log_scale_abs=0.6931471805599453))
 c['evaluation']['primary_arm']=PRIMARY
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Bidirectional-style configuration differs')
 return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
