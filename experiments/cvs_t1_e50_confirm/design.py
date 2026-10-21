from copy import deepcopy
from pathlib import Path
from .plan import SEEDS,NEW_SEEDS,profile
from experiments.cvs_t1_e50 import design as old
from experiments.cvs_same_as_kd_confirm86 import design as parents

ROOT=Path(__file__).resolve().parents[2]
RUN='20261021-phase1-T1-AsKL0-e30plus20-m3-r01'
RELEASE='cvs_T1_AsKL0_E30plus20_m3_20261021_r01'
OWNER='codex/root/T1-e50-three-seed'
PROJECT=old.PROJECT;BASE=Path(PROJECT)/'runs'/RUN
for key in ('SOURCE','FULL_FP32_POLICY','CLASSES','SCENES','VIEWS','FORMAL_VIEWS','TRUTH','RX','SOURCE_CACHE','GPU_IDS','MAX_TOTAL_GPU_PROCESSES'):
 globals()[key]=deepcopy(getattr(old,key))
read,write=old.read,old.write
TEMPLATE=old.config(next(r for r in old.rows() if r['arm']=='E30PLUS20'))

def rows():return [dict(arm='E30PLUS20',row_id='E30PLUS20-s'+str(seed),model_seed=seed) for seed in SEEDS]
def is_reused(row):return row['model_seed']==SEEDS[0]
def row_root(row):
 r=next(r for r in rows() if r['row_id']==row)
 return old.row_root(row) if is_reused(r) else BASE/row
def parent_config(seed):
 return parents.config(next(r for r in parents.rows() if r['arm']=='T1' and r['model_seed']==seed))
def parent_source(seed):return Path(parent_config(seed)['output_root'])
def reference_config(segment):return old.reference_config(segment)

def config(row):
 if row not in rows():raise ValueError('Only the fixed three continuation rows')
 if is_reused(row):return deepcopy(TEMPLATE)
 c=deepcopy(TEMPLATE);c.update(profile(row['model_seed']))
 parent=parent_source(row['model_seed'])
 c.update(schema='T1_E30plus20_three_seed_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(row_root(row['row_id'])/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],
  budget_authorization='User 2026-10-21 requests three seeds;reuse completed01,new02/03 each ownE30+20',
  source_selection='Fixed FINAL50;automatic clean+sixpractical truth-last;V health only',
  baseline_reference='Same-seed original T1 E30 student/ownEMA;no KD,not a scratch50 extension',
  benchmark_context='Human-requested seed extension after seed01 target comparison;descriptive exposed-benchmark replication,not fresh blind confirmation')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed seed,parent,source,head,budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 if c['model_seed'] not in NEW_SEEDS:raise ValueError('Completed seed01 must not be trained again')
 a=old.make_args(TEMPLATE,device)
 a.seed=c['model_seed'];a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 a.epochs=20;a.label_epochs=0;a.pseudo_epochs=20;a.from_scratch=False;a.a1_scratch_only=False
 return a
